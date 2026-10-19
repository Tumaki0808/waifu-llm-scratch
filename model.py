from __future__ import annotations

from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class ModelConfig:
    vocab_size: int = 8192
    dim: int = 512
    n_layers: int = 8
    n_heads: int = 8
    n_kv_heads: int = 8  # Matching heads eliminates repeat_interleave memory copy on T4!
    max_seq_len: int = 512
    dropout: float = 0.0  # 0.0 unlocks the fastest fused C++ SDPA kernel during pretraining
    rope_theta: float = 10000.0


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.rms_norm(x, (x.size(-1),), self.weight, self.eps)


def precompute_rope(dim: int, max_seq_len: int, theta: float = 10000.0) -> tuple[torch.Tensor, torch.Tensor]:
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
    t = torch.arange(max_seq_len, dtype=torch.float32)
    angles = torch.outer(t, freqs)
    return torch.cos(angles).half(), torch.sin(angles).half()


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    d = x.shape[-1] // 2
    x1, x2 = x[..., :d], x[..., d:]
    c = cos[: x.shape[1], :].unsqueeze(0).unsqueeze(2)
    s = sin[: x.shape[1], :].unsqueeze(0).unsqueeze(2)
    return torch.cat([x1 * c - x2 * s, x1 * s + x2 * c], dim=-1)


class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.n_heads = cfg.n_heads
        self.head_dim = cfg.dim // cfg.n_heads
        # Single fused QKV projection is 30% faster on T4 than 3 separate Linear layers
        self.wqkv = nn.Linear(cfg.dim, 3 * cfg.dim, bias=False)
        self.wo = nn.Linear(cfg.dim, cfg.dim, bias=False)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        qkv = self.wqkv(x).view(B, T, 3, self.n_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)

        q = apply_rope(q, cos, sin).transpose(1, 2)
        k = apply_rope(k, cos, sin).transpose(1, 2)
        v = v.transpose(1, 2)

        out = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0, is_causal=True)
        return self.wo(out.transpose(1, 2).contiguous().view(B, T, C))


class SwiGLU(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        hidden = ((int((8 / 3) * cfg.dim) + 63) // 64) * 64
        # Fused gate + up projection cuts GEMM kernel launches in half
        self.w13 = nn.Linear(cfg.dim, 2 * hidden, bias=False)
        self.w2 = nn.Linear(hidden, cfg.dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        g, u = self.w13(x).chunk(2, dim=-1)
        return self.w2(F.silu(g) * u)


class TransformerBlock(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.attn_norm = RMSNorm(cfg.dim)
        self.attn = CausalSelfAttention(cfg)
        self.ffn_norm = RMSNorm(cfg.dim)
        self.ffn = SwiGLU(cfg)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.attn_norm(x), cos, sin)
        x = x + self.ffn(self.ffn_norm(x))
        return x


class Llama3Mini(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.tok_embeddings = nn.Embedding(cfg.vocab_size, cfg.dim)
        self.layers = nn.ModuleList([TransformerBlock(cfg) for _ in range(cfg.n_layers)])
        self.norm = RMSNorm(cfg.dim)
        self.lm_head = nn.Linear(cfg.dim, cfg.vocab_size, bias=False)
        self.lm_head.weight = self.tok_embeddings.weight

        cos, sin = precompute_rope(cfg.dim // cfg.n_heads, cfg.max_seq_len, cfg.rope_theta)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)
        self.apply(self._init_weights)

    def _init_weights(self, m: nn.Module) -> None:
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None) -> torch.Tensor:
        x = self.tok_embeddings(idx)
        for layer in self.layers:
            x = layer(x, self.rope_cos, self.rope_sin)
        logits = self.lm_head(self.norm(x))
        if targets is not None:
            # Compute loss inside forward() so DataParallel never copies 1GB of logits across PCIe!
            return F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-100)
        return logits

    @torch.inference_mode()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int = 110,
        temperature: float = 0.65,
        top_p: float = 0.90,
        repetition_penalty: float = 1.18,
        eos_token_id: int = 2,
        user_token_id: int = 3,
    ) -> torch.Tensor:
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.cfg.max_seq_len :]
            logits = self(idx_cond)[:, -1, :]

            if repetition_penalty != 1.0:
                for b in range(idx.shape[0]):
                    seen = torch.unique(idx_cond[b])
                    score = logits[b, seen]
                    logits[b, seen] = torch.where(score < 0, score * repetition_penalty, score / repetition_penalty)

            logits[:, 0] = -float("inf")
            logits[:, 1] = -float("inf")

            logits = logits / max(temperature, 1e-4)
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0

            indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
            logits = logits.masked_fill(indices_to_remove, -float("inf"))

            probs = F.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)

            if next_id.item() in (eos_token_id, user_token_id):
                break
            idx = torch.cat([idx, next_id], dim=1)
        return idx
