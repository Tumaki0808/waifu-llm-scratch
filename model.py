from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import Tensor, nn
import torch.nn.functional as F


@dataclass
class ModelConfig:
    vocab_size: int = 4096
    dim: int = 384
    n_layers: int = 8
    n_heads: int = 6
    n_kv_heads: int = 2
    max_seq_len: int = 512
    dropout: float = 0.05
    hidden_dim: int = 1024


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        return F.rms_norm(x, (x.shape[-1],), self.weight, self.eps)


class RoPE(nn.Module):
    def __init__(self, head_dim: int, max_seq_len: int) -> None:
        super().__init__()
        inv_freq = 1.0 / (10000 ** (torch.arange(0, head_dim, 2).float() / head_dim))
        positions = torch.arange(max_seq_len).float()
        freqs = torch.outer(positions, inv_freq)
        self.register_buffer("cos", freqs.cos()[None, None, :, :], persistent=False)
        self.register_buffer("sin", freqs.sin()[None, None, :, :], persistent=False)

    def forward(self, x: Tensor) -> Tensor:
        half = x.shape[-1] // 2
        x1, x2 = x[..., :half], x[..., half:]
        cos = self.cos[..., :x.shape[-2], :half]
        sin = self.sin[..., :x.shape[-2], :half]
        return torch.cat((x1 * cos - x2 * sin, x1 * sin + x2 * cos), dim=-1)


class Attention(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.n_heads, self.n_kv_heads = config.n_heads, config.n_kv_heads
        self.head_dim = config.dim // config.n_heads
        self.q = nn.Linear(config.dim, config.n_heads * self.head_dim, bias=False)
        self.k = nn.Linear(config.dim, config.n_kv_heads * self.head_dim, bias=False)
        self.v = nn.Linear(config.dim, config.n_kv_heads * self.head_dim, bias=False)
        self.out = nn.Linear(config.dim, config.dim, bias=False)
        self.rope = RoPE(self.head_dim, config.max_seq_len)
        self.dropout = config.dropout

    def forward(self, x: Tensor) -> Tensor:
        batch, seq, _ = x.shape
        q = self.rope(self.q(x).view(batch, seq, self.n_heads, self.head_dim).transpose(1, 2))
        k = self.rope(self.k(x).view(batch, seq, self.n_kv_heads, self.head_dim).transpose(1, 2))
        v = self.v(x).view(batch, seq, self.n_kv_heads, self.head_dim).transpose(1, 2)
        repeats = self.n_heads // self.n_kv_heads
        k, v = k.repeat_interleave(repeats, dim=1), v.repeat_interleave(repeats, dim=1)
        result = F.scaled_dot_product_attention(q, k, v, dropout_p=self.dropout if self.training else 0.0,
                                                is_causal=True)
        return self.out(result.transpose(1, 2).contiguous().view(batch, seq, -1))


class SwiGLU(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.gate = nn.Linear(config.dim, config.hidden_dim, bias=False)
        self.up = nn.Linear(config.dim, config.hidden_dim, bias=False)
        self.down = nn.Linear(config.hidden_dim, config.dim, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        return self.down(F.silu(self.gate(x)) * self.up(x))


class Block(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.attn_norm, self.ffn_norm = RMSNorm(config.dim), RMSNorm(config.dim)
        self.attn, self.ffn = Attention(config), SwiGLU(config)

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.attn(self.attn_norm(x))
        return x + self.ffn(self.ffn_norm(x))


class Llama3Mini(nn.Module):
    def __init__(self, config: ModelConfig | None = None) -> None:
        super().__init__()
        self.config = config or ModelConfig()
        self.tok_embeddings = nn.Embedding(self.config.vocab_size, self.config.dim)
        self.layers = nn.ModuleList([Block(self.config) for _ in range(self.config.n_layers)])
        self.norm = RMSNorm(self.config.dim)
        self.lm_head = nn.Linear(self.config.dim, self.config.vocab_size, bias=False)
        self.lm_head.weight = self.tok_embeddings.weight

    def forward(self, idx: Tensor) -> Tensor:
        x = self.tok_embeddings(idx)
        for layer in self.layers:
            x = layer(x)
        return self.lm_head(self.norm(x))

    @torch.no_grad()
    def generate(self, idx: Tensor, max_new_tokens: int = 120, temperature: float = 0.68,
                 top_p: float = 0.9, repetition_penalty: float = 1.18,
                 eos_token_id: int = 2) -> Tensor:
        self.eval()
        for _ in range(max_new_tokens):
            context = idx[:, -self.config.max_seq_len:]
            logits = self(context)[:, -1, :] / max(temperature, 1e-5)
            for token in torch.unique(idx):
                value = logits[:, token]
                logits[:, token] = torch.where(value > 0, value / repetition_penalty,
                                               value * repetition_penalty)
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            probabilities = F.softmax(sorted_logits, dim=-1)
            cumulative = torch.cumsum(probabilities, dim=-1)
            remove = cumulative - probabilities > top_p
            sorted_logits[remove] = float("-inf")
            next_token = sorted_indices.gather(-1, torch.multinomial(F.softmax(sorted_logits, dim=-1), 1))
            idx = torch.cat((idx, next_token), dim=1)
            if int(next_token.item()) in (eos_token_id, 3):
                break
        return idx
