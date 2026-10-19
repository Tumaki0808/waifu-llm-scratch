from __future__ import annotations

import argparse
import math
import time
from pathlib import Path

import numpy as np
import torch
from tokenizers import Tokenizer
from torch.amp import GradScaler, autocast
from torch.optim import AdamW
from tqdm import trange

from model import Llama3Mini, ModelConfig

WINDOW = 512


def get_intelligence_stage(loss: float, phase: str) -> str:
    if loss > 4.5:
        return "LVL 1/6: Infant Babbling (Discovering syllables & spaces)"
    elif loss > 3.0:
        return "LVL 2/6: Early Words (Forming basic nouns & verbs, broken grammar)"
    elif loss > 2.2:
        return "LVL 3/6: Basic English (Clean short sentences, learning grammar rules)"
    elif loss > 1.7:
        return "LVL 4/6: Coherent English (Good grammar, understands cause & effect)"
    elif phase == "pretrain":
        return "LVL 5/6: Fluent Pretrained Base (Ready for Phase 2 Chat SFT!)"
    else:
        return "LVL 6/6: Conversational Companion (Full Persona & Natural English Chat!)"


class VRAMBatcher:
    """Stores the entire compact uint16/int16 dataset directly inside GPU VRAM for 0.0ms PCIe overhead."""

    def __init__(self, phase: str, device: torch.device):
        self.phase = phase
        self.device = device
        if phase == "pretrain":
            print("-> Pinning pretrain.bin directly into GPU VRAM...")
            raw = np.fromfile("pretrain.bin", dtype=np.uint16).astype(np.int32)
            self.tokens = torch.from_numpy(raw).to(device)
            self.offsets = torch.arange(WINDOW + 1, device=device, dtype=torch.long)
            self.max_start = len(self.tokens) - WINDOW - 2
            print(f"-> {len(self.tokens):,} tokens loaded directly inside VRAM!")
        else:
            print("-> Pinning sft_x.bin & sft_y.bin directly into GPU VRAM...")
            raw_x = np.fromfile("sft_x.bin", dtype=np.uint16).reshape(-1, WINDOW).astype(np.int32)
            raw_y = np.fromfile("sft_y.bin", dtype=np.int16).reshape(-1, WINDOW)
            self.sft_x = torch.from_numpy(raw_x).to(device)
            self.sft_y = torch.from_numpy(raw_y).to(device)
            self.num_rows = len(self.sft_x)
            print(f"-> {self.num_rows:,} SFT windows loaded directly inside VRAM!")

    def next_batch(self, batch_size: int) -> tuple[torch.Tensor, torch.Tensor]:
        if self.phase == "pretrain":
            starts = torch.randint(0, self.max_start, (batch_size, 1), device=self.device)
            chunk = self.tokens[starts + self.offsets].long()
            return chunk[:, :-1], chunk[:, 1:]
        else:
            idx = torch.randint(0, self.num_rows, (batch_size,), device=self.device)
            return self.sft_x[idx].long(), self.sft_y[idx].long()


def unwrap_model(model: torch.nn.Module) -> Llama3Mini:
    m = model
    if hasattr(m, "_orig_mod"):
        m = m._orig_mod
    if isinstance(m, torch.nn.DataParallel):
        m = m.module
    if hasattr(m, "_orig_mod"):
        m = m._orig_mod
    return m


def evaluate_talking_level(model: torch.nn.Module, tokenizer: Tokenizer, device: torch.device, step: int, loss: float, phase: str) -> None:
    base = unwrap_model(model)
    base.eval()
    stage = get_intelligence_stage(loss, phase)
    print("\n" + "=" * 72)
    print(f" [STEP {step} LIVE CHECK] Loss: {loss:.4f} | {stage}")
    print("-" * 72)

    for p in ("Hi! Who are you?", "I am really tired from college today.", "Can we talk for a little bit?"):
        ids = tokenizer.encode(f"<|bos|><|user|>{p}<|bot|>", add_special_tokens=False).ids
        x = torch.tensor([ids], dtype=torch.long, device=device)
        out = base.generate(x, max_new_tokens=75, temperature=0.65, top_p=0.9, repetition_penalty=1.18)
        reply = tokenizer.decode(out[0, len(ids) :].tolist(), skip_special_tokens=True).strip()
        print(f" USER : {p}")
        print(f" BOT  : {reply}\n")
    print("=" * 72 + "\n")


def main(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = Tokenizer.from_file("tokenizer.json")
    batcher = VRAMBatcher(args.phase, device)

    cfg = ModelConfig(vocab_size=tokenizer.get_vocab_size())
    base_model = Llama3Mini(cfg).to(device)
    print(f"-> Initialized Llama3Mini ({sum(p.numel() for p in base_model.parameters()) / 1e6:.1f}M params)")

    start_step = 0
    if args.resume and Path(args.resume).exists():
        ckpt = torch.load(args.resume, map_location=device)
        base_model.load_state_dict(ckpt["model"] if "model" in ckpt else ckpt)
        if ckpt.get("phase") == args.phase and int(ckpt.get("step", 0)) < args.max_steps:
            start_step = int(ckpt.get("step", 0))
            print(f"-> Resumed {args.phase} at step {start_step}")
        else:
            print(f"-> Loaded pretrained weights! Starting {args.phase.upper()} at step 0.")

    model: torch.nn.Module = base_model
    if torch.cuda.device_count() > 1 and not args.single_gpu:
        print(f"-> Using {torch.cuda.device_count()} GPUs with in-model scalar loss reduction!")
        model = torch.nn.DataParallel(base_model)
    elif hasattr(torch, "compile") and device.type == "cuda":
        print("-> Compiling fused GPU kernels with torch.compile (first step takes ~35s)...")
        model = torch.compile(base_model)

    optimizer = AdamW(base_model.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=0.1, fused=True)
    scaler = GradScaler("cuda", enabled=(device.type == "cuda"))

    warmup = max(1, int(args.max_steps * 0.05))
    best_loss = float("inf")
    Path("checkpoints").mkdir(exist_ok=True)

    started = None
    tokens_processed = 0

    for step in trange(start_step, args.max_steps, desc=f"Training ({args.phase})"):
        if step == start_step + 1:
            # Start timer after step 1 so one-time CUDA/compile warmup doesn't skew tok/s
            started = time.perf_counter()
            tokens_processed = 0

        model.train()
        optimizer.zero_grad(set_to_none=True)
        loss_accum = torch.zeros((), device=device)

        for _ in range(args.grad_accum):
            x, y = batcher.next_batch(args.batch_size)
            with autocast("cuda", dtype=torch.float16, enabled=(device.type == "cuda")):
                # Pass both x and y so each GPU returns a tiny scalar loss instead of 1GB of logits!
                loss = model(x, y).mean()
            scaler.scale(loss / args.grad_accum).backward()
            loss_accum += loss.detach()

        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(base_model.parameters(), 1.0)

        progress = step + 1
        scale = progress / warmup if progress <= warmup else max(0.1, 0.5 * (1.0 + math.cos(math.pi * (progress - warmup) / max(1, args.max_steps - warmup))))
        lr = args.lr * scale
        for group in optimizer.param_groups:
            group["lr"] = lr

        scaler.step(optimizer)
        scaler.update()

        if started is not None:
            tokens_processed += args.batch_size * args.grad_accum * WINDOW

        if progress % 50 == 0 and started is not None:
            avg_loss = (loss_accum / args.grad_accum).item()
            tok_sec = tokens_processed / max(1e-6, time.perf_counter() - started)
            print(f"Step {progress:4d}/{args.max_steps} | Loss: {avg_loss:.4f} | Ppl: {math.exp(min(avg_loss, 15)):6.2f} | LR: {lr:.2e} | Speed: {tok_sec:,.0f} tok/s")

        if progress % 500 == 0 or progress == args.max_steps:
            avg_loss = (loss_accum / args.grad_accum).item()
            evaluate_talking_level(model, tokenizer, device, progress, avg_loss, args.phase)
            state = {"model": base_model.state_dict(), "step": progress, "phase": args.phase, "config": cfg.__dict__}
            torch.save(state, f"checkpoints/{args.phase}_latest.pt")
            if avg_loss < best_loss:
                best_loss = avg_loss
                torch.save(state, f"checkpoints/{args.phase}_best.pt")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("pretrain", "sft"), required=True)
    parser.add_argument("--max_steps", type=int, required=True)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--grad_accum", type=int, default=1)
    parser.add_argument("--lr", type=float, default=8e-4)
    parser.add_argument("--resume", default="")
    parser.add_argument("--single_gpu", action="store_true")
    main(parser.parse_args())
