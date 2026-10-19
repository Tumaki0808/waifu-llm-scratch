from __future__ import annotations

import argparse
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
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


class FastGPUBatcher:
    """Zero-DataLoader vectorized tensor batcher. Slices 128 sequences in <0.2ms with zero Kaggle hangs."""

    def __init__(self, phase: str, device: torch.device):
        self.phase = phase
        self.device = device
        if phase == "pretrain":
            print("-> Loading pretrain.bin into RAM for vectorized slicing...")
            raw = np.fromfile("pretrain.bin", dtype=np.uint16)
            self.tokens = torch.from_numpy(raw.astype(np.int64))
            self.offsets = torch.arange(WINDOW + 1, dtype=torch.long)
            self.max_start = len(self.tokens) - WINDOW - 2
            print(f"-> Loaded {len(self.tokens):,} tokens into fast memory!")
        else:
            print("-> Loading pre-masked sft_x.bin & sft_y.bin into RAM...")
            raw_x = np.fromfile("sft_x.bin", dtype=np.uint16).reshape(-1, WINDOW)
            raw_y = np.fromfile("sft_y.bin", dtype=np.int16).reshape(-1, WINDOW)
            self.sft_x = torch.from_numpy(raw_x.astype(np.int64))
            self.sft_y = torch.from_numpy(raw_y.astype(np.int64))
            self.num_rows = len(self.sft_x)
            print(f"-> Loaded {self.num_rows:,} pre-masked SFT conversation windows!")

    def next_batch(self, batch_size: int) -> tuple[torch.Tensor, torch.Tensor]:
        if self.phase == "pretrain":
            starts = torch.randint(0, self.max_start, (batch_size, 1))
            chunk = self.tokens[starts + self.offsets]  # Vectorized C++ gather
            x = chunk[:, :-1].to(self.device, non_blocking=True)
            y = chunk[:, 1:].to(self.device, non_blocking=True)
            return x, y
        else:
            idx = torch.randint(0, self.num_rows, (batch_size,))
            x = self.sft_x[idx].to(self.device, non_blocking=True)
            y = self.sft_y[idx].to(self.device, non_blocking=True)
            return x, y


def unwrap_model(model: torch.nn.Module) -> Llama3Mini:
    m = model.module if isinstance(model, torch.nn.DataParallel) else model
    if hasattr(m, "_orig_mod"):
        m = m._orig_mod
    return m


def evaluate_talking_level(model: torch.nn.Module, tokenizer: Tokenizer, device: torch.device, step: int, loss: float, phase: str) -> None:
    base = unwrap_model(model)
    base.eval()
    stage = get_intelligence_stage(loss, phase)
    print("\n" + "=" * 72)
    print(f" [STEP {step} LIVE CHECK]Loss: {loss:.4f} | {stage}")
    print("-" * 72)

    test_prompts = [
        "Hi! Who are you?",
        "I am really tired from college today.",
        "Can we talk for a little bit?",
    ]
    for p in test_prompts:
        prompt_str = f"<|bos|><|user|>{p}<|bot|>"
        ids = tokenizer.encode(prompt_str, add_special_tokens=False).ids
        x = torch.tensor([ids], dtype=torch.long, device=device)
        out = base.generate(x, max_new_tokens=75, temperature=0.65, top_p=0.9, repetition_penalty=1.18)
        gen_ids = out[0, len(ids) :].tolist()
        reply = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
        print(f" USER : {p}")
        print(f" BOT  : {reply}\n")
    print("=" * 72 + "\n")


def main(args: argparse.Namespace) -> None:
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = Tokenizer.from_file("tokenizer.json")
    batcher = FastGPUBatcher(args.phase, device)

    cfg = ModelConfig(vocab_size=tokenizer.get_vocab_size())
    model: torch.nn.Module = Llama3Mini(cfg).to(device)
    num_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"-> Initialized Llama3Mini ({num_params:.1f}M parameters) on {device}")

    start_step = 0
    if args.resume and Path(args.resume).exists():
        print(f"-> Loading weights from {args.resume}...")
        ckpt = torch.load(args.resume, map_location=device)
        state_dict = ckpt["model"] if "model" in ckpt else ckpt
        unwrap_model(model).load_state_dict(state_dict)
        ckpt_phase = ckpt.get("phase", "pretrain")
        if ckpt_phase == args.phase and int(ckpt.get("step", 0)) < args.max_steps:
            start_step = int(ckpt.get("step", 0))
            print(f"-> Resuming same phase ({args.phase}) from step {start_step}")
        else:
            print(f"-> Loaded pretrained brain! Resetting step counter to 0 for {args.phase.upper()} phase.")
            start_step = 0

    if torch.cuda.device_count() > 1 and not args.single_gpu:
        print(f"-> Activating Dual-GPU DataParallel across {torch.cuda.device_count()} GPUs!")
        model = torch.nn.DataParallel(model)

    optimizer = AdamW(
        model.parameters(),
        lr=args.lr,
        betas=(0.9, 0.95),
        weight_decay=0.1,
        fused=(device.type == "cuda"),
    )
    scaler = GradScaler("cuda", enabled=(device.type == "cuda"))

    warmup = max(1, int(args.max_steps * 0.05))
    best_loss = float("inf")
    Path("checkpoints").mkdir(exist_ok=True)

    started = time.perf_counter()
    tokens_processed = 0

    for step in trange(start_step, args.max_steps, desc=f"Training ({args.phase})"):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        loss_accum = torch.zeros((), device=device)

        for _ in range(args.grad_accum):
            x, y = batcher.next_batch(args.batch_size)
            with autocast("cuda", dtype=torch.float16, enabled=(device.type == "cuda")):
                logits = model(x)
                loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1), ignore_index=-100)
            scaler.scale(loss / args.grad_accum).backward()
            loss_accum += loss.detach()

        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        progress = step + 1
        if progress <= warmup:
            scale = progress / warmup
        else:
            pct = (progress - warmup) / max(1, args.max_steps - warmup)
            scale = max(0.1, 0.5 * (1.0 + math.cos(math.pi * pct)))
        lr = args.lr * scale
        for group in optimizer.param_groups:
            group["lr"] = lr

        scaler.step(optimizer)
        scaler.update()

        tokens_processed += args.batch_size * args.grad_accum * WINDOW

        if progress % 50 == 0:
            avg_loss = (loss_accum / args.grad_accum).item()
            elapsed = max(1e-6, time.perf_counter() - started)
            tok_sec = tokens_processed / elapsed
            ppl = math.exp(min(avg_loss, 15))
            print(f"Step {progress:4d}/{args.max_steps} | Loss: {avg_loss:.4f} | Ppl: {ppl:6.2f} | LR: {lr:.2e} | Speed: {tok_sec:,.0f} tok/s")

        if progress % 250 == 0 or progress == args.max_steps:
            avg_loss = (loss_accum / args.grad_accum).item()
            evaluate_talking_level(model, tokenizer, device, progress, avg_loss, args.phase)
            state = {
                "model": unwrap_model(model).state_dict(),
                "step": progress,
                "phase": args.phase,
                "config": cfg.__dict__,
            }
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
