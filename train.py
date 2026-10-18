from __future__ import annotations

import argparse
import math
import os
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
BOT_ID, EOS_ID, USER_ID, PAD_ID = 4, 2, 3, 0


def batch_from_memmap(data: np.memmap, batch_size: int, phase: str, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    if phase == "pretrain":
        starts = np.random.randint(0, len(data) - WINDOW - 1, size=batch_size)
        rows = np.stack([data[start:start + WINDOW + 1] for start in starts])
        x, y = rows[:, :-1], rows[:, 1:].astype(np.int64, copy=True)
    else:
        starts = np.random.randint(0, len(data) // WINDOW, size=batch_size) * WINDOW
        rows = np.stack([data[start:start + WINDOW] for start in starts])
        x, targets = rows[:, :-1], rows[:, 1:]
        y = targets.astype(np.int64, copy=True)
        for row_index, row in enumerate(rows):
            active = False
            for token_index, token in enumerate(row[1:]):
                if token == BOT_ID:
                    active = True
                    y[row_index, token_index] = -100
                elif active:
                    if token == PAD_ID:
                        active = False
                        y[row_index, token_index] = -100
                    else:
                        y[row_index, token_index] = int(token)
                        if token == EOS_ID:
                            active = False
                else:
                    y[row_index, token_index] = -100
    return torch.from_numpy(x.astype(np.int64, copy=False)).to(device), torch.from_numpy(y.astype(np.int64, copy=False)).to(device)


def evaluate(model: Llama3Mini, tokenizer: Tokenizer, device: torch.device) -> None:
    prompts = ["Hi! Who are you?", "I am exhausted from college today.", "What are you doing right now?"]
    model.eval()
    for prompt in prompts:
        encoded = tokenizer.encode(f"<|bos|><|user|>{prompt}<|bot|>", add_special_tokens=False).ids
        result = model.generate(torch.tensor([encoded], device=device))
        print("BOT:", tokenizer.decode(result[0].tolist(), skip_special_tokens=False))


def main(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = Tokenizer.from_file("tokenizer.json")
    data = np.memmap("pretrain.bin" if args.phase == "pretrain" else "sft_waifu.bin",
                     dtype=np.uint16, mode="r")
    model = Llama3Mini(ModelConfig()).to(device)
    optimizer = AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=0.1)
    scaler = GradScaler("cuda", enabled=device.type == "cuda")
    start_step = 0
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_step = int(checkpoint.get("step", 0))
    warmup = max(1, int(args.max_steps * 0.05))
    best_loss = float("inf")
    Path("checkpoints").mkdir(exist_ok=True)
    started = time.perf_counter()
    for step in trange(start_step, args.max_steps, desc=args.phase):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        loss_value = 0.0
        for _ in range(args.grad_accum):
            x, y = batch_from_memmap(data, args.batch_size, args.phase, device)
            with autocast("cuda", dtype=torch.float16, enabled=device.type == "cuda"):
                logits = model(x)
                loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1), ignore_index=-100)
            scaler.scale(loss / args.grad_accum).backward()
            loss_value += float(loss.detach())
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        progress = step + 1
        scale = min(1.0, progress / warmup) if progress <= warmup else 0.5 * (
            1 + math.cos(math.pi * (progress - warmup) / max(1, args.max_steps - warmup)))
        lr = args.lr * scale
        for group in optimizer.param_groups:
            group["lr"] = lr
        scaler.step(optimizer)
        scaler.update()
        if progress % 50 == 0:
            avg_loss = loss_value / args.grad_accum
            tok_sec = (progress * args.batch_size * args.grad_accum * (WINDOW - 1)) / max(1e-6, time.perf_counter() - started)
            print(f"{progress} | {avg_loss:.4f} | {math.exp(min(avg_loss, 20)):.3f} | {lr:.6g} | {tok_sec:.1f}")
        if progress % 500 == 0 or progress == args.max_steps:
            state = {"model": model.state_dict(), "optimizer": optimizer.state_dict(), "step": progress}
            torch.save(state, f"checkpoints/{args.phase}_latest.pt")
            if loss_value / args.grad_accum < best_loss:
                best_loss = loss_value / args.grad_accum
                torch.save(state, f"checkpoints/{args.phase}_best.pt")
            evaluate(model, tokenizer, device)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("pretrain", "sft"), required=True)
    parser.add_argument("--max_steps", type=int, required=True)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--grad_accum", type=int, default=4)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--resume", default="")
    main(parser.parse_args())
