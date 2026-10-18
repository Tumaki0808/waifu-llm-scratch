from __future__ import annotations

import argparse
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from tokenizers import Tokenizer
from torch.utils.data import DataLoader, Dataset
from torch.amp import GradScaler, autocast
from torch.optim import AdamW
from tqdm import trange

from model import Llama3Mini, ModelConfig

WINDOW = 512
BOT_ID, EOS_ID, USER_ID, PAD_ID = 4, 2, 3, 0


class TokenDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    """Random-access token windows backed by one RAM-resident uint16 array."""

    def __init__(self, path: str, phase: str) -> None:
        self.phase = phase
        print(f"-> Loading {path} into RAM...")
        self.data = np.fromfile(path, dtype=np.uint16)
        if phase == "pretrain":
            self.length = len(self.data) - WINDOW
        else:
            self.length = len(self.data) // WINDOW
        if self.length <= 0:
            raise ValueError(f"{path} does not contain enough tokens for {phase} training.")
        print(f"-> Dataset loaded successfully! Total windows: {self.length:,}")

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        if self.phase == "pretrain":
            row = self.data[index:index + WINDOW + 1]
        else:
            start = index * WINDOW
            row = self.data[start:start + WINDOW]

        x = torch.from_numpy(row[:-1].astype(np.int64, copy=True))
        y = row[1:].astype(np.int64, copy=True)
        if self.phase == "sft":
            active = False
            for token_index, token in enumerate(row[1:]):
                if token == BOT_ID:
                    active = True
                    y[token_index] = -100
                elif active:
                    if token == PAD_ID:
                        active = False
                        y[token_index] = -100
                    else:
                        if token == EOS_ID:
                            active = False
                else:
                    y[token_index] = -100
        return x, torch.from_numpy(y)


def evaluate(model: torch.nn.Module, tokenizer: Tokenizer, device: torch.device) -> None:
    prompts = ["Hi! Who are you?", "I am exhausted from college today.", "What are you doing right now?"]
    model.eval()
    for prompt in prompts:
        encoded = tokenizer.encode(f"<|bos|><|user|>{prompt}<|bot|>", add_special_tokens=False).ids
        result = model.generate(torch.tensor([encoded], device=device))
        print("BOT:", tokenizer.decode(result[0].tolist(), skip_special_tokens=False))


def main(args: argparse.Namespace) -> None:
    print("-> Initializing device...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"-> Using device: {device}")

    print("-> Loading tokenizer.json...")
    tokenizer = Tokenizer.from_file("tokenizer.json")

    print(f"-> Setting up dataset for phase: {args.phase}")
    dataset = TokenDataset("pretrain.bin" if args.phase == "pretrain" else "sft_waifu.bin", args.phase)
    
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=0,
                        pin_memory=True, drop_last=True)
    batches = iter(loader)

    print("-> Initializing Llama3Mini model structure...")
    model = Llama3Mini(ModelConfig()).to(device)

    print("-> Setting up optimizer and scaler...")
    optimizer = AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=0.1)
    scaler = GradScaler("cuda", enabled=device.type == "cuda")

    start_step = 0
    if args.resume:
        print(f"-> Resuming checkpoint from {args.resume}...")
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_step = int(checkpoint.get("step", 0))
        print(f"-> Resumed successfully at step {start_step}!")

    warmup = max(1, int(args.max_steps * 0.05))
    best_loss = float("inf")
    last_logged_loss = float("inf")
    Path("checkpoints").mkdir(exist_ok=True)

    print("-> Starting training loop now...")
    started = time.perf_counter()
    
    for step in trange(start_step, args.max_steps, desc=args.phase):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        loss_sum = torch.zeros((), device=device)

        for _ in range(args.grad_accum):
            try:
                x, y = next(batches)
            except StopIteration:
                batches = iter(loader)
                x, y = next(batches)

            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            with autocast("cuda", dtype=torch.float16, enabled=device.type == "cuda"):
                logits = model(x)
                loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1), ignore_index=-100)

            scaler.scale(loss / args.grad_accum).backward()
            loss_sum += loss.detach()

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
            avg_loss = (loss_sum / args.grad_accum).item()
            last_logged_loss = avg_loss
            tok_sec = (progress * args.batch_size * args.grad_accum * (WINDOW - 1)) / max(1e-6, time.perf_counter() - started)
            print(f"{progress} | {avg_loss:.4f} | {math.exp(min(avg_loss, 20)):.3f} | {lr:.6g} | {tok_sec:.1f}")

        if progress % 500 == 0 or progress == args.max_steps:
            state = {"model": model.state_dict(), "optimizer": optimizer.state_dict(), "step": progress}
            torch.save(state, f"checkpoints/{args.phase}_latest.pt")
            if last_logged_loss < best_loss:
                best_loss = last_logged_loss
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
