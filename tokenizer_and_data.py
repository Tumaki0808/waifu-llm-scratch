"""Download the training corpora, train a small BPE tokenizer, and write memmaps."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any, Iterable, Iterator

import numpy as np
from datasets import load_dataset
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

SPECIAL_TOKENS = ["<|pad|>", "<|bos|>", "<|eos|>", "<|user|>", "<|bot|>"]
PAD_ID, BOS_ID, EOS_ID, USER_ID, BOT_ID = range(5)
WINDOW_SIZE = 512


def _text(value: Any) -> str:
    if isinstance(value, str):
        return value.strip()
    return str(value).strip() if value is not None else ""


def _role(value: Any) -> str:
    role = _text(value).lower()
    if role in {"human", "user", "prompt", "customer"}:
        return "user"
    if role in {"assistant", "gpt", "char", "character", "bot", "model"}:
        return "assistant"
    if role == "system":
        return "system"
    return ""


def _messages_from_row(row: dict[str, Any]) -> list[tuple[str, str]]:
    """Normalize common chat dataset schemas into adjacent user/assistant pairs."""
    raw = row.get("messages") or row.get("conversations")
    if not isinstance(raw, list):
        candidates = [row.get("dialogue"), row.get("chat"), row.get("turns")]
        raw = next((candidate for candidate in candidates if isinstance(candidate, list)), [])

    messages: list[tuple[str, str]] = []
    pending_user = ""
    if not isinstance(raw, list):
        return messages
    for item in raw:
        if not isinstance(item, dict):
            continue
        role = _role(item.get("role") or item.get("from") or item.get("speaker"))
        content = _text(item.get("content") or item.get("value") or item.get("text") or item.get("message"))
        if not content or role in {"system", ""}:
            continue
        if role == "user":
            pending_user = content
        elif pending_user:
            messages.append((pending_user, content))
            pending_user = ""
    return messages


def _row_text(row: dict[str, Any]) -> str:
    pairs = _messages_from_row(row)
    if pairs:
        return "\n".join(f"{user}\n{bot}" for user, bot in pairs)
    for key in ("text", "story", "content"):
        if key in row and _text(row[key]):
            return _text(row[key])
    return ""


def _iter_tokenizer_text(stories: Iterable[dict[str, Any]],
                         anime: Iterable[dict[str, Any]],
                         casual: Iterable[dict[str, Any]]) -> Iterator[str]:
    for row in stories:
        text = _row_text(row)
        if text:
            yield text
    for row in anime:
        for user, bot in _messages_from_row(row):
            yield f"<|user|>{user}<|bot|>{bot}"
    for row in casual:
        text = _row_text(row)
        if text:
            yield f"*smiles warmly* {text}"


def _encode(tokenizer: Tokenizer, text: str) -> list[int]:
    return tokenizer.encode(text, add_special_tokens=False).ids


def _write_pretrain(tokenizer: Tokenizer, stories: Iterable[dict[str, Any]], path: Path) -> None:
    tokens: list[int] = []
    for row in stories:
        text = _row_text(row)
        if text:
            tokens.extend([BOS_ID, *_encode(tokenizer, text), EOS_ID])
    array = np.asarray(tokens, dtype=np.uint16)
    memmap = np.memmap(path, dtype=np.uint16, mode="w+", shape=(len(array),))
    memmap[:] = array
    memmap.flush()


def _dialogue_windows(tokenizer: Tokenizer, anime: Iterable[dict[str, Any]]) -> list[np.ndarray]:
    windows: list[np.ndarray] = []
    for row in anime:
        pairs = _messages_from_row(row)
        if not pairs:
            continue
        sequence: list[int] = []
        for user, bot in pairs:
            sequence.extend([BOS_ID, USER_ID, *_encode(tokenizer, user), BOT_ID,
                             *_encode(tokenizer, bot), EOS_ID])
        for start in range(0, len(sequence), WINDOW_SIZE):
            chunk = sequence[start:start + WINDOW_SIZE]
            if chunk:
                padded = np.full(WINDOW_SIZE, PAD_ID, dtype=np.uint16)
                padded[:len(chunk)] = chunk
                windows.append(padded)
    return windows


def build(output_dir: str = ".") -> None:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    stories = load_dataset("roneneldan/TinyStories", split="train[:200000]")
    anime = load_dataset("zerofata/Roleplay-Anime-Characters", split="train")
    casual = load_dataset("HuggingFaceTB/smoltalk", "everyday-conversations", split="train[:5000]")

    tokenizer = Tokenizer(models.BPE(unk_token="<|unk|>"))
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(vocab_size=4096, special_tokens=SPECIAL_TOKENS,
                                  show_progress=True, min_frequency=2)
    tokenizer.train_from_iterator(_iter_tokenizer_text(stories, anime, casual), trainer=trainer)
    tokenizer.save(str(output / "tokenizer.json"))

    _write_pretrain(tokenizer, stories, output / "pretrain.bin")
    windows = _dialogue_windows(tokenizer, anime)
    if not windows:
        raise RuntimeError("No valid user/assistant pairs found in the anime dataset.")
    repeated = np.concatenate(windows * 8, axis=0)
    memmap = np.memmap(output / "sft_waifu.bin", dtype=np.uint16, mode="w+", shape=repeated.shape)
    memmap[:] = repeated
    memmap.flush()
    print(f"Wrote tokenizer.json, {len(repeated)} SFT tokens, and {len(repeated) // WINDOW_SIZE} windows.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", default=".")
    build(parser.parse_args().output_dir)
