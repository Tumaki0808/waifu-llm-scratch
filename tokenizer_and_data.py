from __future__ import annotations

import os
from pathlib import Path
import numpy as np
from datasets import load_dataset
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
from tqdm import tqdm

VOCAB_SIZE = 8192
WINDOW = 512
PAD_ID, BOS_ID, EOS_ID, USER_ID, BOT_ID = 0, 1, 2, 3, 4
SPECIAL_TOKENS = ["<|pad|>", "<|bos|>", "<|eos|>", "<|user|>", "<|bot|>"]


def extract_pairs_from_row(row: dict) -> list[tuple[str, str]]:
    """Defensively extracts (user, bot) turn pairs across different HF dataset schemas."""
    msgs = row.get("messages") or row.get("conversations") or []
    pairs = []
    user_buf = None
    for m in msgs:
        if not isinstance(m, dict):
            continue
        role = str(m.get("role") or m.get("from") or "").lower()
        content = str(m.get("content") or m.get("value") or "").strip()
        if not content:
            continue
        if role in ("user", "human"):
            user_buf = content
        elif role in ("assistant", "gpt", "bot", "char", "model") and user_buf is not None:
            pairs.append((user_buf, content))
            user_buf = None
    return pairs


def main() -> None:
    print("1/4 -> Downloading English & Roleplay datasets from Hugging Face...")
    stories = load_dataset("roneneldan/TinyStories", split="train[:220000]")
    smoltalk = load_dataset("HuggingFaceTB/smoltalk", "everyday-conversations", split="train")
    try:
        anime_rp = load_dataset("zerofata/Roleplay-Anime-Characters", split="train")
    except Exception as e:
        print(f"Warning: Could not load zerofata dataset ({e}), using synthetic + smoltalk fallback.")
        anime_rp = []

    # High-quality anchor dialogues so the bot always knows who it is and speaks great English
    seed_dialogues = [
        [
            ("Hi! Who are you?", "*smiles warmly and waves* Hey there! I'm Aoi, your personal AI companion that you trained from scratch! How is your day going?"),
            ("I am exhausted from college today.", "*sighs softly and pulls up a chair for you* You've been working so hard lately! Sit down, take a deep breath, and tell me all about what happened today."),
            ("What are you doing right now?", "*giggles and leans closer to the screen* Just hanging out inside your GPU memory waiting for you to talk to me! What should we chat about?"),
        ],
        [
            ("Can you speak good English?", "*nods proudly with a bright grin* Of course I can! You trained me on hundreds of thousands of stories and conversations so we can talk about anything you like!"),
            ("I feel a bit stressed about my exams.", "*places a gentle hand on your shoulder* Hey, don't panic, okay? You're smart and capable. Let's break your work down one step at a time together!"),
            ("Tell me something fun.", "*eyes sparkle with excitement* Did you know that every single word I'm saying right now is being calculated by forty-two million neural weights that you built yourself? Pretty cool, right?~"),
        ],
    ]

    sft_conversations: list[list[tuple[str, str]]] = []
    # Oversample anchor conversations so core personality is rock-solid
    for _ in range(120):
        sft_conversations.extend(seed_dialogues)

    # Extract Anime Roleplay conversations (oversampled 4x for expressive style)
    anime_count = 0
    for row in anime_rp:
        pairs = extract_pairs_from_row(row)
        if pairs:
            for _ in range(4):
                sft_conversations.append(pairs)
            anime_count += 1

    # Extract SmolTalk everyday English conversations for smart, normal chatting
    smol_count = 0
    for row in smoltalk:
        pairs = extract_pairs_from_row(row)
        if pairs:
            # Add subtle expressive actions to general conversations so tone stays unified
            styled_pairs = []
            for idx, (u, b) in enumerate(pairs):
                prefix = "*smiles* " if idx == 0 and not b.startswith("*") else ""
                styled_pairs.append((u, prefix + b))
            sft_conversations.append(styled_pairs)
            smol_count += 1

    print(f"   Loaded {len(stories):,} stories, {anime_count:,} anime RP chats, {smol_count:,} everyday English chats.")

    print("2/4 -> Training 8,192-vocab BPE Tokenizer...")
    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=VOCAB_SIZE,
        special_tokens=SPECIAL_TOKENS,
        min_frequency=2,
        show_progress=True,
    )

    def text_iterator():
        for i in range(0, min(100000, len(stories)), 2):
            yield stories[i]["text"]
        for conv in sft_conversations[:8000]:
            for u, b in conv:
                yield f"<|user|>{u}<|bot|>{b}"

    tokenizer.train_from_iterator(text_iterator(), trainer=trainer)
    tokenizer.save("tokenizer.json")
    print("   Saved tokenizer.json!")

    print("3/4 -> Building pretrain.bin (Phase 1 English & Common Sense Corpus)...")
    pretrain_chunks = []
    # Mix stories + conversational text in pretraining so it learns dialogue structure early
    for row in tqdm(stories, desc="Tokenizing Stories"):
        txt = row["text"].strip()
        if txt:
            ids = [BOS_ID] + tokenizer.encode(txt).ids + [EOS_ID]
            pretrain_chunks.append(np.array(ids, dtype=np.uint16))

    for conv in tqdm(sft_conversations[:6000], desc="Tokenizing Dialogues for Pretrain"):
        seq = [BOS_ID]
        for u, b in conv:
            seq.extend([USER_ID] + tokenizer.encode(u).ids + [BOT_ID] + tokenizer.encode(b).ids + [EOS_ID])
        pretrain_chunks.append(np.array(seq, dtype=np.uint16))

    pretrain_arr = np.concatenate(pretrain_chunks)
    pretrain_arr.tofile("pretrain.bin")
    print(f"   Saved pretrain.bin ({len(pretrain_arr):,} tokens, {pretrain_arr.nbytes / 1e6:.1f} MB)")

    print("4/4 -> Building sft_x.bin & sft_y.bin (Pre-masked Phase 2 SFT Tensors)...")
    np.random.shuffle(sft_conversations)
    x_rows = []
    y_rows = []

    for conv in tqdm(sft_conversations, desc="Building Masked SFT Windows"):
        seq = [BOS_ID]
        mask = [0]  # 0 = ignore (-100), 1 = train on token
        for u, b in conv:
            u_ids = [USER_ID] + tokenizer.encode(u).ids + [BOT_ID]
            b_ids = tokenizer.encode(b).ids + [EOS_ID]
            seq.extend(u_ids + b_ids)
            mask.extend([0] * len(u_ids) + [1] * len(b_ids))

        if len(seq) < 8:
            continue
        # Truncate or pad to WINDOW + 1 (513 tokens -> 512 x and 512 y)
        seq = seq[: WINDOW + 1]
        mask = mask[: WINDOW + 1]
        if len(seq) < WINDOW + 1:
            pad_len = (WINDOW + 1) - len(seq)
            seq.extend([PAD_ID] * pad_len)
            mask.extend([0] * pad_len)

        x_arr = np.array(seq[:-1], dtype=np.uint16)
        y_raw = np.array(seq[1:], dtype=np.int64)
        y_mask = np.array(mask[1:], dtype=np.int64)
        # Apply -100 mask directly where y_mask == 0
        y_arr = np.where(y_mask == 1, y_raw, -100).astype(np.int16)

        if np.any(y_arr != -100):
            x_rows.append(x_arr)
            y_rows.append(y_arr)

    sft_x = np.stack(x_rows)
    sft_y = np.stack(y_rows)
    sft_x.tofile("sft_x.bin")
    sft_y.tofile("sft_y.bin")
    print(f"   Saved sft_x.bin & sft_y.bin ({len(sft_x):,} conversation windows)!")
    print("ALL DATA READY FOR ULTRA-FAST TRAINING!")


if __name__ == "__main__":
    main()
