#!/usr/bin/env python3
# prepare_dataset.py

import json
from pathlib import Path

from datasets import Dataset
from transformers import AutoTokenizer

# ─────── Config ───────
DATA_FILE   = Path("data/chatbot-dataset.json")  # one JSON per line
MODEL_NAME  = "gpt2"
OUT_DIR     = "tokenized_data"
MAX_LENGTH  = 128
START_TOKEN = "<|startoftext|>"
END_TOKEN   = "<|endoftext|>"

# ─────── Load raw data ───────
lines = DATA_FILE.read_text(encoding="utf-8").splitlines()
records = []
for idx, line in enumerate(lines, 1):
    try:
        obj = json.loads(line)
        prompt     = obj["prompt"].strip()
        completion = obj["completion"].strip()
        text = f"{START_TOKEN}User: {prompt}\nBot: {completion}{END_TOKEN}"
        records.append({"text": text})
    except json.JSONDecodeError as e:
        print(f"⚠️ Skipping invalid JSON on line {idx}: {e}")

# ─────── Build HuggingFace Dataset ───────
dataset = Dataset.from_list(records)
print(f"✅ Loaded {len(dataset)} examples")

# ─────── Tokenizer setup ───────
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
# ensure special tokens exist
specials = {}
if START_TOKEN not in tokenizer.get_vocab():
    specials["bos_token"] = START_TOKEN
if END_TOKEN not in tokenizer.get_vocab():
    specials["eos_token"] = END_TOKEN
if specials:
    tokenizer.add_special_tokens(specials)

# GPT-2 needs a pad token
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({"pad_token": tokenizer.eos_token})

# ─────── Tokenization fn ───────
def tokenize(batch):
    enc = tokenizer(
        batch["text"],
        truncation=True,
        padding="max_length",
        max_length=MAX_LENGTH,
    )
    # use input_ids as labels for causal LM
    enc["labels"] = enc["input_ids"].copy()
    return enc

# ─────── Apply tokenization ───────
tokenized = dataset.map(
    tokenize,
    batched=True,
    remove_columns=["text"],
)

# ─────── Save to disk ───────
tokenized.save_to_disk(OUT_DIR)
print(f"✅ Tokenized data saved to: {OUT_DIR}")
