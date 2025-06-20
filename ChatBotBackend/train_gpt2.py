#!/usr/bin/env python3
# train_gpt2.py  – with DataCollatorForLanguageModeling

from datasets import load_from_disk
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    TrainerCallback,
    DataCollatorForLanguageModeling,
)

# ───── Config ─────
MODEL_NAME  = "gpt2"
DATA_PATH   = "tokenized_data"   # Directory from prepare_dataset.py
OUTPUT_DIR  = "gpt2-chatbot"

# ── Callback to print training loss ─────────────────
class LossPrinter(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs and "loss" in logs:
            print(f"step {state.global_step:>4} | loss {logs['loss']:.4f}")

# ── 1. Load & extend tokenizer ─────────────────
print("⚙️  Loading tokenizer and adding special tokens…")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

specials = {}
if "<|startoftext|>" not in tokenizer.get_vocab():
    specials["bos_token"] = "<|startoftext|>"
if "<|endoftext|>" not in tokenizer.get_vocab():
    specials["eos_token"] = "<|endoftext|>"
if tokenizer.pad_token is None:
    specials["pad_token"] = tokenizer.eos_token

if specials:
    tokenizer.add_special_tokens(specials)
    print(f"Added specials: {specials}")
else:
    print("No new special tokens added.")

# ── 2. Load model and resize token embeddings ─────────────────
print("⚙️  Loading model and resizing token embeddings…")
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
model.resize_token_embeddings(len(tokenizer))

# ── 3. Load tokenized dataset ─────────────────
print(f"⚙️  Loading tokenized dataset from '{DATA_PATH}'…")
dataset = load_from_disk(DATA_PATH)

# Debug: check for out‑of‑range token IDs
max_id = max(sum(dataset["input_ids"], []))
print(f"🔎 Max token id: {max_id}, Vocab size: {len(tokenizer)}")
assert max_id < len(tokenizer), "Token id out of range – check tokenizer & dataset!"

# ── 4. Data collator (creates labels on the fly) ─────────────
print("⚙️  Setting up DataCollatorForLanguageModeling…")
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False,            # causal LM => no masking
)

# ── 5. Training arguments ─────────────────────
training_args = TrainingArguments(
    output_dir             = OUTPUT_DIR,
    overwrite_output_dir   = True,
    per_device_train_batch_size = 1,
    num_train_epochs       = 5,      # more epochs for small data
    learning_rate          = 5e-5,
    logging_steps          = 10,
    save_steps             = 500,
    save_total_limit       = 2,
    dataloader_pin_memory  = False,
    remove_unused_columns  = False,  # keep input_ids & attention_mask for collator
)

# ── 6. Initialize Trainer ─────────────────
trainer = Trainer(
    model           = model,
    args            = training_args,
    train_dataset   = dataset,
    data_collator   = data_collator,
    callbacks       = [LossPrinter()],
)

# ── 7. Start training ─────────────────
print("⚙️  Starting training…")
trainer.train()

# ── 8. Save the fine‑tuned model ─────────────────
print(f"⚙️  Saving fine‑tuned model to '{OUTPUT_DIR}'…")
model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print("✅ Training complete, model saved.")
