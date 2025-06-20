# app/model_loader.py
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# Load once at startup
MODEL_DIR = "gpt2-chatbot"          # your fine‑tuned checkpoint folder
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, padding_side="left")
model     = AutoModelForCausalLM.from_pretrained(MODEL_DIR)

# Ensure GPT‑2 has a pad token
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    model.resize_token_embeddings(len(tokenizer))

def generate_response(user_prompt: str, max_new_tokens: int = 64) -> str:
    """
    Turn a user prompt into a bot reply.
    `user_prompt` is the *string* you got from FastAPI.
    """
    # ---- 1. Encode ----
    inputs = tokenizer(user_prompt, return_tensors="pt", padding=True)

    # ---- 2. Generate (pass inputs ONCE) ----
    # output_ids = model.generate(
    #     **inputs,                     # contains input_ids & attention_mask
    #     max_new_tokens=max_new_tokens,
    #     pad_token_id=tokenizer.pad_token_id,
    #     eos_token_id=tokenizer.eos_token_id,
    # )
    output_ids = model.generate(
        **inputs,
        max_new_tokens=100,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        do_sample=True,
        top_k=50,
        top_p=0.95,
        temperature=0.7,
        repetition_penalty=1.2,   # 👈 KEY FIX
    )


    # ---- 3. Decode and strip the prompt portion ----
    full_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    reply_only = full_text[len(user_prompt):].strip()

    return reply_only
