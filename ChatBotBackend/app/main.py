# app/main.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.schemas import Prompt
from app.models import generate_response

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # exact dev‑server origin
    allow_credentials=True,                  # send cookies / auth if you need
    allow_methods=["POST", "OPTIONS"],       # explicit list
    allow_headers=["content-type", "authorization"],
)

@app.post("/chat")
def chat(prompt: Prompt):
    # prompt.text is the JSON field sent from React
    reply = generate_response(prompt.text)
    print(reply,"================")
    return {"reply": reply}
