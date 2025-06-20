# Chatbot Fine‑Tuning & API

```bash
# 1. install deps
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 2. prepare dataset
python prepare_dataset.py

# 3. fine‑tune GPT‑2 (≈10‑30 min on GPU)
python train_gpt2.py

# 4. serve API
uvicorn app.main:app --reload --port 8000

# 5. frontend (in separate terminal)
# npm install && npm start  (React app on :3001)

# 6. test
curl -X POST http://localhost:8000/chat -H "Content-Type: application/json" \
     -d '{"text":"Hello"}'
