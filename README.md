

# 🤖 GPT-2 Based Chatbot (FastAPI + Custom UI)

This is a simple conversational chatbot built using:

- 🧠 **GPT-2** model from Hugging Face Transformers
- 🚀 **FastAPI** backend for serving responses
- 💬 **Frontend UI** to chat with the bot (custom built)

---

## 📁 Project Structure

ChatBotBackend/
├── prepare_dataset.py # Prepares dataset and tokenizes
├── train_gpt2.py # Trains GPT-2 model
├── app.py # FastAPI backend
├── tokenized_data/ # Tokenized dataset (generated)
├── gpt2-chatbot/ # Trained model (output folder)
├── ui/ # Frontend folder (HTML/JS)
├── requirements.txt # Python dependencies
└── README.md # This file


---

## 🛠️ Requirements

- Python 3.10 or higher
- pip
- virtualenv (optional, recommended)

---

## ⚙️ Setup Instructions

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

🧩 Step 1: Prepare the Dataset

Ensure your dataset is in a JSONL format:

{"prompt": "What is FastAPI?", "completion": "FastAPI is a modern web framework..."}
{"prompt": "How do I install FastAPI?", "completion": "You can use pip to install FastAPI..."}

Run:

python3 prepare_dataset.py

This will tokenize and save the dataset to tokenized_data/.
🧠 Step 2: Train the Model

This uses Hugging Face Trainer:

python3 train_gpt2.py

The trained model will be saved under gpt2-chatbot/.
🚀 Step 3: Run the FastAPI Backend

uvicorn app:app --reload

    Server runs on http://127.0.0.1:8000

    Chat endpoint: POST /chat with JSON:

    {
      "text": "Hello"
    }

💻 Step 4: Run the UI

If you have a simple UI in ui/index.html, open it in the browser:

xdg-open ui/index.html  # For Linux
# OR
open ui/index.html       # For macOS

Ensure CORS is enabled in FastAPI to allow frontend access.
🧪 Sample CURL Test

curl -X POST http://127.0.0.1:8000/chat \
-H "Content-Type: application/json" \
-d '{"text": "What is FastAPI?"}'

📝 To-Do

Add more sample Q&A pairs

Improve response diversity

    Integrate persistent database for storing chat history

📦 Export Dependencies

If you want to update requirements.txt:

pip freeze > requirements.txt

👤 Author

Samrudha VC
ChatGPT-assisted Open Source GPT2 Bot



🧠 References

    HuggingFace Transformers

    FastAPI Docs

    Stanford NLP Vocab Expansion


---

Let me know if you want a `requirements.txt` file generated or want to split `ui/` into React o