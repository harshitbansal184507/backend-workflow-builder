Readme · MD
Copy

# Workflow Builder

AI-driven conversation workflows over WebSocket.

---

## Setup

```bash
# Install dependencies
pip install fastapi uvicorn langgraph langchain-openai python-dotenv

# Add your OpenAI key
echo "OPENAI_API_KEY=sk-..." > .env
```

## Run

```bash
uvicorn backend.server:app --reload --port 8000
```

Open `frontend/index.html` in Chrome or Edge.

---
