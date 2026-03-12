# Workflow Builder

AI-driven conversation workflows over WebSocket.

---

## Setup

```bash
# Install dependencies
pip install -r requirements.txt

# Add your OpenAI key
echo "OPENAI_API_KEY=sk-..." > .env
```

## Run

```bash
uvicorn backend.server:app --reload --port 8000
```

Open `client.html` in Chrome or Edge.

---
