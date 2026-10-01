# E-commerce Customer Service RAG Chatbot (Python)

Claude answers customer questions grounded in your help-center articles, looks up orders, checks products, and escalates to humans via tickets.

## Setup
```bash
pip install -r requirements.txt
set ANTHROPIC_API_KEY=sk-ant-...        # PowerShell: $env:ANTHROPIC_API_KEY="sk-ant-..."
```

## Run
```bash
python cli.py            # terminal chat (add --debug to see tool calls)
streamlit run app.py     # web chat UI
python -m unittest test_offline -v   # offline tests, no API key needed
```

## How it works
1. `rag/chunker.py` splits every `.md`/`.txt` in `data/knowledge_base/` into heading-aware overlapping chunks.
2. `rag/retriever.py` indexes them with BM25 (pure Python, no vector DB).
3. `chatbot.py` runs Claude in a tool-use loop. Claude calls `search_knowledge_base` (the retrieval step), `lookup_order` (needs order id + email), `search_products`, and `create_support_ticket` (appends to `data/tickets.jsonl`).

## Customize
- **Your policies:** drop `.md` files into `data/knowledge_base/`; they are indexed on startup.
- **Real data:** replace the JSON-file reads in `tools.py` with calls to your order/product database or API.
- **Settings:** env vars `STORE_NAME`, `CHATBOT_MODEL` (default `claude-opus-5-5`), `CHATBOT_EFFORT` (default `low`).

Try: *"Where is order ORD-10042? My email is john.doe@example.com"* or *"How long do refunds take for PayPal?"*
