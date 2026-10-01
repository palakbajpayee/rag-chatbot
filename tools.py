"""Tools the assistant can call: knowledge-base search (the RAG step), order lookup,
product lookup and support-ticket creation. Swap the JSON files for real APIs/DBs in production."""
import json
import re
import uuid
from datetime import datetime, timezone

import config
from rag import BM25Retriever, load_chunks

TOOLS = [
    {
        "name": "search_knowledge_base",
        "description": (
            "Search the store's help-center articles (shipping, returns, refunds, payments, "
            "accounts, warranty, FAQs). Call this before answering ANY question about store "
            "policy or procedures. Returns the most relevant passages with their source ids."
        ),
        "strict": True,
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "A focused search query, e.g. 'refund timeline for card payments'."},
            },
            "required": ["query"],
            "additionalProperties": False,
        },
    },
    {
        "name": "lookup_order",
        "description": (
            "Look up an order's status, items, shipping and tracking details. Requires both the "
            "order id and the email address used for the order; ask the customer for any that are missing."
        ),
        "strict": True,
        "input_schema": {
            "type": "object",
            "properties": {
                "order_id": {"type": "string", "description": "Order id, e.g. 'ORD-10042'."},
                "email": {"type": "string", "description": "Email address on the order."},
            },
            "required": ["order_id", "email"],
            "additionalProperties": False,
        },
    },
    {
        "name": "search_products",
        "description": "Search the product catalog by name or keyword to get price, stock and warranty info.",
        "strict": True,
        "input_schema": {
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
            "additionalProperties": False,
        },
    },
    {
        "name": "create_support_ticket",
        "description": (
            "Escalate to a human agent. Use when the customer asks for a human, when the issue "
            "needs an exception to policy, involves damaged/missing items, payment disputes, or "
            "anything you cannot resolve with the other tools. Confirm the customer's email first."
        ),
        "strict": True,
        "input_schema": {
            "type": "object",
            "properties": {
                "email": {"type": "string"},
                "category": {
                    "type": "string",
                    "enum": ["order_issue", "return_refund", "payment", "account", "product", "other"],
                },
                "summary": {"type": "string", "description": "Concise description of the issue and what was already tried."},
                "order_id": {"type": "string", "description": "Related order id, or an empty string if none."},
            },
            "required": ["email", "category", "summary", "order_id"],
            "additionalProperties": False,
        },
    },
]

_EMAIL = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")


class ToolBox:
    def __init__(self):
        self.retriever = BM25Retriever(load_chunks(config.KB_DIR, config.CHUNK_WORDS, config.CHUNK_OVERLAP))
        self.orders = {o["order_id"].upper(): o for o in json.loads(config.ORDERS_FILE.read_text(encoding="utf-8"))}
        self.products = json.loads(config.PRODUCTS_FILE.read_text(encoding="utf-8"))

    def run(self, name: str, args: dict) -> tuple[str, bool]:
        """Execute a tool. Returns (result_json, is_error)."""
        handler = getattr(self, f"_tool_{name}", None)
        if handler is None:
            return json.dumps({"error": f"Unknown tool '{name}'"}), True
        try:
            return json.dumps(handler(**args), ensure_ascii=False), False
        except (TypeError, ValueError) as exc:
            return json.dumps({"error": str(exc)}), True

    # --- tool implementations -------------------------------------------------

    def _tool_search_knowledge_base(self, query: str) -> dict:
        hits = self.retriever.search(query, top_k=config.TOP_K)
        if not hits:
            return {"results": [], "note": "No matching help-center articles found."}
        return {"results": [
            {"source_id": c.id, "article": c.source, "section": c.section, "score": round(s, 2), "text": c.text}
            for c, s in hits
        ]}

    def _tool_lookup_order(self, order_id: str, email: str) -> dict:
        order = self.orders.get(order_id.strip().upper())
        # Same message for "not found" and "email mismatch" so order ids can't be probed.
        if order is None or order["email"].lower() != email.strip().lower():
            return {"found": False, "message": "No order matches that order id and email combination."}
        return {"found": True, "order": {k: v for k, v in order.items() if k != "email"}}

    def _tool_search_products(self, query: str) -> dict:
        terms = query.lower().split()
        matches = [
            p for p in self.products
            if any(t in f"{p['name']} {p['category']} {p['sku']}".lower() for t in terms)
        ]
        return {"products": matches[:5]} if matches else {"products": [], "message": "No products matched."}

    def _tool_create_support_ticket(self, email: str, category: str, summary: str, order_id: str) -> dict:
        if not _EMAIL.match(email.strip()):
            raise ValueError("A valid customer email is required to open a ticket.")
        ticket = {
            "ticket_id": f"TCK-{uuid.uuid4().hex[:8].upper()}",
            "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "email": email.strip(),
            "category": category,
            "order_id": order_id.strip() or None,
            "summary": summary.strip(),
            "status": "open",
        }
        with config.TICKETS_FILE.open("a", encoding="utf-8") as f:
            f.write(json.dumps(ticket) + "\n")
        return {"ticket_id": ticket["ticket_id"], "status": "open",
                "message": "A human agent will reply by email within 24 hours."}
