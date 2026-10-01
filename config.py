"""Central configuration. Override any value with an environment variable."""
import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
KB_DIR = DATA_DIR / "knowledge_base"
ORDERS_FILE = DATA_DIR / "orders.json"
PRODUCTS_FILE = DATA_DIR / "products.json"
TICKETS_FILE = DATA_DIR / "tickets.jsonl"

STORE_NAME = os.getenv("STORE_NAME", "ShopEase")
MODEL = os.getenv("CHATBOT_MODEL", "claude-opus-5-5")
# "low" keeps chat replies fast and cheap; raise to "medium"/"high" if answers need more reasoning.
EFFORT = os.getenv("CHATBOT_EFFORT", "low")
MAX_TOKENS = 16000

# Retrieval
CHUNK_WORDS = 120
CHUNK_OVERLAP = 30
TOP_K = 4

# Safety valve for the tool-use loop
MAX_TOOL_ROUNDS = 8
