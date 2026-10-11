"""Settings shared by the API and the ingestion script.

Both must use the same embedding model, or retrieval silently degrades:
query vectors and document vectors would come from different models.
"""

import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
INDEX_DIR = BASE_DIR / "index"

EMBEDDING_MODEL = "text-embedding-3-small"
CHAT_MODEL = os.getenv("CHAT_MODEL", "gpt-4o-mini")

TOP_K = 5

# Chunks scoring below this cosine similarity are treated as "not relevant".
# If nothing clears it, the API declines instead of letting the model guess.
# Tune it with the evaluation set: raise it if off-topic questions get answers,
# lower it if real questions get refused.
MIN_SIMILARITY = float(os.getenv("MIN_SIMILARITY", "0.37"))

MAX_MESSAGE_CHARS = 1000

# Simple per-IP limit. On serverless this is per running instance, so treat it
# as a speed bump. The real cost protection is the spending limit on your
# OpenAI account.
RATE_LIMIT_REQUESTS = int(os.getenv("RATE_LIMIT_REQUESTS", "10"))
RATE_LIMIT_WINDOW_SECONDS = 60

# Comma-separated list, e.g. "https://medical-rag.vercel.app,http://localhost:3000"
ALLOWED_ORIGINS = [
    origin.strip()
    for origin in os.getenv(
        "ALLOWED_ORIGINS",
        "http://localhost:3000,http://127.0.0.1:3000",
    ).split(",")
    if origin.strip()
]
