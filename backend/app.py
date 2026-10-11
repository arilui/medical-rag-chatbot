import json
import time
from collections import defaultdict, deque

import numpy as np
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from openai import OpenAI
from pydantic import BaseModel, Field

from config import (
    ALLOWED_ORIGINS,
    CHAT_MODEL,
    EMBEDDING_MODEL,
    INDEX_DIR,
    MAX_MESSAGE_CHARS,
    MIN_SIMILARITY,
    RATE_LIMIT_REQUESTS,
    RATE_LIMIT_WINDOW_SECONDS,
    TOP_K,
)

load_dotenv()

app = FastAPI(title="Medical RAG Chatbot API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)

client = OpenAI()


# --- Retrieval index -------------------------------------------------------
# The corpus is a few hundred chunks, so a brute-force dot product over an
# in-memory matrix takes well under a millisecond. A vector database would add
# a large dependency and a writable disk without making retrieval better.

def load_index():
    metadata = json.loads((INDEX_DIR / "chunks.json").read_text(encoding="utf-8"))

    if metadata["embedding_model"] != EMBEDDING_MODEL:
        raise RuntimeError(
            f"Index was built with {metadata['embedding_model']} but the API "
            f"queries with {EMBEDDING_MODEL}. Re-run ingest.py."
        )

    return metadata["chunks"], np.load(INDEX_DIR / "embeddings.npy")


CHUNKS, EMBEDDINGS = load_index()


def retrieve(question: str, k: int = TOP_K):
    response = client.embeddings.create(model=EMBEDDING_MODEL, input=question)
    query = np.asarray(response.data[0].embedding, dtype=np.float32)
    query /= np.linalg.norm(query)

    scores = EMBEDDINGS @ query
    top = np.argsort(scores)[::-1][:k]

    return [{**CHUNKS[i], "score": float(scores[i])} for i in top]


# --- Rate limiting -----------------------------------------------------------

_request_log: dict[str, deque] = defaultdict(deque)


def check_rate_limit(client_ip: str) -> None:
    now = time.monotonic()
    timestamps = _request_log[client_ip]

    while timestamps and now - timestamps[0] > RATE_LIMIT_WINDOW_SECONDS:
        timestamps.popleft()

    if len(timestamps) >= RATE_LIMIT_REQUESTS:
        raise HTTPException(
            status_code=429,
            detail="Too many requests. Please wait a minute and try again.",
        )

    timestamps.append(now)


def get_client_ip(request: Request) -> str:
    forwarded = request.headers.get("x-forwarded-for")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


# --- API -------------------------------------------------------------------------

class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=MAX_MESSAGE_CHARS)


NO_CONTEXT_ANSWER = (
    "I don't have reference material on that topic, so I can't answer it reliably. "
    "I currently cover diabetes, high blood pressure, heart disease, nutrition, "
    "and anxiety.\n\n"
    "This is educational information only. For personal medical questions, "
    "please talk to a qualified healthcare professional."
)

PROMPT_TEMPLATE = """
You are a careful medical education assistant.

Answer the user's question using only the provided medical reference context.

Rules:
- Use the context as your source of truth.
- If the context only partly answers the question, say what it supports and what it doesn't.
- If the context doesn't answer the question, say so instead of answering from memory.
- Do not diagnose.
- Do not prescribe medication.
- Do not replace professional medical advice.
- If emergency symptoms are mentioned, tell the user to seek urgent medical care.

Always include:
1. A clear answer
2. A short safety note
3. A reminder that this is educational information

Context:
{context}

User question:
{question}
"""


@app.get("/")
def root():
    return {"message": "Medical RAG backend running", "chunks_indexed": len(CHUNKS)}


@app.post("/chat")
def chat(body: ChatRequest, request: Request):
    check_rate_limit(get_client_ip(request))

    question = body.message.strip()
    results = retrieve(question)
    relevant = [r for r in results if r["score"] >= MIN_SIMILARITY]

    if not relevant:
        return {"answer": NO_CONTEXT_ANSWER, "sources": []}

    context = "\n\n".join(
        f"Source: {r['source']}\nURL: {r['url']}\n\nContent:\n{r['text']}"
        for r in relevant
    )

    response = client.chat.completions.create(
        model=CHAT_MODEL,
        temperature=0.2,
        max_tokens=600,
        messages=[
            {
                "role": "user",
                "content": PROMPT_TEMPLATE.format(context=context, question=question),
            }
        ],
    )

    sources = [
        {
            "source": r["source"],
            "url": r["url"],
            "score": round(r["score"], 3),
            "preview": r["text"][:250],
        }
        for r in relevant
    ]

    return {"answer": response.choices[0].message.content, "sources": sources}
