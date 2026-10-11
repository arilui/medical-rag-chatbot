"""
Build the retrieval index for the Medical RAG Chatbot.

Run this locally whenever you change the source pages:

    pip install -r requirements-dev.txt
    python ingest.py

It writes two files into backend/index/, which are committed to the repo and
shipped with the deployed API:

    chunks.json      the text chunks and their source metadata
    embeddings.npy   one L2-normalised embedding per chunk (float32)

The deployed API never calls this script, so the scraping and chunking
libraries stay out of the production bundle.
"""

import json
from pathlib import Path

import numpy as np
import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from openai import OpenAI

from config import EMBEDDING_MODEL, INDEX_DIR

load_dotenv()

# Add pages here to grow the knowledge base. Keep `source` accurate:
# it is shown to users as the citation.
SOURCES = [
    {"source": "MedlinePlus", "url": "https://medlineplus.gov/diabetes.html"},
    {"source": "MedlinePlus", "url": "https://medlineplus.gov/highbloodpressure.html"},
    {"source": "MedlinePlus", "url": "https://medlineplus.gov/heartdiseases.html"},
    {"source": "MedlinePlus", "url": "https://medlineplus.gov/nutrition.html"},
    {"source": "MedlinePlus", "url": "https://medlineplus.gov/anxiety.html"},
]

CHUNK_SIZE = 800
CHUNK_OVERLAP = 150
EMBED_BATCH_SIZE = 100


def fetch_page_text(url: str) -> str:
    response = requests.get(url, timeout=15)
    response.raise_for_status()

    soup = BeautifulSoup(response.text, "html.parser")

    for tag in soup(["script", "style", "nav", "footer", "header"]):
        tag.decompose()

    main = soup.find("main") or soup.body
    return main.get_text(separator="\n", strip=True)


def embed(client: OpenAI, texts: list[str]) -> np.ndarray:
    vectors = []
    for start in range(0, len(texts), EMBED_BATCH_SIZE):
        batch = texts[start : start + EMBED_BATCH_SIZE]
        response = client.embeddings.create(model=EMBEDDING_MODEL, input=batch)
        vectors.extend(item.embedding for item in response.data)

    matrix = np.asarray(vectors, dtype=np.float32)
    # Normalise once here so the API can use a plain dot product for cosine similarity.
    matrix /= np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix


def main() -> None:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )

    chunks = []
    for page in SOURCES:
        print(f"Fetching {page['url']}")
        text = fetch_page_text(page["url"])

        for piece in splitter.split_text(text):
            chunks.append(
                {
                    "text": piece,
                    "source": page["source"],
                    "url": page["url"],
                }
            )

    print(f"Embedding {len(chunks)} chunks with {EMBEDDING_MODEL}...")
    matrix = embed(OpenAI(), [chunk["text"] for chunk in chunks])

    INDEX_DIR.mkdir(exist_ok=True)
    np.save(INDEX_DIR / "embeddings.npy", matrix)
    (INDEX_DIR / "chunks.json").write_text(
        json.dumps(
            {"embedding_model": EMBEDDING_MODEL, "chunks": chunks},
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"Indexed {len(chunks)} chunks from {len(SOURCES)} pages into {INDEX_DIR}/")


if __name__ == "__main__":
    main()
