# Medical RAG Chatbot

A full-stack Retrieval-Augmented Generation (RAG) chatbot for medical education, built with Next.js, FastAPI, and OpenAI.

It answers health questions using retrieved passages from MedlinePlus, cites the pages it used, and declines questions its reference material doesn't cover instead of answering from the model's memory.

**Live demo:** _add your Vercel URL here_

---

## Features

- Retrieval-Augmented Generation over trusted public health pages
- Source citations with links back to the original pages
- Relevance threshold: off-topic questions are declined without calling the LLM
- Input length limit and per-IP rate limiting
- Educational safety guardrails in the prompt
- Next.js + Tailwind chat interface
- Serverless deployment on Vercel

---

## Tech Stack

**Frontend:** Next.js, React, TypeScript, Tailwind CSS

**Backend:** FastAPI, NumPy, OpenAI API (`text-embedding-3-small`, `gpt-4o-mini`)

**Ingestion:** LangChain text splitters, BeautifulSoup

---

## Architecture

```text
            Offline (ingest.py)                          Online (app.py)

MedlinePlus pages                               User question
      ↓                                               ↓
Scrape + clean (BeautifulSoup)                  Embed question
      ↓                                               ↓
Chunk (800 chars, 150 overlap)                  Cosine similarity vs. index
      ↓                                               ↓
Embed (text-embedding-3-small)                  Keep chunks above threshold
      ↓                                               ↓
index/embeddings.npy + chunks.json  ──────────▶ None? → decline
                                                Some? → gpt-4o-mini with context
                                                      ↓
                                                Answer + cited sources
```

### Why no vector database?

The knowledge base is a few hundred chunks. A brute-force dot product over an in-memory NumPy matrix searches it in well under a millisecond. A vector database such as Chroma would add a large dependency and need a writable disk, neither of which suits a serverless deployment, without improving retrieval at this scale. If the corpus grew to hundreds of thousands of chunks, an approximate-nearest-neighbor index would become worthwhile.

---

## Data Sources

The index is built from these MedlinePlus topic pages:

- Diabetes
- High Blood Pressure
- Heart Diseases
- Nutrition
- Anxiety

To add more, append entries to `SOURCES` in `backend/ingest.py` and re-run ingestion.

---

## Project Structure

```text
medical-rag-chatbot/
├── backend/
│   ├── app.py               # FastAPI app: retrieval + generation
│   ├── config.py            # Shared settings (models, thresholds, CORS)
│   ├── ingest.py            # Builds the retrieval index
│   ├── index/               # Generated index, committed and deployed
│   ├── requirements.txt     # Runtime dependencies (what Vercel installs)
│   └── requirements-dev.txt # Adds ingestion + local server dependencies
└── frontend/
    └── app/page.tsx         # Chat UI
```

---

## Running Locally

### Backend

```bash
cd backend
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements-dev.txt
```

Create `backend/.env`:

```env
OPENAI_API_KEY=your_openai_api_key
```

Build the index (only needed when sources change), then start the API:

```bash
python ingest.py
python -m uvicorn app:app --reload
```

The API runs at `http://127.0.0.1:8000`.

### Frontend

```bash
cd frontend
npm install
npm run dev
```

The app runs at `http://localhost:3000`. It calls the API at `NEXT_PUBLIC_API_URL`, which defaults to `http://127.0.0.1:8000`.

---

## Deploying to Vercel

The backend and frontend deploy as two Vercel projects from this one repository.

1. **Commit the index.** Run `python ingest.py` locally and commit `backend/index/`. The deployed API reads it at startup.
2. **Backend project.** In Vercel, import this repo, set **Root Directory** to `backend`, and add these environment variables:
   - `OPENAI_API_KEY`
   - `ALLOWED_ORIGINS`: the frontend URL, for example `https://medical-rag-chatbot.vercel.app`
3. **Frontend project.** Import the repo again, set **Root Directory** to `frontend`, and add:
   - `NEXT_PUBLIC_API_URL`: the backend URL, for example `https://medical-rag-api.vercel.app`
4. **Cap spending.** Set a monthly budget limit in your OpenAI account settings. The rate limiter runs per server instance, so it slows abuse down but the budget is what actually caps cost.

Optional settings: `MIN_SIMILARITY` (default `0.30`), `RATE_LIMIT_REQUESTS` per minute (default `10`), `CHAT_MODEL` (default `gpt-4o-mini`).

---

## Safety Notes

This application is for educational purposes only. It does not diagnose conditions, prescribe treatment, replace professional medical advice, or provide emergency guidance. Consult a qualified healthcare professional for medical decisions.
