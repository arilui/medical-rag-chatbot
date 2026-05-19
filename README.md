# Medical RAG Chatbot

A full-stack Retrieval-Augmented Generation (RAG) medical education chatbot built with Next.js, FastAPI, OpenAI embeddings, and Chroma vector search.

The application retrieves trusted medical information from public health sources and generates grounded responses through a ChatGPT-style interface.

---

## Features

- Full-stack AI chatbot application
- ChatGPT/Claude-style frontend UI
- Retrieval-Augmented Generation (RAG)
- Vector similarity search using ChromaDB
- OpenAI embeddings + LLM integration
- Medical information retrieval from trusted public sources
- Source-grounded responses with citations
- FastAPI backend API
- Next.js + Tailwind frontend
- Educational safety guardrails

---

## Tech Stack

### Frontend
- Next.js
- React
- TypeScript
- Tailwind CSS

### Backend
- FastAPI
- LangChain
- OpenAI API
- ChromaDB
- BeautifulSoup

### AI / Retrieval
- OpenAI Embeddings (`text-embedding-3-small`)
- GPT-4o Mini
- Vector similarity search
- Recursive text chunking

---

## Architecture

```text
Frontend (Next.js)
        ↓
FastAPI Backend
        ↓
User Query
        ↓
Chroma Vector Search
        ↓
Relevant Medical Context
        ↓
OpenAI LLM
        ↓
Grounded Response + Sources
```

---

## Data Sources

This project currently retrieves educational medical information from:

- MedlinePlus
- CDC
- NIH

The current ingestion pipeline scrapes trusted public health web pages and converts them into vector embeddings for semantic retrieval.

---

## Project Structure

```text
medical-rag-chatbot/
│
├── backend/
│   ├── app.py
│   ├── ingest_web.py
│   ├── chroma_db/
│   └── docs/
│
├── frontend/
│   ├── app/
│   ├── public/
│   └── package.json
│
└── .gitignore
```

---

## Installation

### Clone Repository

```bash
git clone https://github.com/YOUR_USERNAME/medical-rag-chatbot.git
cd medical-rag-chatbot
```

---

## Backend Setup

```bash
cd backend
python -m venv venv
```

Activate environment:

### Windows

```bash
venv\Scripts\activate
```

### macOS/Linux

```bash
source venv/bin/activate
```

Install dependencies:

```bash
pip install fastapi uvicorn langchain langchain-openai langchain-community langchain-chroma langchain-text-splitters chromadb pypdf python-dotenv requests beautifulsoup4
```

Create `.env`:

```env
OPENAI_API_KEY=your_openai_api_key
```

Run ingestion:

```bash
python ingest_web.py
```

Start backend:

```bash
python -m uvicorn app:app --reload
```

Backend runs at:

```text
http://127.0.0.1:8000
```

---

## Frontend Setup

```bash
cd frontend
npm install
npm run dev
```

Frontend runs at:

```text
http://localhost:3000
```

---

## Example Questions

- What are common symptoms of diabetes?
- What lifestyle changes help reduce high blood pressure?
- What is heart disease?
- What foods support heart health?

---

## Safety Notes

This application is intended for educational purposes only.

It does not:
- diagnose medical conditions
- prescribe treatment
- replace professional medical advice
- provide emergency care guidance

Users should consult qualified healthcare professionals for medical decisions.
