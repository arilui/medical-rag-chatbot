from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI

load_dotenv()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

DB_PATH = "chroma_db"

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

db = Chroma(
    persist_directory=DB_PATH,
    embedding_function=embeddings,
)

llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0.2,
)


class ChatRequest(BaseModel):
    message: str


@app.get("/")
def root():
    return {"message": "Medical RAG backend running"}


@app.post("/chat")
def chat(request: ChatRequest):
    question = request.message

    results = db.similarity_search(question, k=5)

    context = "\n\n".join(
        [
            f"""
Source: {doc.metadata.get("source", "Unknown")}
URL: {doc.metadata.get("url", "Unknown")}

Content:
{doc.page_content}
"""
            for doc in results
        ]
    )

    prompt = f"""
You are a careful medical education assistant.

Answer the user's question using the provided medical reference context.

Rules:
- Use the context as your source of truth.
- If the context contains relevant information, summarize it clearly.
- If the context is only partially relevant, say what the context supports.
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

    response = llm.invoke(prompt)

    sources = [
        {
            "source": doc.metadata.get("source", "Unknown"),
            "url": doc.metadata.get("url", None),
            "page": doc.metadata.get("page", None),
            "preview": doc.page_content[:250],
        }
        for doc in results
    ]

    return {
        "answer": response.content,
        "sources": sources,
    }