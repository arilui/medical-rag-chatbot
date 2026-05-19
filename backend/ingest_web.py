import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma

load_dotenv()

DB_PATH = "chroma_db"

TOPIC_URLS = [
    "https://medlineplus.gov/diabetes.html",
    "https://medlineplus.gov/highbloodpressure.html",
    "https://medlineplus.gov/heartdiseases.html",
    "https://medlineplus.gov/nutrition.html",
    "https://medlineplus.gov/anxiety.html",
]


def fetch_page_text(url: str) -> str:
    response = requests.get(url, timeout=15)
    response.raise_for_status()

    soup = BeautifulSoup(response.text, "html.parser")

    for tag in soup(["script", "style", "nav", "footer", "header"]):
        tag.decompose()

    main = soup.find("main") or soup.body
    text = main.get_text(separator="\n", strip=True)

    return text


documents = []

for url in TOPIC_URLS:
    print(f"Fetching {url}")

    text = fetch_page_text(url)

    documents.append(
        Document(
            page_content=text,
            metadata={
                "source": "MedlinePlus",
                "url": url,
            },
        )
    )

splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=150,
)

chunks = splitter.split_documents(documents)

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory=DB_PATH,
)

print(f"Indexed {len(chunks)} chunks from {len(TOPIC_URLS)} web pages.")