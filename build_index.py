import os
import json
import pandas as pd
from langchain_core.documents import Document
from langchain_chroma import Chroma

PARQUET_FILENAME = "docker_docs_rag.parquet"


def _download_from_kaggle():
    """Pobiera dataset z Kaggle (kagglehub)."""
    dataset = os.environ.get("KAGGLE_DATASET", "martininf1n1ty/docker-docs-rag-dataset")
    import kagglehub
    path = kagglehub.dataset_download(dataset)
    print("Path to dataset files:", path)
    local = os.path.join(path, PARQUET_FILENAME)
    if os.path.isfile(local):
        return local
    for root, _, files in os.walk(path):
        for f in files:
            if f == PARQUET_FILENAME or f.endswith(".parquet"):
                return os.path.join(root, f)
    return None

PARQUET_PATH = os.path.join(os.path.dirname(__file__), PARQUET_FILENAME)
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")

def _load_dataframe():
    """Wczytuje DataFrame z parquet (lokalnie lub Kaggle)."""
    df = pd.DataFrame()
    if os.path.isfile(PARQUET_PATH):
        df = pd.read_parquet(PARQUET_PATH)
        print("✅ Dataset wczytany z pliku parquet")
    if df.empty:
        local_path = os.path.join(DATA_DIR, PARQUET_FILENAME)
        if os.path.isfile(local_path):
            df = pd.read_parquet(local_path)
            print("✅ Dataset wczytany z pliku parquet (data/)")
        else:
            print("📥 Próba pobrania z Kaggle...")
            downloaded = _download_from_kaggle()
            if downloaded:
                df = pd.read_parquet(downloaded)
                print("✅ Dataset pobrany z Kaggle")
            else:
                print("❌ Nie znaleziono pliku parquet.")
    return df

def _chroma_safe_metadata_value(value):
    """Convert value to a type Chroma accepts: str, int, float, bool, or None."""
    if value is None:
        return None
    if isinstance(value, (bool, int, float)):
        return value
    if isinstance(value, str):
        return value
    # list, ndarray, or other iterable -> JSON string
    try:
        if hasattr(value, "tolist"):  # numpy ndarray
            value = value.tolist()
        return json.dumps(value, ensure_ascii=False) if value else ""
    except (TypeError, ValueError):
        return str(value)


def _df_to_docs(df: pd.DataFrame):
    """Zamienia DataFrame na listę Document."""
    docs = []
    for _, row in df.iterrows():
        content = row.get("content", "")
        if not content or (isinstance(content, str) and not content.strip()):
            title = row.get("title", "")
            desc = row.get("description", "")
            content = f"{title}\n\n{desc}".strip() or "(brak treści)"
        if isinstance(content, str) and not content.strip():
            continue
        metadata = {
            "file_path": _chroma_safe_metadata_value(row.get("file_path", "")),
            "title": _chroma_safe_metadata_value(row.get("title", "")),
        }
        for key in ("tags", "keywords", "aliases"):
            if key in row and row[key] is not None:
                metadata[key] = _chroma_safe_metadata_value(row[key])
        docs.append(Document(page_content=content, metadata=metadata))
    return docs

CHROMA_DIR = os.path.join(os.path.dirname(__file__), "chroma")
COLLECTION_NAME = "docker_docs_rag"

def _build_vectorstore(doc_splits, embeddings):
    """Buduje lub ładuje Chroma; zwraca vectorstore."""
    force_rebuild = os.environ.get("REBUILD_INDEX", "").lower() in ("1", "true", "yes")
    if force_rebuild and os.path.isdir(CHROMA_DIR):
        import shutil
        shutil.rmtree(CHROMA_DIR)
        print("🔄 REBUILD_INDEX=1 — usunięto stary indeks, budowanie od zera...")
    if doc_splits and os.path.isdir(CHROMA_DIR) and os.listdir(CHROMA_DIR):
        try:
            vs = Chroma(
                collection_name=COLLECTION_NAME,
                embedding_function=embeddings,
                persist_directory=CHROMA_DIR,
            )
            print("✅ Indeks Chroma wczytany z katalogu:", CHROMA_DIR)
            return vs
        except Exception as e:
            print(f"⚠️ Nie udało się wczytać Chroma ({e}), budowanie od zera...")
    if doc_splits:
        os.makedirs(os.path.dirname(CHROMA_DIR) or ".", exist_ok=True)
        vs = Chroma.from_documents(
            doc_splits,
            embeddings,
            collection_name=COLLECTION_NAME,
            persist_directory=CHROMA_DIR,
        )
        print("✅ Indeks Chroma zbudowany i zapisany do:", CHROMA_DIR)
        return vs
    vs = Chroma.from_documents(
        [Document(page_content="(brak dokumentów)", metadata={})],
        embeddings,
        collection_name=COLLECTION_NAME,
        persist_directory=CHROMA_DIR,
    )
    return vs

from langchain_core.tools import create_retriever_tool
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
load_dotenv()

if __name__ == "__main__":
    _download_from_kaggle()
    df = _load_dataframe()
    docs = _df_to_docs(df)
    _text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=400,
        chunk_overlap=100,
    )
    doc_splits = _text_splitter.split_documents(docs) if docs else []
    if doc_splits:
        print(f"   • Chunków: {len(doc_splits):,}")

    _embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    _vectorstore = _build_vectorstore(doc_splits, _embeddings)
    retriever = _vectorstore.as_retriever()

    retriever_tool = create_retriever_tool(
        retriever,
        "search_docker_docs",
        "Wyszukuj fragmenty dokumentacji Docker (instrukcje, API, konfiguracja, opisy).",
    )
    results = retriever.invoke("Jak zainstalować Docker Desktop na Linuxie?")
    if results:
        print(results[0].page_content)
    else:
        print("(brak wyników)")