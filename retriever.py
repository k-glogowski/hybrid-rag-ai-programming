from build_index import _download_from_kaggle, _load_dataframe, _df_to_docs, _build_vectorstore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_core.tools import create_retriever_tool

from langchain_core.tools import create_retriever_tool
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
load_dotenv()

def get_retriever(k: int = 4):
    """Retriever z dokumentacji Docker. k – liczba chunków zwracanych przy wyszukiwaniu."""
    df = _load_dataframe()
    docs = _df_to_docs(df)
    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=400,
        chunk_overlap=100,
    )
    doc_splits = text_splitter.split_documents(docs) if docs else []
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vectorstore = _build_vectorstore(doc_splits, embeddings)
    return vectorstore.as_retriever(search_kwargs={"k": k})

if __name__ == "__main__":
    retriever = get_retriever()
    results = retriever.invoke("Jak zainstalować Docker Desktop na Linuxie?")
    print(results)