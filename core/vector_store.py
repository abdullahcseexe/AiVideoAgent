import os
import uuid
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

CHROMA_DIR = "vector_db"
COLLECTION_NAME = "meeting_transcript"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"


def get_embeddings():
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
    )


# def build_vector_store(transcript: str) -> Chroma:
#     print("Building vector Store")

#     splitter = RecursiveCharacterTextSplitter(
#         chunk_size=500,
#         chunk_overlap=50,
#     )
#     chunks = splitter.split_text(transcript)

#     docs = [
#         Document(page_content=chunk, metadata={"chunk_index": i})
#         for i, chunk in enumerate(chunks)
#     ]

#     embeddings = get_embeddings()

#     # Clear chunks left over from a previous video, so answers only
#     # come from the current transcript.
#     try:
#         Chroma(
#             collection_name=COLLECTION_NAME,
#             embedding_function=embeddings,
#             persist_directory=CHROMA_DIR,
#         ).delete_collection()
#     except Exception:
#         pass

#     vector_store = Chroma.from_documents(
#         documents=docs,
#         embedding=embeddings,
#         collection_name=COLLECTION_NAME,
#         persist_directory=CHROMA_DIR,
#     )

#     return vector_store

def build_vector_store(transcript: str) -> Chroma:
    print("Building vector store")

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_text(transcript)

    docs = [
        Document(page_content=chunk, metadata={"chunk_index": i})
        for i, chunk in enumerate(chunks)
    ]

    return Chroma.from_documents(
        documents=docs,
        embedding=get_embeddings(),
        collection_name=f"transcript_{uuid.uuid4().hex[:12]}",
    )  # no persist_directory, so it stays in memory and is unique per session


def load_vector_store() -> Chroma:
    embeddings = get_embeddings()
    vector_store = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=CHROMA_DIR,
    )

    return vector_store


def get_retriever(vector_store: Chroma, k: int = 4):
    return vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": k},
    )