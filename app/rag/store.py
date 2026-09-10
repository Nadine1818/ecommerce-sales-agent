# chromaDB connection layer, owns the connection to the embedding model and vector store
import os
import chromadb
from langchain_huggingface import HuggingFaceEmbeddings

# runs locally with no API key needed.
_EMBEDDING_MODEL_NAME = "BAAI/bge-small-en-v1.5"

# Lazy-singleton pattern for the embedding model and ChromaDB collection.
# only load the model and connect to the DB once, then reuse those objects for all subsequent calls.
_embedding_model = None
_chroma_client = None
_collection = None


def get_embedding_model():
    # Loaded once, reused everywhere. Loading a HuggingFace model is slow
    # as it reads weights from disk, so its done only once and the model object is reused for all subsequent calls.
    global _embedding_model
    if _embedding_model is None:
        _embedding_model = HuggingFaceEmbeddings(model_name=_EMBEDDING_MODEL_NAME)
    return _embedding_model


def get_collection():
    # Lazy-singleton pattern for the ChromaDB collection.
    global _chroma_client, _collection
    if _collection is None:
        basedir = os.path.abspath(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
        persist_dir = os.path.join(basedir, "data", "chroma")
        os.makedirs(persist_dir, exist_ok=True)
        # client is the connection to the ChromaDB vector store, which is persisted to disk in the "data/chroma" directory.
        # collection is the specific collection of vectors that we will use for our e-commerce knowledge base.
        _chroma_client = chromadb.PersistentClient(path=persist_dir)
        _collection = _chroma_client.get_or_create_collection(name="ecommerce_knowledge_base")
    return _collection