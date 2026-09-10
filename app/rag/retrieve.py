# Retrieve relevant information from the knowledge base. called by the agent's tool code to get the most relevant chunks for a given query.

from app.rag.store import get_collection, get_embedding_model


def retrieve(query: str, top_k: int = 3, item_type: str = None) -> list[dict]:
    # returns top_k most relevant chunks, filtered by item_type if provided (product, FAQ, policy), for a given query string

    collection = get_collection()

    # query uses the same embedding model as ingest.py, so that the query vector is in the same space as the stored vectors.
    embedding_model = get_embedding_model()
    query_vector = embedding_model.embed_query(query)

    # ChromaDB's "where" filter narrows the search to only entries whose metadata matches
    where_filter = {"type": item_type} if item_type else None

    results = collection.query(
        query_embeddings=[query_vector],
        n_results=top_k,
        where=where_filter,
    )

    matches = []
    documents = results["documents"][0]
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]

    for text, metadata, distance in zip(documents, metadatas, distances):
        matches.append({"text": text, "metadata": metadata, "distance": distance})

    return matches