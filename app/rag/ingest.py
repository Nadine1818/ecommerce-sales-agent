# turns raw business data (products, FAQs, policies) into embedded chunks in ChromaDB, and exposes add/update/delete so the dashboard
# can manage the knowledge base without touching code
# it uses metadata to tag each chunk with its type (product, FAQ, policy) so that retrieve.py can filter by type later

from app.rag.store import get_collection, get_embedding_model


def _build_chunk_text(item_type: str, data: dict) -> str:
    # builds a single string from the raw data for a product, FAQ, or policy, which is what we embed and store in ChromaDB
    if item_type == "product":
        return (
            f"Product: {data['name']}. "
            f"Category: {data['category_name']}. "
            f"Price: ${data['price']}. "
            f"Description: {data['description']}"
        )
    elif item_type == "faq":
        return f"Question: {data['question']} Answer: {data['answer']}"
    elif item_type == "policy":
        return f"{data['title']}: {data['content']}"
    else:
        raise ValueError(f"Unknown item_type: {item_type}")


def add_or_update_item(item_id: str, item_type: str, data: dict):
    # dashboard admin calls this to add or update a product, FAQ, or policy in the knowledge base
    # it adds if item id doesn't exist, or updates if it does.
    collection = get_collection()
    text = _build_chunk_text(item_type, data)

    embedding_model = get_embedding_model()
    vector = embedding_model.embed_documents([text])[0]

    # metadata is what lets retrieve.py filter by type later
    metadata = {"type": item_type, **{k: str(v) for k, v in data.items()}}

    # either adds a new item or updates an existing one, depending on whether the item_id already exists in the collection
    collection.upsert(
        ids=[item_id], # to check if item already exists 
        documents=[text], # to retrieve the text later
        embeddings=[vector], # to search for similar items later 
        metadatas=[metadata], # to filter by type later
    )


def delete_item(item_id: str):
    # dashboard admin calls this to delete a product, FAQ, or policy from the knowledge base
    collection = get_collection()
    collection.delete(ids=[item_id])


def ingest_all(items: list, item_type: str, id_field: str = "id"):
    # generic bulk ingestion function for products, FAQs, or policies. It takes a list of items and calls add_or_update_item for each one
    for item in items:
        add_or_update_item(
            item_id=f"{item_type}-{item[id_field]}",
            item_type=item_type,
            data=item,
        )