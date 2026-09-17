import sys

from app import create_app
from app.rag.store import get_collection

app = create_app()
with app.app_context():
    if len(sys.argv) < 2:
        print("Usage: python check_rag_item.py <item_id>")
        print("Example: python check_rag_item.py faq-2")
        sys.exit(1)

    item_id = sys.argv[1]
    collection = get_collection()
    result = collection.get(ids=[item_id])

    if not result["ids"]:
        print(f"'{item_id}' does not exist in ChromaDB.")
    else:
        print(f"id: {result['ids'][0]}")
        print(f"embedded text: {result['documents'][0]}")
        print(f"metadata: {result['metadatas'][0]}")