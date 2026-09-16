from app import create_app
from app.models import Product
from app.rag import delete_item, get_all_items, ingest_all
from app.rag.knowledge_data import FAQS, POLICIES
 

def run_ingestion():
    app = create_app()

    with app.app_context():
        products = []
        for p in Product.query.all():
            product_dict = p.to_dict()
            # add the category name to the product dict so we can embed it in the chunk text
            product_dict["category_name"] = p.category.name
            products.append(product_dict)

        ingest_all(products, "product")
        ingest_all(FAQS, "faq")
        ingest_all(POLICIES, "policy")

        # Reconciliation: ingest_all only pushes forward what currently
        # exists, it never removes anything. If a product was deleted
        # from the database since the last ingestion, its embedding would
        # otherwise sit in ChromaDB forever with no product behind it.
        current_product_ids = {str(p["id"]) for p in products}
        # gets chromaDB entries for all products, then deletes any that are no longer in the database
        existing_product_entries = get_all_items(item_type="product")
        stale_count = 0
        for entry in existing_product_entries:
            entry_product_id = entry["metadata"].get("id")
            if entry_product_id not in current_product_ids:
                delete_item(entry["id"])
                stale_count += 1
 
        print(
            f"Ingested {len(products)} products, {len(FAQS)} FAQs, "
            f"{len(POLICIES)} policies into ChromaDB."
            + (f" Removed {stale_count} stale product entries." if stale_count else "")
        )


if __name__ == "__main__":
    run_ingestion()