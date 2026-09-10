from app import create_app
from app.models import Product
from app.rag.ingest import ingest_all
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

        print(
            f"Ingested {len(products)} products, {len(FAQS)} FAQs, "
            f"{len(POLICIES)} policies into ChromaDB."
        )


if __name__ == "__main__":
    run_ingestion()