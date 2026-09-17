from app import create_app
from app.rag import get_all_items

app = create_app()
with app.app_context():
    faqs = get_all_items(item_type="faq")
    policies = get_all_items(item_type="policy")

    print(f"Total FAQs: {len(faqs)}")
    for f in faqs:
        print(f"  {f['id']}: {f['metadata'].get('question')}")

    print(f"\nTotal Policies: {len(policies)}")
    for p in policies:
        print(f"  {p['id']}: {p['metadata'].get('title')}")