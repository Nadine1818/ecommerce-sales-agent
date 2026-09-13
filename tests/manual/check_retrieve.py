from app import create_app
from app.rag.retrieve import retrieve

app = create_app()
with app.app_context():
    results = retrieve("do you ship internationally?", top_k=2)
    for r in results:
        print(r["distance"], r["text"])