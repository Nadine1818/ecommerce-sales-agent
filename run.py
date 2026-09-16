import os

from app import create_app

# Ensure the data directory exists before creating the app, so that SQLite can create the database file there.
os.makedirs(os.path.join(os.path.dirname(__file__), "data"), exist_ok=True)
# Create the Flask app instance using the factory function.
app = create_app()

if __name__ == "__main__":
    # Warm up the BGE embedding model now, at startup, instead of letting
    # it load lazily on the first real chat message. It's a one-time cost
    # per server run either way (get_embedding_model is a singleton)
    # WERKZEUG_RUN_MAIN is only set to "true" in the real worker process,
    # so this guard stops the (expensive) warmup from running twice.
    if os.environ.get("WERKZEUG_RUN_MAIN") == "true":
        print("Warming up embedding model...")
        with app.app_context():
            from app.rag.store import get_embedding_model
 
            get_embedding_model()
        print("Ready.")

    app.run(debug=True, port=5000)