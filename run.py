import os

from app import create_app

# Ensure the data directory exists before creating the app, so that SQLite can create the database file there.
os.makedirs(os.path.join(os.path.dirname(__file__), "data"), exist_ok=True)
# Create the Flask app instance using the factory function.
app = create_app()

if __name__ == "__main__":
    app.run(debug=True, port=5000)