from app import create_app
import os

app = create_app()
print("GROQ_API_KEY as seen after create_app():", repr(os.environ.get("GROQ_API_KEY")))