import os
from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
UPLOAD_FOLDER = "data/uploads"
VECTOR_STORE_PATH = "data/vector_store"
