import os
import numpy as np
from openai import OpenAI

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def generate_embeddings(chunks, model="text-embedding-3-small"):
    texts = [c["text"] for c in chunks if c["text"].strip()]

    response = client.embeddings.create(
        model=model,
        input=texts
    )

    vectors = [item.embedding for item in response.data]
    return np.array(vectors, dtype="float32")
