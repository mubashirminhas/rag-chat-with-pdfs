import faiss
import os
import pickle

INDEX_PATH = "rag/index/faiss.index"
META_PATH = "rag/index/metadata.pkl"

def save_faiss_index(embeddings, metadata):
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)

    os.makedirs("rag/index", exist_ok=True)

    faiss.write_index(index, INDEX_PATH)

    with open(META_PATH, "wb") as f:
        pickle.dump(metadata, f)

    print("FAISS index saved successfully")
    return index


def load_faiss_index():
    if not os.path.exists(INDEX_PATH):
        return None, None

    index = faiss.read_index(INDEX_PATH)

    with open(META_PATH, "rb") as f:
        metadata = pickle.load(f)

    return index, metadata

def faiss_exists():
    return os.path.exists("faiss.index") and os.path.exists("metadata.pkl")