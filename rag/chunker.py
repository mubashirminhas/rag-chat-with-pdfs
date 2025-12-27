from rag.text_cleaner import clean_text

def chunk_text(documents, chunk_size=500, overlap=100):
    """
    Splits text into overlapping chunks while preserving metadata.
    """
    chunks = []

    for doc in documents:
        text = doc["text"]
        metadata = doc["metadata"]

        start = 0
        while start < len(text):
            end = start + chunk_size
            chunk_text = text[start:end]

            chunks.append({
                "text": clean_text(chunk_text),
                "metadata": metadata
            })

            start += chunk_size - overlap

    return chunks
