import unicodedata

def clean_text(text: str) -> str:
    """
    Cleans text to be UTF-8 safe for embeddings.
    """
    if not isinstance(text, str):
        return ""

    # Normalize unicode characters
    text = unicodedata.normalize("NFKD", text)

    # Remove invalid surrogate characters
    text = text.encode("utf-8", "ignore").decode("utf-8")

    return text
