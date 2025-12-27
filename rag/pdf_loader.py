from PyPDF2 import PdfReader

def load_pdfs(file_paths):
    """
    Reads multiple PDF files and extracts text page by page.
    Returns a list of documents with metadata.
    """
    documents = []

    for path in file_paths:
        reader = PdfReader(path)

        for page_num, page in enumerate(reader.pages):
            text = page.extract_text()

            if text and text.strip():
                documents.append({
                    "text": text,
                    "metadata": {
                        "source": path.split("/")[-1],
                        "page": page_num + 1
                    }
                })

    return documents
