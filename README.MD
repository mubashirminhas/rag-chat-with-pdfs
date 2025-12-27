# 📄 RAG_DOC_CHAT – Chat with Your PDFs

[![Python](https://img.shields.io/badge/Python-3.11-blue)](https://www.python.org/)
[![Flask](https://img.shields.io/badge/Flask-3.1.2-green)](https://flask.palletsprojects.com/)
[![OpenAI](https://img.shields.io/badge/OpenAI-GPT-4.0-orange)](https://openai.com/)

**RAG_DOC_CHAT** is a portfolio-ready Retrieval-Augmented Generation (RAG) web application built with **Flask**, **FAISS**, and **OpenAI GPT models**. Upload PDFs, and interactively ask questions. The app intelligently extracts context and provides precise answers with **sources and page references**.

---

## Features

- Upload multiple PDFs at once.
- Extract meaningful chunks and generate embeddings automatically.
- Chat with your documents using AI.
- Answers include **source PDF filenames and page numbers**.
- Chat history with timestamps is maintained.
- Latest questions appear on top for a professional UX.
- Clean and responsive interface using **Bootstrap + custom CSS**.

## Project Structure

RAG_DOC_CHAT/
│
├─ data/
│ ├─ uploads/ # PDF files uploaded by user (ignored in git)
│ └─ vector_store/ # FAISS vector indices (ignored in git)
│
├─ rag/
│ ├─ init.py
│ ├─ chunker.py
│ ├─ embeddings.py
│ ├─ pdf_loader.py
│ ├─ qa_chain.py
│ ├─ retriever.py
│ ├─ text_cleaner.py
│ └─ vector_store.py
│
├─ static/
│ └─ style.css
│
├─ templates/
│ ├─ answer.html
│ └─ chat.html
│
├─ app.py
├─ config.py
├─ requirements.txt
├─ README.md
└─ .gitignore

yaml
Copy code

---

## Installation

1. Clone the repository:
```bash
- git clone <your-github-repo-url>
- cd RAG_DOC_CHAT
- Create and activate a virtual environment:

## bash
- Copy code
- python -m venv venv

# Windows
- venv\Scripts\activate

# macOS/Linux
- source venv/bin/activate

## Install dependencies:
- bash
- Copy code
- pip install -r requirements.txt
- Add your OpenAI API key in config.py or a .env file:
- python
- Copy code
- OPENAI_API_KEY = "your_api_key_here"
- UPLOAD_FOLDER = "data/uploads"

## Running the Application
- bash
- Copy code
- python app.py
- Then open http://127.0.0.1:5000 in your browser.

## Demo / Screenshots
- Upload PDFs and Ask Questions

- Chat Interface

- Answer with Sources

## Usage
- Upload one or more PDFs using the Upload Documents section.
- Ask any question about the uploaded documents.
- Answers will include:
- Extracted content from PDFs.
- Source PDF filename(s) and page numbers.
- Chat history will show the latest questions on top with timestamps.
- Highlight the latest question in chat.
- Collapse older messages.

## Future Enhancements
- Add more detailed PDF sources per answer.
- Persist chat history in SQLite.
- Track token usage per query for cost efficiency.

## Requirements
- Python 3.11+
- Flask 3.1.2
- OpenAI API key
- FAISS (CPU or GPU)
- PyPDF2
- Numpy, Pandas, Tiktoken

## Install all dependencies using:

- bash
- Copy code
- pip install -r requirements.txt

## Notes
- Uploaded PDFs and FAISS indices are not included in the repo; they are stored locally in data/uploads and data/vector_store.
- Keep your API key secret. Do not commit it to GitHub.
- Tested on Windows 11 / Python 3.11.

##nReferences
- OpenAI API
- FAISS – Facebook AI Similarity Search
- Flask Documentation