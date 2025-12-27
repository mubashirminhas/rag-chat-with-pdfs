from flask import Flask, render_template, request, redirect, url_for, session
import os
import numpy as np
from datetime import datetime
from collections import defaultdict
from config import UPLOAD_FOLDER
from rag.pdf_loader import load_pdfs
from rag.chunker import chunk_text
from rag.embeddings import generate_embeddings
from rag.vector_store import save_faiss_index, load_faiss_index, faiss_exists
from openai import OpenAI
from dotenv import load_dotenv

# -------------------------------
# Load environment variables
# -------------------------------
load_dotenv()

# -------------------------------
# OpenAI client
# -------------------------------
client = OpenAI()

# -------------------------------
# Flask app setup
# -------------------------------
app = Flask(__name__)
app.secret_key = "rag-chat-secret-key"

# Folders for uploads and vector store
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
VECTOR_STORE_FOLDER = "data/vector_store"

# Ensure folders exist at startup
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(VECTOR_STORE_FOLDER, exist_ok=True)

# -------------------------------
# Home page
# -------------------------------
@app.route("/", methods=["GET"])
def index():
    if "chat_history" not in session:
        session["chat_history"] = []

    return render_template(
        "chat.html",
        chat_history=session["chat_history"]
    )

# -------------------------------
# Upload PDFs
# -------------------------------
@app.route("/upload", methods=["POST"])
def upload_pdfs():
    if faiss_exists():
        print("FAISS index already exists — skipping embedding generation")
        return redirect(url_for("index"))

    files = request.files.getlist("pdfs")
    file_paths = []

    for file in files:
        if file.filename.endswith(".pdf"):
            file_path = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
            file.save(file_path)
            file_paths.append(file_path)

    documents = load_pdfs(file_paths)
    chunks = chunk_text(documents)
    embeddings = generate_embeddings(chunks)

    # Store text + metadata together
    faiss_data = [
        {
            "text": chunk["text"],
            "metadata": {
                "source": chunk["metadata"].get("source"),
                "page": chunk["metadata"].get("page")
            }
        }
        for chunk in chunks
    ]

    save_faiss_index(embeddings, faiss_data)
    print("FAISS index created successfully")
    return redirect(url_for("index"))

# -------------------------------
# Ask Question
# -------------------------------
@app.route("/ask", methods=["POST"])
def ask_question():
    query = request.form.get("question")
    if not query:
        return redirect(url_for("index"))

    # Generate query embedding
    query_embedding = generate_embeddings([{"text": query}])[0]

    index, metadata = load_faiss_index()
    k = 5
    D, I = index.search(np.array([query_embedding]), k)

    retrieved_chunks = [metadata[i] for i in I[0]]
    context = "\n\n".join(chunk["text"] for chunk in retrieved_chunks)

    # Build messages with chat history
    messages = [{"role": "system", "content": "You are a helpful assistant."}]
    for turn in session["chat_history"]:
        messages.append({"role": "user", "content": turn["question"]})
        messages.append({"role": "assistant", "content": turn["answer"]})

    messages.append({
        "role": "user",
        "content": f"""
Answer the question using ONLY the following context:

{context}

Question: {query}
"""
    })

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        max_tokens=300
    )

    answer = response.choices[0].message.content.strip()

    # Merge sources nicely
    source_map = defaultdict(set)
    for chunk in retrieved_chunks:
        src = chunk["metadata"].get("source")
        page = chunk["metadata"].get("page")
        if src and page:
            source_map[src].add(page)

    sources_str = ", ".join([
        f"{src} (Page: {', '.join(map(str, sorted(pages)))})"
        for src, pages in source_map.items()
    ])

    # Save conversation
    session["chat_history"].append({
        "question": query,
        "answer": answer,
        "sources": sources_str,
        "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    })
    session.modified = True

    return render_template(
        "chat.html",
        chat_history=session["chat_history"],
        context=context
    )

# -------------------------------
# Clear Chat
# -------------------------------
@app.route("/clear", methods=["POST"])
def clear_chat():
    session.pop("chat_history", None)
    return redirect(url_for("index"))

# -------------------------------
# Main (Local Test Only)
# -------------------------------
if __name__ == "__main__":
    app.run(debug=False, use_reloader=False)
