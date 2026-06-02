import os
import sys
import shutil
import boto3
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse

# Global cache
cached_index = None
cached_texts = None
cached_file = None

sys.path.append(os.path.dirname(__file__))

from ingestion import upload_file
from classifier import classify_document
from embedder import chunk_text, get_embedding, build_index, save_index, load_index, search
from bedrock_llm import ask_claude
from confidence_scorer import score_confidence

def load_index_from_s3(file_name):
    """Download index from S3 and load into FAISS - uses local cache if exists"""
    import faiss
    import pickle

    # Check local cache first
    local_chunks = f"models/faiss_index_texts.pkl"
    local_index = f"models/faiss_index.bin"
    cache_file = f"models/current_file.txt"

    # If same file already cached locally, use it
    if os.path.exists(cache_file):
        with open(cache_file, "r") as f:
            cached = f.read().strip()
        if cached == file_name and os.path.exists(local_chunks) and os.path.exists(local_index):
            print("✅ Using local cache")
            return load_index()

    # Otherwise download from S3
    s3 = boto3.client('s3', region_name='ap-south-1')
    bucket = os.getenv('AWS_BUCKET_NAME')

    chunks_key = f"indexes/{file_name}_chunks.pkl"
    embeddings_key = f"indexes/{file_name}_embeddings.pkl"

    chunks_obj = s3.get_object(Bucket=bucket, Key=chunks_key)
    chunks = pickle.loads(chunks_obj['Body'].read())

    embeddings_obj = s3.get_object(Bucket=bucket, Key=embeddings_key)
    embeddings = pickle.loads(embeddings_obj['Body'].read())

    import numpy as np
    embeddings_array = np.array(embeddings, dtype='float32')
    dimension = embeddings_array.shape[1]

    import faiss
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings_array)

    # Save locally
    save_index(index, chunks)
    os.makedirs("models", exist_ok=True)
    with open(cache_file, "w") as f:
        f.write(file_name)

    print(f"✅ Index loaded from S3 and cached locally for {file_name}")
    return index, chunks

app = FastAPI(title="Multimodal RAG System")

UPLOAD_DIR = "uploaded_docs"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# ── Upload + process document ──────────────────────────────
@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    try:
        # Save file locally
        local_path = os.path.join(UPLOAD_DIR, file.filename)
        with open(local_path, "wb") as f:
            shutil.copyfileobj(file.file, f)

        # Classify document
        doc_type = classify_document(local_path)

        # Upload to S3
        upload_file(local_path)

        # Extract text based on file type
        if local_path.endswith(".pdf"):
            from pypdf import PdfReader
            reader = PdfReader(local_path)
            raw_text = "\n".join(
                page.extract_text() for page in reader.pages
            )
            chunks = chunk_text(raw_text)

        elif local_path.endswith(".csv"):
            from tabular_pipeline import extract_text_from_csv
            chunks = extract_text_from_csv(local_path)

        elif local_path.endswith((".png", ".jpg", ".jpeg", ".webp")):
            import sys
            sys.path.append(os.path.dirname(__file__))
            from vision_pipeline import extract_text_from_image
            chunks = extract_text_from_image(local_path)
        else:
            with open(local_path, "r", encoding="utf-8") as f:
                raw_text = f.read()
            chunks = chunk_text(raw_text)
        

        return JSONResponse({
            "status": "success",
            "filename": file.filename,
            "doc_type": doc_type,
            "chunks": len(chunks)
        })

    except Exception as e:
        return JSONResponse({"status": "error", "message": str(e)}, status_code=500)


# ── Ask a question ─────────────────────────────────────────
@app.post("/ask")
async def ask_question(payload: dict):
    try:
        question = payload.get("question")
        file_name = payload.get("file_name")
        if not question:
            return JSONResponse({"error": "No question provided"}, status_code=400)

        # Load index from S3 if file_name provided
        global cached_index, cached_texts, cached_file

        # Use cache if same file
        if file_name and file_name == cached_file and cached_index is not None:
            index, texts = cached_index, cached_texts
            print("✅ Using cached index")
        else:
            try:
                if file_name:
                    index, texts = load_index_from_s3(file_name)
                    cached_index = index
                    cached_texts = texts
                    cached_file = file_name
                else:
                    index, texts = load_index()
            except Exception:
                index, texts = load_index()
        # Retrieve chunks
        chunks = search(question, index, texts, top_k=3)

        # Ask Claude
        # Score confidence
        score, label, warning = score_confidence(question, chunks)

        # Ask Claude
        answer = ask_claude(question, chunks)

        return JSONResponse({
            "status": "success",
            "question": question,
            "answer": answer,
            "confidence_score": round(score, 2),
            "confidence_label": label,
            "confidence_warning": warning
        })

    except Exception as e:
        return JSONResponse({"status": "error", "message": str(e)}, status_code=500)


# ── Health check ───────────────────────────────────────────
@app.get("/")
def root():
    return {"message": "Multimodal RAG System is running!"}