import boto3
import json
import faiss
import numpy as np
import pickle
import os

# ── Chunk text ─────────────────────────────────────────────
def chunk_text(text, chunk_size=500, overlap=50):
    """
    Splits text into overlapping chunks.
    chunk_size = number of characters per chunk
    overlap    = how many characters to repeat between chunks
                 (keeps context at boundaries)
    """
    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start += chunk_size - overlap  # slide forward with overlap

    print(f"✅ Text split into {len(chunks)} chunks")
    return chunks

# ── Bedrock client ─────────────────────────────────────────
bedrock = boto3.client('bedrock-runtime', region_name='ap-south-1')

# ── Get embedding from Titan ───────────────────────────────
def get_embedding(text):
    body = json.dumps({"inputText": text})
    response = bedrock.invoke_model(
        modelId="amazon.titan-embed-text-v2:0",
        contentType="application/json",
        accept="application/json",
        body=body
    )
    result = json.loads(response['body'].read())
    return np.array(result['embedding'], dtype='float32')

# ── Build FAISS index from list of texts ───────────────────
def build_index(texts):
    print("⏳ Generating embeddings...")
    embeddings = [get_embedding(t) for t in texts]
    embeddings = np.stack(embeddings)

    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    print(f"✅ FAISS index built with {index.ntotal} vectors")
    return index, embeddings

# ── Save index and texts ───────────────────────────────────
def save_index(index, texts, path="models/faiss_index"):
    os.makedirs("models", exist_ok=True)
    faiss.write_index(index, path + ".bin")
    with open(path + "_texts.pkl", "wb") as f:
        pickle.dump(texts, f)
    print("✅ Index saved to models/")

# ── Load index and texts ───────────────────────────────────
def load_index(path="models/faiss_index"):
    index = faiss.read_index(path + ".bin")
    with open(path + "_texts.pkl", "rb") as f:
        texts = pickle.load(f)
    return index, texts

# ── Search index ───────────────────────────────────────────
def search(query, index, texts, top_k=3):
    query_vec = get_embedding(query).reshape(1, -1)
    distances, indices = index.search(query_vec, top_k)
    results = [texts[i] for i in indices[0] if i < len(texts)]
    return results

# ── Test ───────────────────────────────────────────────────
if __name__ == "__main__":
    # Sample texts to index
    sample_texts = [
        "Amazon Bedrock provides access to foundation models via API.",
        "FAISS is a library for efficient similarity search.",
        "RAG stands for Retrieval Augmented Generation.",
        "Python is a popular programming language for data science.",
        "LangChain helps build LLM-powered applications.",
    ]

    # Build and save index
    index, embeddings = build_index(sample_texts)
    save_index(index, sample_texts)

    # Test search
    query = "What is RAG?"
    print(f"\n🔍 Query: {query}")
    results = search(query, index, sample_texts)
    print("📚 Top results:")
    for i, r in enumerate(results):
        print(f"  {i+1}. {r}")