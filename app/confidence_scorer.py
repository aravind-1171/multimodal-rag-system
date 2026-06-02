import numpy as np
from embedder import get_embedding, load_index, search

def cosine_similarity(vec1, vec2):
    """Calculate cosine similarity between two vectors"""
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return dot_product / (norm1 * norm2)

def score_confidence(question, retrieved_chunks):
    """
    Scores how confident we are in the retrieved chunks
    by measuring similarity between question and chunks.
    
    Returns:
        score: 0.0 to 1.0
        label: HIGH / MEDIUM / LOW
        warning: message if low confidence
    """
    if not retrieved_chunks:
        return 0.0, "LOW", "⚠️ No relevant chunks found"

    # Get question embedding
    question_vec = get_embedding(question)

    # Get chunk embeddings and calculate similarities
    similarities = []
    for chunk in retrieved_chunks:
        chunk_vec = get_embedding(chunk)
        sim = cosine_similarity(question_vec, chunk_vec)
        similarities.append(sim)

    # Average similarity score
    avg_score = float(np.mean(similarities))
    max_score = float(np.max(similarities))

    # Label based on score
    # Label based on score
    if max_score >= 0.25:
        label = "HIGH"
        warning = None
    elif max_score >= 0.10:
        label = "MEDIUM"
        warning = "⚠️ Moderate confidence — answer may be incomplete"
    else:
        label = "LOW"
        warning = "⚠️ Low confidence — answer may not be reliable"

    print(f"📊 Confidence Score: {max_score:.2f} → {label}")
    if warning:
        print(warning)

    return max_score, label, warning


# ── Test ───────────────────────────────────────────────────
if __name__ == "__main__":
    index, texts = load_index()

    # High confidence question — in our documents
    q1 = "What does RAG stand for?"
    chunks1 = search(q1, index, texts, top_k=3)
    score, label, warning = score_confidence(q1, chunks1)
    print(f"Question: {q1}")
    print(f"Result: {score:.2f} → {label}\n")

    # Low confidence question — not in our documents
    q2 = "What is the capital of France?"
    chunks2 = search(q2, index, texts, top_k=3)
    score, label, warning = score_confidence(q2, chunks2)
    print(f"Question: {q2}")
    print(f"Result: {score:.2f} → {label}")