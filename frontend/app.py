import streamlit as st # type: ignore
import requests

API_URL = "http://localhost:8000"

st.set_page_config(page_title="Multimodal RAG", page_icon="🤖")
st.title("🤖 Multimodal RAG System")
st.caption("Upload a document and ask questions about it")

# ── Upload section ─────────────────────────────────────────
st.subheader("📄 Upload Document")
uploaded_file = st.file_uploader(
    "Choose a file", 
    type=["pdf", "txt", "csv", "png", "jpg"]
)

if uploaded_file:
    with st.spinner("Processing document..."):
        response = requests.post(
            f"{API_URL}/upload",
            files={"file": (uploaded_file.name,
                           uploaded_file.getvalue())}
        )
        result = response.json()

    if result["status"] == "success":
        st.success(f"✅ Uploaded **{result['filename']}**")
        col1, col2 = st.columns(2)
        col1.metric("Document Type", result["doc_type"])
        col2.metric("Chunks Created", result["chunks"])
        st.session_state['file_name'] = result['filename']
        st.info("⏳ Document uploaded! Wait 10-15 seconds for processing before asking questions.")
    else:
        st.error(f"❌ Error: {result['message']}")

# ── Question section ───────────────────────────────────────
st.subheader("💬 Ask a Question")
question = st.text_input("Type your question here...")

if st.button("Ask") and question:
    with st.spinner("Thinking..."):
        response = requests.post(
            f"{API_URL}/ask",
            json={
                "question": question,
                "file_name": st.session_state.get('file_name')
            }
        )
        result = response.json()

    if result["status"] == "success":
        # Confidence badge
        label = result["confidence_label"]
        score = result["confidence_score"]
        if label == "HIGH":
            st.success(f"✅ Confidence: {label} ({score})")
        elif label == "MEDIUM":
            st.warning(f"⚠️ Confidence: {label} ({score})")
        else:
            st.error(f"❌ Confidence: {label} ({score})")

        if result["confidence_warning"]:
            st.caption(result["confidence_warning"])

        st.markdown("### 🤖 Answer")
        st.write(result["answer"])
    else:
        st.error(f"❌ Error: {result['message']}")