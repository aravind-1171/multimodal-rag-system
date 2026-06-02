import os
from groq import Groq
from embedder import load_index, search

import os
from dotenv import load_dotenv

import os
from dotenv import load_dotenv
load_dotenv(dotenv_path=r"C:\Users\USER\multimodal-rag\.env")
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

def ask_claude(question, context_chunks):
    context = "\n\n".join(context_chunks)
    prompt = f"""You are a helpful AI assistant.
Use the following context to answer the question.
If the answer is not in the context, say "I don't have enough information to answer that."

Context:
{context}

Question: {question}
Answer:"""

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content

def rag_query(question):
    print(f"\n🔍 Question: {question}")
    index, texts = load_index()
    chunks = search(question, index, texts, top_k=3)
    print(f"📚 Retrieved {len(chunks)} chunks from FAISS")
    answer = ask_claude(question, chunks)
    print(f"\n🤖 Answer:\n{answer}")
    return answer

if __name__ == "__main__":
    rag_query("What is RAG and how does it work?")