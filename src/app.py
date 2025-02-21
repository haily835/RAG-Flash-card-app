import os
import chromadb
import openai
import streamlit as st
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize ChromaDB client
chroma_client = chromadb.PersistentClient(path="models/embeddings/chroma_db")
collection = chroma_client.get_or_create_collection(name="tech_docs")

# Load embedding model
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
embedder = SentenceTransformer(MODEL_NAME)

# OpenAI API key
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

st.set_page_config(page_title="AI RAG Chatbot", layout="wide")
st.title("💬 AI Chatbot with Retrieval-Augmented Generation (RAG)")

def retrieve_context(query, k=3):
    """Retrieve top-k most relevant text chunks from ChromaDB."""
    query_embedding = embedder.encode([query], convert_to_numpy=True).tolist()[0]
    results = collection.query(query_embeddings=[query_embedding], n_results=k)
    return results["documents"][0] if "documents" in results else []

def generate_response(query, retrieved_texts):
    """Generate a response using GPT with retrieved context."""
    context = "\n".join(retrieved_texts) if retrieved_texts else "No relevant context found."

    prompt = f"""
    You are an expert AI assistant. Answer the following question based on the provided context.

    **Question:** {query}
    **Context:** {context}

    Provide a concise and well-structured response:
    """

    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[{"role": "system", "content": "You are a helpful AI assistant."},
                  {"role": "user", "content": prompt}],
        temperature=0.3
    )

    return response["choices"][0]["message"]["content"]

# Streamlit UI
query = st.text_input("🔍 Ask me anything:")
if query:
    with st.spinner("Retrieving information..."):
        retrieved_docs = retrieve_context(query, k=3)
        response = generate_response(query, retrieved_docs)

    # Display answer
    st.subheader("🤖 AI Response")
    st.write(response)

    # Display sources
    st.subheader("📚 Sources")
    for i, doc in enumerate(retrieved_docs, 1):
        st.write(f"**{i}.** {doc[:300]}...")  # Show snippet of source
