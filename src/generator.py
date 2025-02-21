import os
import chromadb
import openai
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
from transformers import pipeline

# Load environment variables (API keys)
load_dotenv()

# Initialize ChromaDB client
chroma_client = chromadb.PersistentClient(path="../models/embeddings/chroma_db")
collection = chroma_client.get_or_create_collection(name="tech_docs")

# Load embedding model
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
embedder = SentenceTransformer(MODEL_NAME)

# OpenAI API key (or replace with Mistral/Llama API)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


def generate_response(query, retrieved_texts):
    """Generate an answer using an LLM with retrieved context."""
    context = "\n".join(retrieved_texts)
    
    prompt = f"""
    You are an expert AI assistant for technical interviews.
    
    **Question:** {query}
    
    **Context:** 
    {context}

    Based on the above information, provide a clear and concise answer:
    """

    response = openai.ChatCompletion.create(
        model="gpt-4",  # Use "gpt-3.5-turbo" for a cheaper option
        messages=[{"role": "system", "content": "You are a helpful AI assistant."},
                  {"role": "user", "content": prompt}],
        temperature=0.3  # Low temperature for factual accuracy
    )

    return response["choices"][0]["message"]["content"]

if __name__ == "__main__":
    user_query = input("Enter your interview question: ")

    # Retrieve relevant documents
    retrieved_docs = retrieve_context(user_query, k=3)

    if not retrieved_docs:
        print("\nNo relevant documents found. Answering without context...")
        retrieved_docs = ["No additional context available."]

    # Generate LLM response
    answer = generate_response(user_query, retrieved_docs)

    # Print results
    print("\n🔹 **AI Response:**")
    print(answer)

    print("\n📚 **Sources:**")
    for i, doc in enumerate(retrieved_docs, 1):
        print(f"{i}. {doc[:300]}...")  # Show first 300 chars of each source
