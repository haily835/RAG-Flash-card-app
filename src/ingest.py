import os
import json
import numpy as np
from sentence_transformers import SentenceTransformer
from langchain.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import pickle
from chromadb import Client  # Import ChromaDB client
from chromadb.config import Settings  # Import ChromaDB settings



def save_text(texts, vector_db_path="../models/embeddings/chroma_index"):
    """save text to retriver mapping"""
    with open(f"{vector_db_path}_texts.pkl", "wb") as f:
        pickle.dump(texts, f)

def load_documents(data_dir="./data"):
    """Load and preprocess documents from the data directory."""
    docs = []
    
    for file in os.listdir(data_dir):
        file_path = os.path.join(data_dir, file)
        
        if file.endswith(".pdf"):
            loader = PyPDFLoader(file_path)
            pages = loader.load()
            docs.extend([p for p in pages])
        
        elif file.endswith(".txt") or file.endswith(".md"):
            loader = TextLoader(file_path)
            pages = loader.load()
            docs.extend([p for p in pages])
        
        elif file.endswith(".json"):
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                docs.extend(data.values())  # Assuming JSON is a key-value dictionary
    
    return docs


def get_embedders(model_name="sentence-transformers/all-MiniLM-L6-v2"):
    """Get the embedding function from sentence_transformers"""
    return SentenceTransformer(model_name)
    
 
def chunk_text(documents, chunk_size=500, chunk_overlap=50):    
    """Chunk long documents into smaller segments."""
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size,
                                          chunk_overlap=chunk_overlap,
                                          length_function=len,
                                          separators=["\n\n", "\n", " "])
    
    return text_splitter.split_documents(documents)


def create_chroma_index(embeddings):
    """Create a ChromaDB index from embeddings."""
    client = Client(Settings(chroma_db_impl="duckdb+parquet", persist_directory="../models/embeddings/chroma_db"))  # Initialize ChromaDB client
    collection = client.create_collection("documents")  # Create a collection
    collection.add(embeddings)  # Add embeddings to the collection
    return collection

def main():
    """Ingest documents, generate embeddings, and store in ChromaDB."""
    print("Loading documents...")
    raw_texts = load_documents()

    print(f"Loaded {len(raw_texts)} documents. Chunking...")
    texts = chunk_text(raw_texts)
    embedder = get_embedders()
    print("Generating embeddings...")
    embeddings = embedder.encode(texts, convert_to_numpy=True)

    print("Creating ChromaDB index...")
    collection = create_chroma_index(embeddings)  # Use ChromaDB instead of FAISS
    
    save_text(texts)
    
    print("Ingestion complete!")

if __name__ == "__main__":
    main()
