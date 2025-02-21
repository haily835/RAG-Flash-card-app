import os
import json
import numpy as np
from sentence_transformers import SentenceTransformer
from langchain.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import pickle
from chromadb import PersistentClient  # Import ChromaDB client
import uuid



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
    
 
def chunk_text(documents, chunk_size=1500, chunk_overlap=250):    
    """Chunk long documents into smaller segments."""
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size,
                                          chunk_overlap=chunk_overlap,
                                          length_function=len,
                                          separators=["\n\n", "\n", " "])
    
    return text_splitter.split_documents(documents)


def create_chroma_index(db_path, texts):
    """Create a ChromaDB index using the default embedding function"""
    client = PersistentClient(path=db_path)
    unique_ids = set()
    unique_chunks = []
    ids = [str(uuid.uuid5(uuid.NAMESPACE_DNS, t)) for t in texts]

    for t, id in zip(texts, ids):     
        if id not in unique_ids:       
            unique_ids.add(id)
            unique_chunks.append(t)     
    
    collection = client.create_collection("documents")  # Create a collection
    collection.add(ids, documents=unique_chunks)  # Add embeddings to the collection
    return collection

