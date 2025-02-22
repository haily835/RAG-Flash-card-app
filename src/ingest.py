import os
import json
import numpy as np
from langchain.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import pickle
from langchain_chroma import Chroma
import uuid
from langchain_openai import OpenAIEmbeddings
import re


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





def get_embedder(model_name="text-embedding-3-small"):
    """Get the embedding function from sentence_transformers"""
    return OpenAIEmbeddings(model=model_name)
    
 
def chunk_text(documents, chunk_size=1500, chunk_overlap=250):    
    """Chunk long documents into smaller segments."""
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size,
                                          chunk_overlap=chunk_overlap,
                                          length_function=len,
                                          separators=["\n\n", "\n", " "])
    
    return text_splitter.split_documents(documents)

def load_vectorstore(file_name, embedding_function, vectorstore_path="db"):

    """
    Load a previously saved Chroma vector store from disk.

    :param file_name: The name of the file to load (without the path)
    :param api_key: The OpenAI API key used to create the vector store
    :param vectorstore_path: The path to the directory where the vector store was saved (default: "db")
    
    :return: A Chroma vector store object
    """
    return Chroma(persist_directory=vectorstore_path, 
                  embedding_function=embedding_function, 
                  collection_name=clean_filename(file_name))


def clean_filename(filename):
    """
    Remove "(number)" pattern from a filename 
    (because this could cause error when used as collection name when creating Chroma database).

    Parameters:
        filename (str): The filename to clean

    Returns:
        str: The cleaned filename
    """
    # Regular expression to find "(number)" pattern
    new_filename = re.sub(r'\s\(\d+\)', '', filename)
    return new_filename

def create_chroma_index(db_path, file_name, embedding_function , texts):
    """Create a ChromaDB index using the default embedding function"""
  
    
    unique_ids = set()
    unique_chunks = []
    ids = [str(uuid.uuid5(uuid.NAMESPACE_DNS, t.page_content)) for t in texts]



    for t, id in zip(texts, ids):     
        if id not in unique_ids:
            unique_ids.add(id)
            unique_chunks.append(t)
    vector_store = Chroma.from_documents(
        collection_name=clean_filename(file_name),
        ids=list(unique_ids), 
        documents=unique_chunks, 
        embedding=embedding_function,
        persist_directory=db_path,  # Where to save data locally, remove if not necessary
    )
    return vector_store

