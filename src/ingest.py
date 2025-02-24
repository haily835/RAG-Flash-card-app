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

def load_documents(file_path="./data/Paper.pdf"):
    """Load and preprocess documents from the data directory."""
    docs = []


    loader = PyPDFLoader(file_path)

    pages = loader.load()
    docs.extend([p for p in pages])
    
    
    return docs





def get_embedder(api_key, model_name="text-embedding-3-small"):
    """Get the embedding function from sentence_transformers"""
    return OpenAIEmbeddings(model=model_name, api_key=api_key)
    
 
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
    try:
        vec = Chroma(persist_directory=vectorstore_path, 
                  embedding_function=embedding_function, 
                  collection_name=clean_filename(file_name))
        return vec
    except:
        return None


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

def create_vectorstore(file_name, embedding_function , texts, vectorstore_path='db'):
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
        persist_directory=vectorstore_path,  # Where to save data locally, remove if not necessary
    )
    return vector_store

