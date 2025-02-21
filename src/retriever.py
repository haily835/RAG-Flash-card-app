import chromadb

def retrieve(chroma_db_path, query, k=3):
    """Retrieve top-k most relevant text chunks for a query."""


    chroma_client = chromadb.PersistentClient(path=chroma_db_path)
    collection = chroma_client.get_or_create_collection(name="documents")
    
    
    results = collection.query(
        query_texts=[query],
        n_results=k
    )

    return results["documents"] if "documents" in results else []
