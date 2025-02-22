def retrieve(vectorstore, query):
    """Retrieve  most relevant text chunks for a query."""
    
    retriever = vectorstore.as_retriever(search_type='similarity')
    
    results = retriever.invoke(query)

    return results
