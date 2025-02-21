import pytest
from retriever import retrieve

def test_retriever_with_valid_query():
    """Test retriever with a valid question."""
    query = "What is machine learning?"
    results = retrieve(query, k=2)

    assert isinstance(results, list)
    assert len(results) > 0  # Should retrieve at least one result
    assert isinstance(results[0], str)  # Results should be text

def test_retriever_with_invalid_query():
    """Test retriever with an unrelated/random query."""
    query = "qwertyuiop123456"  # Likely no match in DB
    results = retrieve(query, k=2)

    assert isinstance(results, list)
    assert len(results) == 0  # Should return an empty list

def test_retriever_empty_query():
    """Test retriever with an empty query string."""
    query = ""
    results = retrieve(query, k=2)

    assert isinstance(results, list)
    assert len(results) == 0  # Should return an empty list
