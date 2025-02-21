import pytest
from retriever import retrieve
from generator import generate_response

def test_end_to_end_with_valid_query():
    """Test full pipeline: retrieval + response generation with valid input."""
    query = "What is deep learning?"
    
    # Step 1: Retrieve context
    retrieved_texts = retrieve(query, k=2)
    
    # Step 2: Generate response
    response = generate_response(query, retrieved_texts)

    assert isinstance(response, str)
    assert len(response) > 0  # Response should not be empty
    assert any(term in response.lower() for term in ["deep learning", "neural network"])  # Ensure topic relevance

def test_end_to_end_with_random_query():
    """Test full pipeline with a nonsense query (should gracefully fail)."""
    query = "asdkjfhawelrh"
    
    retrieved_texts = retrieve(query, k=2)
    response = generate_response(query, retrieved_texts)

    assert isinstance(response, str)
    assert len(response) > 0  # Should still return a fallback response
    assert "no relevant context" in response.lower() or "unable to find relevant" in response.lower()

def test_end_to_end_with_empty_query():
    """Test full pipeline with an empty question."""
    query = ""
    
    retrieved_texts = retrieve(query, k=2)
    response = generate_response(query, retrieved_texts)

    assert isinstance(response, str)
    assert len(response) > 0  # Should handle gracefully
    assert "please provide a valid query" in response.lower() or "question cannot be empty" in response.lower()
