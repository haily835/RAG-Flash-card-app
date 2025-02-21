import pytest
from generator import generate_response

def test_generator_with_retrieved_text():
    """Test response generation with valid retrieved context."""
    query = "Explain neural networks"
    retrieved_texts = ["A neural network is a series of algorithms that mimic the human brain."]

    response = generate_response(query, retrieved_texts)

    assert isinstance(response, str)
    assert len(response) > 0  # Should generate a response
    assert "neural network" in response.lower()  # Ensure topic relevance

def test_generator_without_retrieved_text():
    """Test response generation when no context is available."""
    query = "Explain neural networks"
    retrieved_texts = []  # No context

    response = generate_response(query, retrieved_texts)

    assert isinstance(response, str)
    assert len(response) > 0  # Should still generate an answer

def test_generator_empty_query():
    """Test response generation with an empty query."""
    query = ""
    retrieved_texts = ["Machine learning is a field of AI."]

    response = generate_response(query, retrieved_texts)

    assert isinstance(response, str)
    assert len(response) > 0  # Should provide some response
