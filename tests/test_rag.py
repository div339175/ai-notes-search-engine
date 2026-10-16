"""Tests for ai-notes-search-engine RAG and fallback behavior."""
import os
import pytest
from src.rag import answer_question


def test_answer_question_missing_api_key_graceful_fallback(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    result = answer_question("What is normalization?", "Normalization is organizing data to reduce redundancy.")
    assert "GEMINI_API_KEY is not set" in result
    assert "Normalization is organizing" in result
