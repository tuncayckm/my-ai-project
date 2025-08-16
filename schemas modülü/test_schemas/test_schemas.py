import pytest
from pydantic import ValidationError
from schemas import ApplyKnowledgeRequest


def test_valid_request_minimal():
    """Tüm zorunlu alanlarla modelin doğru oluşturulduğunu test eder."""
    data = {
        "user_id": "user123",
        "prompt": "Some question?",
        "token": "abc123"
    }
    req = ApplyKnowledgeRequest(**data)
    assert req.user_id == "user123"
    assert req.prompt == "Some question?"
    assert req.token == "abc123"
    assert req.plugin_data is None
    assert req.max_context_tokens is None


def test_valid_request_with_all_fields():
    """Tüm alanlar dolu olduğunda modelin çalışmasını test eder."""
    data = {
        "user_id": "user123",
        "prompt": "Some question?",
        "token": "abc123",
        "plugin_data": {"key": "value"},
        "max_context_tokens": 100
    }
    req = ApplyKnowledgeRequest(**data)
    assert req.plugin_data == {"key": "value"}
    assert req.max_context_tokens == 100


def test_missing_required_fields():
    """Zorunlu alanlardan biri eksik olduğunda hata fırlatılmasını test eder."""
    with pytest.raises(ValidationError):
        ApplyKnowledgeRequest(prompt="Hi", token="abc123")

    with pytest.raises(ValidationError):
        ApplyKnowledgeRequest(user_id="u1", token="abc123")

    with pytest.raises(ValidationError):
        ApplyKnowledgeRequest(user_id="u1", prompt="Hello")


def test_invalid_field_types():
    """Alan tipleri yanlış verildiğinde hata fırlatılmasını test eder."""
    with pytest.raises(ValidationError):
        ApplyKnowledgeRequest(user_id=123, prompt="Hi", token="abc123")

    with pytest.raises(ValidationError):
        ApplyKnowledgeRequest(user_id="u1", prompt=["list"], token="abc123")

    with pytest.raises(ValidationError):
        ApplyKnowledgeRequest(user_id="u1", prompt="Hi", token=456)


def test_optional_fields_none():
    """Opsiyonel alanların None olması durumunu test eder."""
    data = {
        "user_id": "user123",
        "prompt": "Some question?",
        "token": "abc123",
        "plugin_data": None,
        "max_context_tokens": None
    }
    req = ApplyKnowledgeRequest(**data)
    assert req.plugin_data is None
    assert req.max_context_tokens is None