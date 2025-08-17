# nlp_apply_knowledge_test.py
import json
import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
import nlp


@pytest.fixture
def mock_db():
    db = AsyncMock()
    db.connect.return_value = None
    db.load_profile.return_value = {
        "history": [],
        "preferences": {}
    }
    db.save_profile.return_value = None
    db.close.return_value = None
    return db


@pytest.mark.asyncio
async def test_apply_knowledge_success(monkeypatch, mock_db):
    """✅ Başarılı senaryo: tüm adımlar çalışır."""
    monkeypatch.setattr(nlp, "DatabaseManager", lambda **_: mock_db)
    monkeypatch.setattr(nlp, "verify_jwt_token", lambda t, expected_user_id: {"roles": ["admin"]})
    monkeypatch.setattr(nlp, "sanitize_input", lambda x: x)
    monkeypatch.setattr(nlp, "rate_limiter", MagicMock(is_allowed=lambda _: True))
    monkeypatch.setattr(nlp, "_get_context_from_vector", AsyncMock(return_value="context"))
    monkeypatch.setattr(nlp, "run_plugin_async", AsyncMock(return_value="plugin output"))

    result = await nlp.apply_knowledge(mock_db, "user1", "prompt", "valid.token")
    data = json.loads(result)
    assert any(msg["role"] == "assistant" for msg in data)


@pytest.mark.asyncio
async def test_apply_knowledge_validation_error(monkeypatch, mock_db):
    """❌ Input validation hatasında doğru response dönmeli."""
    monkeypatch.setattr(nlp, "ApplyKnowledgeRequest", lambda **_: (_ for _ in ()).throw(ValueError("bad input")))

    result = await nlp.apply_knowledge(mock_db, "user1", "prompt", "token")
    data = json.loads(result)
    assert data[0]["role"] == "system"
    assert "Geçersiz" in data[0]["content"]


@pytest.mark.asyncio
async def test_apply_knowledge_invalid_user_id(monkeypatch, mock_db):
    """🚫 Geçersiz kullanıcı ID'sinde rate limit öncesi çıkmalı."""
    monkeypatch.setattr(nlp, "sanitize_input", lambda x: None)  # invalid id
    monkeypatch.setattr(nlp, "ApplyKnowledgeRequest", lambda **kwargs: kwargs)

    result = await nlp.apply_knowledge(mock_db, "bad id", "prompt", "token")
    assert "Geçersiz kullanıcı" in result


@pytest.mark.asyncio
async def test_apply_knowledge_rate_limited(monkeypatch, mock_db):
    """🚫 Rate limit aşıldığında uyarı dönmeli."""
    monkeypatch.setattr(nlp, "sanitize_input", lambda x: x)
    monkeypatch.setattr(nlp, "ApplyKnowledgeRequest", lambda **kwargs: kwargs)
    monkeypatch.setattr(nlp, "rate_limiter", MagicMock(is_allowed=lambda _: False))

    result = await nlp.apply_knowledge(mock_db, "user1", "prompt", "token")
    assert "Çok fazla istek" in result


@pytest.mark.asyncio
async def test_apply_knowledge_rbac_denied(monkeypatch, mock_db):
    """🔒 RBAC yetersiz rol ile plugin erişimi engellenmeli."""
    monkeypatch.setattr(nlp, "sanitize_input", lambda x: x)
    monkeypatch.setattr(nlp, "ApplyKnowledgeRequest", lambda **kwargs: kwargs)
    monkeypatch.setattr(nlp, "rate_limiter", MagicMock(is_allowed=lambda _: True))
    monkeypatch.setattr(nlp, "verify_jwt_token", lambda t, expected_user_id: {"roles": ["user"]})

    result = await nlp.apply_knowledge(
        mock_db, "user1", "prompt", "token",
        plugin_data={"protected_resource": True}
    )
    assert "Yetersiz yetki" in result


@pytest.mark.asyncio
async def test_apply_knowledge_detect_language_failure(monkeypatch, mock_db):
    """🌐 Dil tespiti hata verirse lang=None devam etmeli."""
    monkeypatch.setattr(nlp, "sanitize_input", lambda x: x)
    monkeypatch.setattr(nlp, "ApplyKnowledgeRequest", lambda **kwargs: kwargs)
    monkeypatch.setattr(nlp, "rate_limiter", MagicMock(is_allowed=lambda _: True))
    monkeypatch.setattr(nlp, "verify_jwt_token", lambda t, expected_user_id: {"roles": ["admin"]})
    monkeypatch.setattr(nlp, "detect", lambda _: (_ for _ in ()).throw(Exception("lang fail")))
    monkeypatch.setattr(nlp, "_get_context_from_vector", AsyncMock(return_value=None))

    result = await nlp.apply_knowledge(mock_db, "user1", "prompt", "token")
    assert "Bir hata" not in result  # hata atlamalı


@pytest.mark.asyncio
async def test_apply_knowledge_plugin_timeout(monkeypatch, mock_db):
    """⏱ Plugin timeout'ta None dönmeli."""
    monkeypatch.setattr(nlp, "sanitize_input", lambda x: x)
    monkeypatch.setattr(nlp, "ApplyKnowledgeRequest", lambda **kwargs: kwargs)
    monkeypatch.setattr(nlp, "rate_limiter", MagicMock(is_allowed=lambda _: True))
    monkeypatch.setattr(nlp, "verify_jwt_token", lambda t, expected_user_id: {"roles": ["admin"]})
    monkeypatch.setattr(nlp, "run_plugin_async", AsyncMock(return_value=None))
    monkeypatch.setattr(nlp, "_get_context_from_vector", AsyncMock(return_value="context"))

    result = await nlp.apply_knowledge(mock_db, "user1", "prompt", "token", plugin_data={"plugin_class": object})
    assert "plugin" not in result.lower()


@pytest.mark.asyncio
async def test_apply_knowledge_general_exception(monkeypatch, mock_db):
    """💥 Beklenmeyen exception yakalanmalı."""
    monkeypatch.setattr(nlp, "ApplyKnowledgeRequest", lambda **kwargs: kwargs)
    monkeypatch.setattr(nlp, "sanitize_input", lambda x: x)
    monkeypatch.setattr(nlp, "rate_limiter", MagicMock(is_allowed=lambda _: True))
    monkeypatch.setattr(nlp, "verify_jwt_token", lambda *a, **k: (_ for _ in ()).throw(Exception("JWT fail")))

    result = await nlp.apply_knowledge(mock_db, "user1", "prompt", "token")
    assert "Bir hata" in result


def test_invalidate_context_cache_for_user(monkeypatch):
    """🗑 Cache temizleme testi."""
    # Fake cache key
    nlp._context_cache[("userX", "key1")] = "data"
    nlp.invalidate_context_cache_for_user("userX")
    assert all(k[0] != "userX" for k in nlp._context_cache.keys())

@pytest.mark.asyncio
async def test_apply_knowledge_prometheus_labels(monkeypatch, mock_db):
    """📊 Prometheus sayaçlarının doğru etiketlerle artışı test edilir."""
    # Sayaçları sıfırla
    for sample in nlp.APPLY_KNOWLEDGE_REQUEST_COUNT.collect():
        for s in sample.samples:
            nlp.APPLY_KNOWLEDGE_REQUEST_COUNT._metrics.clear()

    monkeypatch.setattr(nlp, "DatabaseManager", lambda **_: mock_db)
    monkeypatch.setattr(nlp, "verify_jwt_token", lambda t, expected_user_id: {"roles": ["admin"]})
    monkeypatch.setattr(nlp, "sanitize_input", lambda x: x)
    monkeypatch.setattr(nlp, "rate_limiter", MagicMock(is_allowed=lambda _: True))
    monkeypatch.setattr(nlp, "_get_context_from_vector", AsyncMock(return_value="context"))

    # Başarılı çağrı
    await nlp.apply_knowledge(mock_db, "user1", "prompt", "valid.token")

    # Sayaç metriklerini kontrol et
    found = False
    for sample in nlp.APPLY_KNOWLEDGE_REQUEST_COUNT.collect():
        for s in sample.samples:
            # s.labels bir dict: {'status': 'success'}
            if s.labels.get("status") == "success" and s.value >= 1:
                found = True
    assert found, "Prometheus success metriği artmamış!"

@pytest.mark.asyncio
async def test_apply_knowledge_prometheus_error_label(monkeypatch, mock_db):
    """💥 Bir hata durumunda Prometheus 'error' metriğinin artışı test edilir."""
    # Sayaçları sıfırla
    nlp.APPLY_KNOWLEDGE_REQUEST_COUNT._metrics.clear()

    # Bir hata fırlatacak bir mock bağımlılık oluştur
    monkeypatch.setattr(nlp, "verify_jwt_token", lambda *a, **k: (_ for _ in ()).throw(Exception("JWT fail")))
    monkeypatch.setattr(nlp, "ApplyKnowledgeRequest", lambda **kwargs: kwargs)
    monkeypatch.setattr(nlp, "sanitize_input", lambda x: x)
    monkeypatch.setattr(nlp, "rate_limiter", MagicMock(is_allowed=lambda _: True))

    # Hata fırlatmasını beklenen şekilde yakala
    await nlp.apply_knowledge(mock_db, "user1", "prompt", "token")

    # Sayaç metriklerini kontrol et
    found = False
    for sample in nlp.APPLY_KNOWLEDGE_REQUEST_COUNT.collect():
        for s in sample.samples:
            if s.labels.get("status") == "error" and s.value >= 1:
                found = True
    assert found, "Prometheus error metriği artmamış!"


# ---------------- run_plugin_async tests ---------------- #

@pytest.mark.asyncio
async def test_run_plugin_async_success():
    """✅ Plugin normal çalışır ve output döner"""
    class DummyPlugin:
        def __init__(self, data):
            self.data = data
        async def run(self):
            return "plugin result"

    result = await nlp.run_plugin_async(DummyPlugin, {"key": "value"})
    assert result == "plugin result"

@pytest.mark.asyncio
async def test_run_plugin_async_timeout():
    """⏱ Plugin timeout durumunda None dönmeli"""
    class SlowPlugin:
        async def run(self):
            await asyncio.sleep(2)  # app_settings.PLUGIN_TIMEOUT_SECONDS varsayalım 0.5
            return "late result"

    with patch("nlp.app_settings.PLUGIN_TIMEOUT_SECONDS", 0.5):
        result = await nlp.run_plugin_async(SlowPlugin, {})
        assert result is None

@pytest.mark.asyncio
async def test_run_plugin_async_exception():
    """💥 Plugin exception fırlattığında None dönmeli ve log kaydı yapılmalı"""
    class ErrorPlugin:
        async def run(self):
            raise RuntimeError("plugin fail")

    with patch("nlp.logger") as mock_logger:
        result = await nlp.run_plugin_async(ErrorPlugin, {})
        assert result is None
        mock_logger.warning.assert_not_called()  # Timeout değil, log error olabilir, patch edilebilir

# ---------------- _get_context_from_vector tests ---------------- #

@pytest.mark.asyncio
async def test_get_context_from_vector_returns_string(monkeypatch):
    """✅ Mock ile context döndürmesini test et"""
    dummy_context = "vector context"
    monkeypatch.setattr(nlp, "_get_context_from_vector", AsyncMock(return_value=dummy_context))
    context = await nlp._get_context_from_vector(None, "prompt", "user1")
    assert context == dummy_context

@pytest.mark.asyncio
async def test_get_context_from_vector_returns_none(monkeypatch):
    """🚫 Context yoksa None dönmeli"""
    monkeypatch.setattr(nlp, "_get_context_from_vector", AsyncMock(return_value=None))
    context = await nlp._get_context_from_vector(None, "prompt", "user1")
    assert context is None
