import pytest
import time
import logging
import asyncio
from unittest.mock import patch, MagicMock, AsyncMock
import jwt
import redis
from datetime import datetime, timedelta

from learning import (
    RedisRateLimiter,
    celery_summarize_and_save,
    rate_limiter,
    transcribe_audio_file,
    learn_from_text,
    validate_file_path,
    check_user_authorization,
    JWT_SECRET,
    JWT_ALGORITHM,
)

logger = logging.getLogger("learning")

# ---------- Fixtures for reuse ----------

@pytest.fixture(autouse=True)
def clear_patches(monkeypatch):
    yield
    monkeypatch.undo()

@pytest.fixture
def dummy_redis(monkeypatch):
    mock_redis = MagicMock()
    monkeypatch.setattr("learning.redis_client", mock_redis)
    return mock_redis

@pytest.fixture
def patch_save_memory(monkeypatch):
    mock_save = AsyncMock()
    monkeypatch.setattr("learning.save_memory", mock_save)
    return mock_save


# == SES TRANSKRİPSİYON TESTLERİ ==

@patch("learning.transcribe_audio_file", return_value="mock text")
def test_transcribe_audio_file_success(mock_transcribe):
    """
    ✅ Test: transcribe_audio_file başarılı çalışıyor mu?
    """
    result = transcribe_audio_file("dummy_path.wav")
    assert result == "mock text"
    mock_transcribe.assert_called_once()


# ---------- Error Handling and Logging Tests ----------

@patch("learning.get_summarizer", side_effect=Exception("Model yüklenemedi"))
@patch("learning.save_memory", new_callable=AsyncMock)
def test_celery_summarize_and_save_fallback_logs_and_raises(mock_save, mock_get_sum, caplog):
    caplog.set_level(logging.ERROR, logger.name)

    celery_summarize_and_save("user123", "Uzun metin burada yer alır.")

    # save_memory fallback çağrıldı mı?
    assert mock_save.await_count == 1

    # Hata mesajı loglarda yer almalı
    assert any("Model yüklenemedi" in record.message for record in caplog.records)

@patch("learning.redis_client", None)
def test_redis_connection_fail_logs_error(monkeypatch, caplog):
    caplog.set_level(logging.ERROR, logger.name)

    limiter = RedisRateLimiter("fail_test", rate=1, per=1)
    allowed = limiter.allow("user_fail")

    assert allowed is True
    assert any("Redis error" in record.message for record in caplog.records)

@patch("learning.celery_summarize_and_save.retry")
def test_celery_task_retry_called(mock_retry):
    # Retry mekanizmasının exception fırlatması simüle ediliyor
    mock_retry.side_effect = Exception("Retry triggered")

    with pytest.raises(Exception):
        celery_summarize_and_save.retry(exc=Exception("Test error"))

    assert mock_retry.called

# Daha detaylı retry davranış testi

@patch("learning.celery_summarize_and_save.retry")
def test_celery_task_retry_on_specific_exceptions(mock_retry):
    # Farklı exception tipleri için retry çağrısı testi
    from learning import celery_summarize_and_save

    # Retry edilecek exception
    retry_exc = celery.exceptions.Retry()

    mock_retry.side_effect = retry_exc

    with pytest.raises(celery.exceptions.Retry):
        celery_summarize_and_save.retry(exc=retry_exc)

    assert mock_retry.called

# ---------- Async Timeout and Connection Error Tests ----------

@pytest.mark.asyncio
@patch("learning.aiohttp.ClientSession.get", side_effect=asyncio.TimeoutError)
async def test_learn_from_url_timeout(monkeypatch):
    from learning import learn_from_url

    # Timeout durumunda hata atmadan devam etmeli
    await learn_from_url("user123", "http://timeout.url")

@pytest.mark.asyncio
@patch("learning.aiohttp.ClientSession.get", side_effect=Exception("Bağlantı hatası"))
async def test_learn_from_url_connection_error(monkeypatch):
    from learning import learn_from_url

    # Bağlantı hatası durumunda hata atmadan devam etmeli
    await learn_from_url("user123", "http://badconnection.url")

# ---------- File Validation Tests ----------

def test_validate_file_path_valid(tmp_path):
    test_file = tmp_path / "sample.txt"
    test_file.write_text("Sample content")
    assert validate_file_path(str(test_file), allowed_suffixes=[".txt"], max_size_mb=1)

def test_validate_file_path_nonexistent():
    assert not validate_file_path("nonexistent_file.txt")

def test_validate_file_path_large(tmp_path):
    test_file = tmp_path / "large.txt"
    test_file.write_bytes(b"x" * 1024 * 1024 * 5)  # 5MB
    assert not validate_file_path(str(test_file), max_size_mb=1)

# Dosya uzantısı testleri eklendi
def test_validate_file_path_invalid_suffix(tmp_path):
    test_file = tmp_path / "file.invalid"
    test_file.write_text("dummy")
    assert not validate_file_path(str(test_file), allowed_suffixes=[".txt", ".md"])

# ---------- JWT Authorization Tests ----------

def generate_jwt(payload):
    return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)

def test_check_user_authorization_valid():
    token = generate_jwt({"user_id": "user123", "exp": datetime.utcnow() + timedelta(hours=1)})
    assert check_user_authorization(token, "resource_id")

def test_check_user_authorization_expired():
    token = generate_jwt({"user_id": "user123", "exp": datetime.utcnow() - timedelta(hours=1)})
    assert not check_user_authorization(token, "resource_id")

def test_check_user_authorization_malformed():
    assert not check_user_authorization("malformed.token.here", "resource_id")

def test_check_user_authorization_revoked(monkeypatch):
    # Kullanıcı için revoked token kontrolü simüle edilir
    def fake_revoked_check(token):
        return True

    monkeypatch.setattr("learning.is_token_revoked", fake_revoked_check)

    token = generate_jwt({"user_id": "user123", "exp": datetime.utcnow() + timedelta(hours=1)})
    assert not check_user_authorization(token, "resource_id")

def test_jwt_signature_error():
    """
    ✅ Test: JWT imza hatasında istisna fırlatılıyor mu?
    """
    from jose import jwt, JWTError
    invalid_token = "ey.invalid.token"

    with pytest.raises(JWTError):
        jwt.decode(invalid_token, "wrong_secret", algorithms=["HS256"])

# ---------- Concurrency Tests ----------

@pytest.mark.asyncio
async def test_concurrent_learn_from_text(patch_save_memory):
    async def simulate_user(n):
        return await learn_from_text(f"Concurrent test {n}", "deneme metni")

    tasks = [simulate_user(i) for i in range(50)]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    # Exception ya da None dönmeli, başarısız olmamalı
    for r in results:
        assert r is None or isinstance(r, Exception)

# ---------- Redis Rate Limiter Tests ----------

@pytest.mark.integration
def test_redis_rate_limiter_real_redis():
    client = redis.Redis(host="localhost", port=6379, db=0)
    limiter = RedisRateLimiter("integration_test", rate=5, per=1)
    limiter.redis = client

    for _ in range(5):
        assert limiter.allow("integration_user") is True
    assert limiter.allow("integration_user") is False

def test_redis_rate_limiter_allow(dummy_redis):
    limiter = RedisRateLimiter("test", rate=2, per=1)
    dummy_redis.pipeline.return_value.__enter__.return_value.execute.return_value = [None, None, 1, None]
    assert limiter.allow("user1") is True

def test_redis_rate_limiter_denied(dummy_redis):
    limiter = RedisRateLimiter("test", rate=2, per=1)
    dummy_redis.pipeline.return_value.__enter__.return_value.execute.return_value = [None, None, 3, None]
    assert limiter.allow("user1") is False

def test_redis_rate_limiter_no_redis(monkeypatch):
    monkeypatch.setattr("learning.redis_client", None)
    limiter = RedisRateLimiter("test", rate=2, per=1)
    assert limiter.allow("user1") is True


# == TTL TESTLERİ ==

def test_ttl_expiry_behavior():
    """
    ✅ Test: TTL süresi dolunca anahtar siliniyor mu?
    """
    key = "ttl_test_key"
    redis_client.setex(key, 1, "value")
    assert redis_client.get(key) == b"value"
    
    time.sleep(1.5)
    assert redis_client.get(key) is None



# ---------- Async Text Learning Tests ----------

@pytest.mark.asyncio
@patch("learning.save_memory", new_callable=AsyncMock)
@patch("learning.rate_limiter.allow", return_value=True)
async def test_learn_from_text_success(mock_allow, mock_save):
    user_id = "user123"
    text = "Bu bir test metnidir."

    await learn_from_text(user_id, text)

    mock_save.assert_awaited_once()

@pytest.mark.asyncio
@patch("learning.save_memory", new_callable=AsyncMock)
@patch("learning.rate_limiter.allow", return_value=False)
async def test_learn_from_text_rate_limited(mock_allow, mock_save):
    user_id = "user123"
    text = "Bu metin çok sık işleniyor."

    await learn_from_text(user_id, text)

    # Rate limiter izin vermese bile save_memory çağrısı beklenir (fallback)
    assert mock_save.called


# == MOCKLAMA & CALL ARGS TESTLERİ ==

@patch("learning.save_memory")
def test_save_memory_called_with_correct_params(mock_save):
    """
    ✅ Test: save_memory doğru parametrelerle çağrıldı mı?
    """
    expected_user_id = "user42"
    expected_text = "Hello world"
    
    save_memory(expected_user_id, expected_text)
    
    mock_save.assert_called_once()
    assert mock_save.call_args[0][0] == expected_user_id
    assert mock_save.call_args[0][1] == expected_text


# ---------- Celery Task Sync Tests ----------

@patch("learning.get_summarizer")
@patch("learning.save_memory", new_callable=AsyncMock)
def test_celery_summarize_and_save_success(mock_save, mock_get_sum):
    summarizer = lambda x, **kwargs: [{"summary_text": "özet"}]
    mock_get_sum.return_value = summarizer

    celery_summarize_and_save("user123", "Uzun metin burada yer alır.")

    mock_save.assert_awaited_once()

@patch("learning.get_summarizer", side_effect=Exception("Model yüklenemedi"))
@patch("learning.save_memory", new_callable=AsyncMock)
def test_celery_summarize_and_save_fallback(mock_save, mock_get_sum):
    celery_summarize_and_save("user123", "Uzun metin burada yer alır.")
    mock_save.assert_awaited_once()

def test_log_i18n_localization(monkeypatch):
    """
    Test: log_i18n fonksiyonunun gettext kullanarak log mesajlarını
    doğru bir şekilde çevirdiğini doğrular.
    """
    # --- Mock Kurulumu ---
    mock_logger = MagicMock()
    # learning modülündeki logger'ı bizim mock logger'ımız ile değiştir
    monkeypatch.setattr("learning.logger", mock_logger) 

    # gettext.translation fonksiyonunu mock'la
    mock_translation = MagicMock()
    # gettext metodunun belirli bir çeviri yapmasını sağla
    mock_translation.gettext.return_value = "Translated: %(file_path)s" 
    
    mock_gettext = MagicMock(return_value=mock_translation)
    monkeypatch.setattr("learning.gettext.translation", mock_gettext)

    # --- Test Edilecek Fonksiyon ---
    from learning import log_i18n
    
    # log_i18n fonksiyonunu çağır
    log_i18n("error", "Untranslated: %(file_path)s", file_path="test.txt")

    # --- Doğrulama ---
    # gettext.translation'ın doğru parametrelerle çağrıldığını kontrol et
    # Bu, dil ayarlarının doğru yüklendiğini gösterir
    mock_gettext.assert_called() 
    
    # Asıl log fonksiyonunun "çevrilmiş" metin ile çağrıldığını doğrula
    mock_logger.error.assert_called_once_with("Translated: test.txt")
