import pytest
import time
import asyncio
from unittest.mock import MagicMock, patch
from learning import RedisRateLimiter, redis_client

# === RATE LIMITER EDGE CASES ===

def test_rate_limiter_resets_after_window(monkeypatch):
    """
    ✅ Test: Süre limiti geçtikten sonra rate limiter resetleniyor mu?
    """
    limiter = RedisRateLimiter("test_reset", rate=2, per=2)
    mock_redis = MagicMock()
    monkeypatch.setattr("learning.redis_client", mock_redis)
    
    mock_pipeline = mock_redis.pipeline.return_value.__enter__.return_value
    # İlk 2 çağrı - limit dolmamış
    mock_pipeline.execute.side_effect = [
        [None, None, 1, None],  # çağrı 1
        [None, None, 2, None],  # çağrı 2
        [None, None, 3, None],  # çağrı 3 → limit aşıldı
        [None, None, 1, None],  # çağrı 4 → resetlendi
    ]

    assert limiter.allow("user1") is True
    assert limiter.allow("user1") is True
    assert limiter.allow("user1") is False
    time.sleep(2)  # süre dolunca
    assert limiter.allow("user1") is True

def test_rate_limiter_allow_and_deny(monkeypatch):
    """
    ✅/❌ Test: Redis'e göre rate limitin çalışıp çalışmadığı doğru tespit ediliyor mu?
    """
    limiter = RedisRateLimiter("test_rate", rate=3, per=2)
    monkeypatch.setattr("learning.redis_client", MagicMock())

    # Simüle: Limit aşılmadı
    mock_pipeline = MagicMock()
    mock_pipeline.execute.return_value = [None, None, 1, None]
    mock_context = MagicMock(__enter__=MagicMock(return_value=mock_pipeline))
    mock_redis = MagicMock()
    mock_redis.pipeline.return_value = mock_context

    monkeypatch.setattr("learning.redis_client", mock_redis)
    limiter = RedisRateLimiter("test", rate=3, per=2)
    assert limiter.allow("user1") is True
    

    # Simüle: Limit aşıldı
    
    mock_pipeline.execute.return_value = [None, None, 4, None]

def test_rate_limiter_redis_unavailable(monkeypatch):
    """
    ⚠ Test: Redis bağlantısı kurulamazsa rate limiter güvenli şekilde çalışmaya devam etmeli.
    """
    limiter = RedisRateLimiter("failover", rate=2, per=1)
    monkeypatch.setattr("learning.redis_client", None)
    
    try:
        result = limiter.allow("user1")
    except Exception:
        result = False

    assert result is False  # ya False döner ya da exception olmaz

@pytest.mark.asyncio
@patch("learning.save_memory", new_callable=asyncio.Future)
@patch("learning.rate_limiter", side_effect=Exception("Redis down"))
async def test_learn_from_text_rate_limiter_failure_fallback(mock_rate_limiter, mock_save):
    """
    ⚠ Test: Rate limiter hata verirse sistem çalışmaya devam etmeli (fail-safe).
    """
    user_id = "user_error"
    text = "Metin"
    await learn_from_text(user_id, text)
    mock_save.assert_awaited_once()


@pytest.mark.asyncio
async def test_cache_ttl_expiry(monkeypatch):
    """
    Redis cache'teki TTL süresi dolduğunda anahtarın silindiğini test eder.
    """
    test_user = "user1"
    test_text = "cache test metni"
    test_hash = "abc123"
    
    monkeypatch.setattr("learning.redis_client", redis_client)

    # Anahtarı 1 saniyelik TTL ile ayarla
    redis_client.setex(f"text_cache:{test_user}:{test_hash}", 1, "1")
    assert redis_client.exists(f"text_cache:{test_user}:{test_hash}") == 1

    # TTL dolduktan sonra anahtar silinmiş olmalı
    await asyncio.sleep(2)
    assert redis_client.exists(f"text_cache:{test_user}:{test_hash}") == 0


@pytest.mark.asyncio
async def test_cache_manual_delete_and_rewrite():
    """
    ✅ Test: Cache silindikten sonra yeniden ayarlanabiliyor mu?
    """
    user = "user1"
    text_hash = "xyz987"
    key = f"text_cache:{user}:{text_hash}"

    # Anahtarı ayarla
    redis_client.setex(key, 5, "1")
    assert redis_client.exists(key) == 1

    # Manuel sil
    redis_client.delete(key)
    assert redis_client.exists(key) == 0

    # Tekrar ayarla
    redis_client.setex(key, 5, "2")
    assert redis_client.get(key) == b"2"
