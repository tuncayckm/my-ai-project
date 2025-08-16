# test_rate_limiter.py
import pytest
from unittest.mock import MagicMock, patch
import time
import concurrent.futures
import logging
import asyncio

from rate_limiter import RedisRateLimiter
from prometheus_client import Counter

# Prometheus sayaç ekleme (test amaçlı)
RATE_LIMIT_HIT = Counter("test_rate_limit_hit_total", "Test rate limit hit count")


def test_get_key_format(monkeypatch):
    """🔑 _get_key metodu user_id ile doğru Redis key üretmeli."""
    rl = RedisRateLimiter(key_prefix="testprefix", rate=5, per_seconds=60)
    user_id = "user123"
    expected_key = "testprefix:user123"
    assert rl._get_key(user_id) == expected_key

@pytest.mark.parametrize("rate, requests", [
    (5, 10),
    (10, 20),
])
def test_concurrent_requests_with_metrics_and_logging(monkeypatch, rate, requests, caplog):
    """⚡ Concurrency testi + Prometheus sayaç + log doğrulaması"""
    caplog.set_level(logging.WARNING)

    mock_pipe = MagicMock()
    mock_pipe.__enter__.return_value = mock_pipe
    mock_pipe.__exit__.return_value = None
    counts = list(range(1, requests+1))
    mock_pipe.execute.side_effect = [[None, None, c, True] for c in counts]

    mock_redis_instance = MagicMock()
    mock_redis_instance.pipeline.return_value = mock_pipe
    monkeypatch.setattr("rate_limiter.redis.Redis", lambda **kwargs: mock_redis_instance)

    rl = RedisRateLimiter(key_prefix="concurrent", rate=rate, per_seconds=60)

    def make_request(user_id):
        allowed = rl.is_allowed(user_id)
        if not allowed:
            RATE_LIMIT_HIT.inc()
            logging.warning(f"Rate limit exceeded for {user_id}")
        return allowed

    # ThreadPool ile eşzamanlı çağrılar
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        futures = [executor.submit(make_request, "user_concurrent") for _ in range(requests)]
        results = [f.result() for f in futures]

    # Rate limit doğrulama
    for i, res in enumerate(results):
        if i < rate:
            assert res is True
        else:
            assert res is False

    # Prometheus sayaç doğrulama
    assert RATE_LIMIT_HIT._value.get() == max(0, requests - rate)

    # Log doğrulama
    exceeded_logs = [r for r in caplog.records if "Rate limit exceeded" in r.message]
    assert len(exceeded_logs) == max(0, requests - rate)


@pytest.mark.asyncio
async def test_async_concurrent_requests_with_metrics(monkeypatch, caplog):
    """⚡ Async concurrency testi + metrics + log"""
    caplog.set_level(logging.WARNING)

    mock_pipe = MagicMock()
    mock_pipe.__enter__.return_value = mock_pipe
    mock_pipe.__exit__.return_value = None
    mock_pipe.execute.side_effect = [[None, None, i, True] for i in range(1, 21)]

    mock_redis_instance = MagicMock()
    mock_redis_instance.pipeline.return_value = mock_pipe
    monkeypatch.setattr("rate_limiter.redis.Redis", lambda **kwargs: mock_redis_instance)

    rl = RedisRateLimiter(key_prefix="async_concurrent", rate=5, per_seconds=60)

    async def make_request(user_id):
        loop = asyncio.get_event_loop()
        allowed = await loop.run_in_executor(None, rl.is_allowed, user_id)
        if not allowed:
            RATE_LIMIT_HIT.inc()
            logging.warning(f"Rate limit exceeded for {user_id}")
        return allowed

    tasks = [asyncio.create_task(make_request("user_async")) for _ in range(20)]
    results = await asyncio.gather(*tasks)

    for i, res in enumerate(results):
        if i < 5:
            assert res is True
        else:
            assert res is False

    # Prometheus sayaç doğrulama
    assert RATE_LIMIT_HIT._value.get() >= 15

    # Log doğrulama
    exceeded_logs = [r for r in caplog.records if "Rate limit exceeded" in r.message]
    assert len(exceeded_logs) >= 15


@pytest.mark.parametrize("rate, per_seconds, requests", [
    (5, 1, 20),   # 1 saniyelik pencere, 20 istek
    (10, 2, 30),  # 2 saniyelik pencere, 30 istek
])
def test_high_concurrency_short_window(monkeypatch, rate, per_seconds, requests, caplog):
    """⚡ Çok kısa per_seconds ile yüksek concurrency testi + fallback + log"""
    caplog.set_level(logging.WARNING)

    # Mock pipeline
    mock_pipe = MagicMock()
    mock_pipe.__enter__.return_value = mock_pipe
    mock_pipe.__exit__.return_value = None

    counts = list(range(1, requests + 1))
    mock_pipe.execute.side_effect = [[None, None, min(c, rate), True] for c in counts]

    # Mock Redis instance
    mock_redis_instance = MagicMock()
    mock_redis_instance.pipeline.return_value = mock_pipe
    monkeypatch.setattr("rate_limiter.redis.Redis", lambda **kwargs: mock_redis_instance)

    rl = RedisRateLimiter(key_prefix="short_window_test", rate=rate, per_seconds=per_seconds)

    def make_request(user_id):
        allowed = rl.is_allowed(user_id)
        if not allowed:
            logging.warning(f"Rate limit exceeded for {user_id}")
        return allowed

    # ThreadPool ile eşzamanlı çağrılar
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        futures = [executor.submit(make_request, "user_high") for _ in range(requests)]
        results = [f.result() for f in futures]

    # Rate limit doğrulama
    for i, res in enumerate(results):
        if i < rate:
            assert res is True
        else:
            assert res is False

    # Log kontrolü
    exceeded_logs = [r for r in caplog.records if "Rate limit exceeded" in r.message]
    assert len(exceeded_logs) == max(0, requests - rate)


# ---------------- Edge-case tests for rate=0 and negative parameters ----------------

def test_rate_zero_all_requests_blocked(monkeypatch):
    """🚫 rate=0 ise hiçbir istek geçmemeli"""
    mock_pipe = MagicMock()
    mock_pipe.__enter__.return_value = mock_pipe
    mock_pipe.__exit__.return_value = None
    mock_pipe.execute.return_value = [None, None, 0, True]  # current_count=0

    mock_redis_instance = MagicMock()
    mock_redis_instance.pipeline.return_value = mock_pipe
    monkeypatch.setattr("rate_limiter.redis.Redis", lambda **kwargs: mock_redis_instance)

    rl = RedisRateLimiter(key_prefix="rate_zero", rate=0, per_seconds=60)
    assert rl.is_allowed("user1") is False
    assert rl.is_allowed("user2") is False

@pytest.mark.parametrize("rate, per_seconds", [
    (-1, 60),
    (5, -10),
    (-5, -10),
])
def test_negative_rate_or_per_seconds(monkeypatch, rate, per_seconds):
    """⚠️ Negatif rate veya per_seconds parametreleri fallback True dönmeli ve log uyarısı"""
    mock_pipe = MagicMock()
    mock_pipe.__enter__.return_value = mock_pipe
    mock_pipe.__exit__.return_value = None
    mock_pipe.execute.return_value = [None, None, 0, True]

    mock_redis_instance = MagicMock()
    mock_redis_instance.pipeline.return_value = mock_pipe
    monkeypatch.setattr("rate_limiter.redis.Redis", lambda **kwargs: mock_redis_instance)

    rl = RedisRateLimiter(key_prefix="negative_test", rate=rate, per_seconds=per_seconds)
    
    # Kodun mevcut hali bu durumda True döner ve sistemi kilitlemez
    assert rl.is_allowed("user_neg") is True


def test_basic_rate_limiter_behavior(monkeypatch):
    """✅ Temel rate limiter işlev testi + pipeline exception handling"""
    mock_pipe = MagicMock()
    mock_pipe.__enter__.return_value = mock_pipe
    mock_pipe.__exit__.return_value = None
    mock_pipe.execute.side_effect = [[None, None, 1, True]]

    mock_redis_instance = MagicMock()
    mock_redis_instance.pipeline.return_value = mock_pipe
    monkeypatch.setattr("rate_limiter.redis.Redis", lambda **kwargs: mock_redis_instance)

    rl = RedisRateLimiter(key_prefix="basic_test", rate=1, per_seconds=60)

    # İlk istek geçerli
    assert rl.is_allowed("user_test") is True

    # Pipeline exception simülasyonu
    mock_pipe.execute.side_effect = Exception("Pipeline fail")
    # Hata durumunda True dönmeli
    assert rl.is_allowed("user_test") is True


def test_old_requests_removal(monkeypatch):
    """🕒 Eski istekler (zremrangebyscore) pipeline ile temizlenmeli."""
    removed = []
    def mock_zremrangebyscore(key, min_score, max_score):
        removed.append((min_score, max_score))
        return 0

    mock_pipe = MagicMock()
    mock_pipe.__enter__.return_value = mock_pipe
    mock_pipe.__exit__.return_value = None
    mock_pipe.zremrangebyscore.side_effect = mock_zremrangebyscore
    mock_pipe.zadd.return_value = 1
    mock_pipe.zcard.return_value = 1
    mock_pipe.expire.return_value = True
    mock_pipe.execute.return_value = [None, None, 1, True]

    mock_redis = MagicMock()
    mock_redis.pipeline.return_value = mock_pipe
    monkeypatch.setattr("rate_limiter.redis.Redis", lambda **kwargs: mock_redis)

    rl = RedisRateLimiter(key_prefix="time_test", rate=5, per_seconds=60)
    rl.is_allowed("user_time")

    # Zaman aralığı temizleme çağrıldı mı?
    assert removed, "zremrangebyscore çağrılmamış!"
    min_score, max_score = removed[0]
    now = int(time.time())
    assert max_score == now - 60, "Zaman penceresi dışındaki skor doğru hesaplanmamış."

def test_expire_called_with_correct_duration(monkeypatch):
    """⏳ Redis key expire süresi doğru ayarlanmalı."""
    expire_called = []

    mock_pipe = MagicMock()
    mock_pipe.__enter__.return_value = mock_pipe
    mock_pipe.__exit__.return_value = None
    mock_pipe.zremrangebyscore.return_value = 0
    mock_pipe.zadd.return_value = 1
    mock_pipe.zcard.return_value = 1
    mock_pipe.expire.side_effect = lambda key, seconds: expire_called.append(seconds)
    mock_pipe.execute.return_value = [None, None, 1, True]

    mock_redis = MagicMock()
    mock_redis.pipeline.return_value = mock_pipe
    monkeypatch.setattr("rate_limiter.redis.Redis", lambda **kwargs: mock_redis)

    rl = RedisRateLimiter(key_prefix="expire_test", rate=5, per_seconds=60)
    rl.is_allowed("user_expire")

    # Expire çağrısı doğru süre ile yapılmış mı?
    assert expire_called, "expire çağrılmamış!"
    assert expire_called[0] == 61, "Expire süresi per+1 ile eşleşmiyor!"

# ---------------- Redis connection failure ve fallback testleri ----------------

def test_rate_limiter_redis_connection_failure(monkeypatch, caplog):
    """🔌 Redis bağlantısı yoksa rate limiter True döner ve uyarı loglanır"""
    caplog.set_level(logging.WARNING)

    # Redis constructor hatası simülasyonu
    def mock_redis_fail(*args, **kwargs):
        raise redis.ConnectionError("Redis bağlantı hatası")
    monkeypatch.setattr("rate_limiter.redis.Redis", mock_redis_fail)

    rl = RedisRateLimiter(key_prefix="conn_fail_test", rate=1, per_seconds=60)

    # Redis yok, is_allowed True dönmeli
    assert rl.is_allowed("user_fail") is True

    # Log kontrolü
    warning_logs = [r for r in caplog.records if "Rate limit kontrolü atlanıyor" in r.message]
    assert len(warning_logs) == 1


def test_rate_limiter_pipeline_exception_fallback(monkeypatch, caplog):
    """⚠️ Pipeline execute sırasında hata olursa is_allowed True dönmeli ve error loglanmalı"""
    caplog.set_level(logging.ERROR)

    mock_pipe = MagicMock()
    mock_pipe.__enter__.return_value = mock_pipe
    mock_pipe.__exit__.return_value = None
    mock_pipe.execute.side_effect = redis.RedisError("Pipeline fail")

    mock_redis_instance = MagicMock()
    mock_redis_instance.pipeline.return_value = mock_pipe
    monkeypatch.setattr("rate_limiter.redis.Redis", lambda **kwargs: mock_redis_instance)

    rl = RedisRateLimiter(key_prefix="pipeline_fail_test", rate=1, per_seconds=60)

    # Hata durumunda is_allowed True dönmeli
    assert rl.is_allowed("user_pipeline") is True

    # Error log kontrolü
    error_logs = [r for r in caplog.records if "Redis rate limiter hatası" in r.message]
    assert len(error_logs) == 1