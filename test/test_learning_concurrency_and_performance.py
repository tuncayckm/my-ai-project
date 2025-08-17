import pytest
import asyncio
from unittest.mock import MagicMock, patch
from learning import chunk_text, RedisRateLimiter, learn_from_text, save_memory


@pytest.mark.asyncio
async def test_concurrent_learn_from_text_with_errors():
    """
    🔁 Aynı anda çoklu isteklerde sistem hata fırlatmadan çalışıyor mu?
    Ayrıca, bazı çağrılarda hata senaryoları simüle edilir.
    """
    user_id = "concurrent_user"
    text = "Concurrent metin"

    async def faulty_learn(text):
        if "fail" in text:
            raise RuntimeError("Simulated failure")
        await asyncio.sleep(0.01)
        return None

    with patch("learning.learn_from_text", side_effect=faulty_learn):
        tasks = [learn_from_text(user_id, text if i % 10 else "fail") for i in range(50)]
        results = await asyncio.gather(*tasks, return_exceptions=True)

    error_count = sum(1 for res in results if isinstance(res, RuntimeError))
    success_count = sum(1 for res in results if res is None)
    assert error_count > 0
    assert success_count > 0
    assert all(isinstance(res, (type(None), RuntimeError)) for res in results)
    for res in results:
        if isinstance(res, RuntimeError):
            assert "Simulated failure" in str(res)


def test_chunk_text_edge_cases():
    """
    ✂️ chunk_text fonksiyonunun sınır durumlarda doğru çalışması.
    """
    assert chunk_text("") == []
    assert len(chunk_text("kelime " * 1000, max_tokens=50)) > 10
    assert any("😊" in chunk for chunk in chunk_text("çalışma 😊 test", max_tokens=10))

def test_chunk_text_token_count_accuracy():
    """
    🎯 chunk_text: Gerçek token sayımına göre doğruluk testi (tiktoken ile).
    """
    try:
        import tiktoken
    except ImportError:
        pytest.skip("tiktoken yüklü değil")

    text = "Bu metin birkaç cümleden oluşur. Yapay zeka testleri önemlidir. Token sayımı yapılacaktır."
    chunks = chunk_text(text, max_tokens=10)

    enc = tiktoken.encoding_for_model("gpt-3.5-turbo")

    for chunk in chunks:
        tokens = enc.encode(chunk)
        assert len(tokens) <= 10

def test_rate_limiter_allow_and_deny(monkeypatch):
    """
    ✅/❌ Redis rate limiter'ın limit ve izin durumlarını doğru tespit etmesi.
    """
    limiter = RedisRateLimiter("test_rate", rate=3, per=2)
    monkeypatch.setattr("learning.redis_client", MagicMock())

    # Limit aşılmadı durumu
    monkeypatch.setattr(
        limiter.redis.pipeline.return_value.__enter__().execute,
        'return_value', [None, None, 1, None]
    )
    assert limiter.allow("user1") is True

    # Limit aşıldı durumu
    monkeypatch.setattr(
        limiter.redis.pipeline.return_value.__enter__().execute,
        'return_value', [None, None, 4, None]
    )
    assert limiter.allow("user1") is False


@pytest.mark.asyncio
async def test_rate_limiter_stateful_behavior(monkeypatch):
    """
    🕒 Redis rate limiter anahtar TTL ve sayaç davranışının simülasyonu.
    """
    class DummyRedis:
        def __init__(self):
            self.store = {}
            self.ttls = {}

        def pipeline(self):
            class Pipe:
                def __init__(self, outer):
                    self.outer = outer
                    self.commands = []

                def incr(self, key):
                    self.commands.append(('incr', key))
                    return self

                def expire(self, key, seconds):
                    self.commands.append(('expire', key, seconds))
                    return self

                def execute(self):
                    results = []
                    for cmd in self.commands:
                        if cmd[0] == 'incr':
                            self.outer.store[cmd[1]] = self.outer.store.get(cmd[1], 0) + 1
                            results.append(self.outer.store[cmd[1]])
                        elif cmd[0] == 'expire':
                            self.outer.ttls[cmd[1]] = cmd[2]
                            results.append(True)
                    return results

                def __enter__(self):
                    return self

                def __exit__(self, exc_type, exc_value, traceback):
                    pass
            return Pipe(self)

    dummy_redis = DummyRedis()
    monkeypatch.setattr("learning.redis_client", dummy_redis)

    limiter = RedisRateLimiter("test_key", rate=3, per=10)

    # İlk 3 istek izinli olmalı
    assert limiter.allow("userA") is True
    assert limiter.allow("userA") is True
    assert limiter.allow("userA") is True

    # 4. istek limit aşıldı
    assert limiter.allow("userA") is False

    # TTL ayarlanmış mı kontrolü
    assert dummy_redis.ttls.get("test_key:userA") == 10

@pytest.mark.asyncio
async def test_rate_limiter_reset_after_ttl(monkeypatch):
    """
    🔄 TTL süresi dolduktan sonra tekrar istek yapılabilmeli.
    """
    import time

    class DummyRedis:
        def __init__(self):
            self.store = {}
            self.timestamps = {}

        def pipeline(self):
            class Pipe:
                def __init__(self, outer):
                    self.outer = outer
                    self.commands = []

                def incr(self, key):
                    self.commands.append(("incr", key))
                    return self

                def expire(self, key, seconds):
                    self.commands.append(("expire", key, seconds))
                    return self

                def execute(self):
                    results = []
                    for cmd in self.commands:
                        if cmd[0] == "incr":
                            now = int(time.time())
                            count = self.outer.store.get(cmd[1], (0, 0))[0]
                            ts = self.outer.store.get(cmd[1], (0, 0))[1]
                            if now - ts >= 2:
                                count = 0
                            count += 1
                            self.outer.store[cmd[1]] = (count, now)
                            results.append(count)
                        elif cmd[0] == "expire":
                            results.append(True)
                    return results

                def __enter__(self):
                    return self

                def __exit__(self, exc_type, exc_value, traceback):
                    pass
            return Pipe(self)

    dummy_redis = DummyRedis()
    monkeypatch.setattr("learning.redis_client", dummy_redis)

    limiter = RedisRateLimiter("reset_test", rate=2, per=2)

    assert limiter.allow("userX") is True
    assert limiter.allow("userX") is True
    assert limiter.allow("userX") is False

    # Bekleme ile TTL geçmeli
    await asyncio.sleep(2.1)
    assert limiter.allow("userX") is True

@pytest.mark.benchmark(group="chunk_text")
def test_chunk_text_performance(benchmark):
    """
    chunk_text performans testi.
    """
    text = "kelime " * 500
    benchmark(lambda: chunk_text(text, max_tokens=50))

@pytest.mark.benchmark(group="chunk_text")
def test_chunk_text_benchmark_threshold(benchmark):
    """
    ⏱️ chunk_text performansı 50ms altında olmalı (örnek eşik).
    """
    text = "kelime " * 500

    result = benchmark(lambda: chunk_text(text, max_tokens=50))
    assert result.stats.mean < 0.05  # 50 ms

@pytest.mark.benchmark(group="learn_from_text")
@pytest.mark.asyncio
async def test_learn_from_text_performance(benchmark):
    """
    learn_from_text performans testi.
    """
    async def run():
        # save_memory async fonksiyonunu mocklayalım performansı gerçek metinden izole etmek için
        with patch("learning.save_memory", new_callable=AsyncMock) as mock_save:
            await learn_from_text("user_perf", "deneme metni " * 100)
            mock_save.assert_awaited()

    await benchmark(run)


@pytest.mark.asyncio
async def test_rate_limiter_performance(monkeypatch, benchmark):
    """
    RedisRateLimiter.allow performans testi 100 çağrı için.
    """
    mock_redis = MagicMock()
    monkeypatch.setattr("learning.redis_client", mock_redis)
    mock_redis.pipeline.return_value.__enter__.return_value.execute.return_value = [None, None, 1, None]

    limiter = RedisRateLimiter("perf_test", rate=10, per=1)

    def call_allow():
        for _ in range(100):
            limiter.allow("user1")

    benchmark(call_allow)
