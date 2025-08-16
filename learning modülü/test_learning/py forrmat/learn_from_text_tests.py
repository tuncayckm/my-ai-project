import asyncio
import logging
import time
import jwt
import pytest
from unittest.mock import AsyncMock, MagicMock, patch, call
from prometheus_client import Counter

# === Test edilen modül ve bileşenler ===
from learning import (
    validate_file_path, check_user_authorization, chunk_text,
    RedisRateLimiter, learn_from_text, celery_summarize_and_save,
    get_tokenizer, get_summarizer, learn_from_pdf, learn_from_docx,
    learn_from_url, celery_audio_to_text_and_save, rate_limiter,
    redis_client, learning_calls_total
)
from core.metrics import LEARN_COUNTER, LEARN_ERROR_COUNTER, metrics

# === Otomatik sayaç sıfırlama ===
@pytest.fixture(autouse=True)
def reset_counters():
    """
    Her testten önce LEARN_COUNTER ve LEARN_ERROR_COUNTER sayaçlarını sıfırlar.
    Bu sayede testler birbirinden bağımsız metrik sonuçlarıyla çalışır.
    """
    LEARN_COUNTER._value.set(0)
    LEARN_ERROR_COUNTER._value.set(0)
    yield


# === Sabit tanımlar (JWT, logger, metrik) ===
JWT_SECRET = "testsecret"
JWT_ALGORITHM = "HS256"

# Örnek bir metrik sayaç tanımı (test içinde kullanılacak)
learning_calls_total = Counter('learning_calls_total', 'Total learning calls')

# Logger yapılandırması
logger = logging.getLogger(__name__)


# === METRIK / MONITORING TESTLERİ ===

@pytest.mark.asyncio
async def test_learn_from_text_metric_on_tokenizer_failure():
    """
    Tokenizer hatasında LEARN_ERROR_COUNTER artışını test eder.
    """
    initial = LEARN_ERROR_COUNTER._value.get()

    with patch("learning.chunk_text", return_value=["chunk"]), \
         patch("learning.get_tokenizer", side_effect=RuntimeError("tokenizer failed")):

        with pytest.raises(RuntimeError):
            await learn_from_text("örnek metin")

    assert LEARN_ERROR_COUNTER._value.get() == initial + 1


@pytest.mark.asyncio
async def test_learn_from_text_metric_on_success():
    """
    Metin başarıyla işlendiğinde LEARN_COUNTER artışını test eder.
    """
    initial = LEARN_COUNTER._value.get()

    with patch("learning.chunk_text", return_value=["chunk"]), \
         patch("learning.get_tokenizer"), \
         patch("learning.get_summarizer"), \
         patch("learning.celery_summarize_and_save.delay"):

        await learn_from_text("örnek metin")

    assert LEARN_COUNTER._value.get() == initial + 1


@pytest.mark.asyncio
async def test_learn_from_text_metric_on_chunking_failure():
    """
    chunk_text başarısız olduğunda LEARN_ERROR_COUNTER artışını test eder.
    """
    initial = LEARN_ERROR_COUNTER._value.get()

    with patch("learning.chunk_text", side_effect=ValueError("chunk failed")):
        with pytest.raises(ValueError, match="chunk failed"):
            await learn_from_text("deneme metin")

    assert LEARN_ERROR_COUNTER._value.get() == initial + 1


def test_learning_metric_increments(monkeypatch):
    """
    Metric sayaçlarının elle artırılması test edilir.
    learning_calls_total metriği testte manuel olarak artırılır.
    """
    initial_count = learning_calls_total._value.get()

    # save_memory fonksiyonu yerine metrik artıran mock fonksiyonu kullan
    async def mock_save(*args, **kwargs):
        learning_calls_total.inc()

    monkeypatch.setattr("learning.save_memory", mock_save)

    import asyncio
    asyncio.run(learning.learn_from_text("user_metric", "metin"))

    assert learning_calls_total._value.get() == initial_count + 1


@pytest.mark.asyncio
@patch("core.metrics.metrics.learn_request_counter.inc")
async def test_learn_metric_increment(mock_inc):
    """
    learn_request_counter metriğinin artıp artmadığını test eder.
    """
    await learn_from_text("Metric test")
    mock_inc.assert_called_once()


# === EDGE CASE TESTLERİ ===

def test_validate_file_path_too_small(tmp_path):
    """
    Çok küçük dosyaların (örneğin 0 byte) geçersiz sayıldığını test eder.
    """
    test_file = tmp_path / "empty.txt"
    test_file.write_bytes(b"")  # 0 byte dosya oluştur
    assert not validate_file_path(str(test_file), min_size_mb=0.1)

@pytest.mark.asyncio
@patch("learning.aiohttp.ClientSession.get")
async def test_learn_from_url_very_long_url(mock_get):
    """
    Aşırı uzun URL'lerin de işlenebilir olduğunu doğrular.
    """
    class MockResponse:
        status = 200
        async def text(self):
            return "<html><body><p>İçerik</p></body></html>"

        async def __aenter__(self): return self
        async def __aexit__(self, exc_type, exc, tb): pass

    mock_get.return_value = MockResponse()
    long_url = "http://example.com/" + ("a" * 5000)
    await learn_from_url("user_longurl", long_url)

@pytest.mark.parametrize("text, expected_chunks", [
    ("", 0),  # Boş metin
    ("A" * 100000, pytest.approx(100000 / 512, rel=0.5)),  # Çok uzun metin
    ("\u202e" * 1024, 1),  # Unicode kontrol karakterleri
])
def test_chunk_text_boundaries(text, expected_chunks):
    """
    chunk_text fonksiyonunun sınır koşullarındaki davranışını test eder.
    """
    chunks = chunk_text(text)
    if isinstance(expected_chunks, (int, float)):
        assert len(chunks) == pytest.approx(expected_chunks, rel=0.5)
    else:
        assert len(chunks) == expected_chunks

def test_chunk_text_edge_cases():
    """
    chunk_text fonksiyonunun boş, kısa, çok uzun ve özel karakter içeren metinlerdeki davranışını test eder.
    """
    # Boş metin
    assert chunk_text("") == []

    # Çok kısa metin
    short_text = "Merhaba."
    chunks = chunk_text(short_text, max_tokens=10)
    assert chunks == [short_text]

    # Çok uzun metin
    long_text = "kelime " * 1000
    chunks = chunk_text(long_text, max_tokens=50)
    assert len(chunks) > 10

    # Özel karakter ve emoji içeren metin
    special_text = "çalışma 😊 test — hızlı & doğru."
    chunks = chunk_text(special_text, max_tokens=10)
    assert any("😊" in chunk for chunk in chunks)

def test_advanced_content_filter_edge_cases():
    """
    Farklı spam içerik varyasyonlarının içerik filtresinden geçip geçmediğini test eder.
    """
    spam_texts = [
        "This is a SpamWord1 attempt",
        "Beware of phishing attacks!",
        "MALWARE detected inside.",
        "Possible Hack attempt",
        "No bad content here."
    ]
    results = [advanced_content_filter(text) for text in spam_texts]
    assert results == [False, False, False, False, True]

def test_rate_limiter_edge_cases(monkeypatch):
    """
    Redis bağlantısının olmaması veya hata vermesi durumunda rate limiter'ın izin vermeye devam etmesini doğrular.
    """
    limiter = RedisRateLimiter("test_rate", rate=3, per=2)

    # Redis client yoksa True dönmeli
    monkeypatch.setattr("learning.redis_client", None)
    assert limiter.allow("userX")

    # Redis pipeline hata verirse yine True dönmeli
    mock_redis = MagicMock()
    mock_redis.pipeline.side_effect = Exception("Redis hata")
    monkeypatch.setattr("learning.redis_client", mock_redis)
    assert limiter.allow("userX")

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

logger = logging.getLogger("learning")


# ==============================
#          PERFORMANS TESTLERİ
# ==============================

@pytest.mark.benchmark(group="chunk_text")
def test_chunk_text_performance(benchmark):
    """
    chunk_text fonksiyonunun performansını test eder.
    
    Bu test, verilen metni tokenize edip parçalara ayırma işleminin
    ne kadar sürede tamamlandığını ölçer.
    """
    text = "kelime " * 500
    benchmark(lambda: chunk_text(text, max_tokens=50))


@pytest.mark.benchmark(group="learn_from_text")
@pytest.mark.asyncio
async def test_learn_from_text_performance(benchmark):
    """
    learn_from_text fonksiyonunun performansını test eder.

    Asenkron çalışan bu fonksiyon, uzun bir metinden bilgi öğrenme sürecini ölçer.
    """
    async def run():
        await learn_from_text("user_perf", "deneme metni " * 100)

    benchmark(run)


@pytest.mark.asyncio
async def test_rate_limiter_performance(monkeypatch, benchmark):
    """
    RedisRateLimiter sınıfının allow metodunun performansını test eder.

    Redis yapısı mock'lanmıştır. 100 istek üzerinde hız sınırlama ölçülür.
    """
    mock_redis = MagicMock()
    monkeypatch.setattr("learning.redis_client", mock_redis)
    mock_redis.pipeline.return_value.__enter__.return_value.execute.return_value = [None, None, 1, None]

    limiter = RedisRateLimiter("perf_test", rate=10, per=1)

    def call_allow():
        for _ in range(100):
            limiter.allow("user1")

    benchmark(call_allow)
