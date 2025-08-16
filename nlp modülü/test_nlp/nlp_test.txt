import pytest
import os
import time
import concurrent.futures
from unittest.mock import patch
from nlp import estimate_token_count, safe_truncate_nlp

@pytest.mark.parametrize(
    "text, expected_min_tokens",
    [
        ("Merhaba dünya!", 2),
        ("", 0),
        (" ", 0),
        ("Merhaba dünya! " * 1000, 20_000),
        ("😊👍🏽", 3),
    ]
)
def test_estimate_token_count(text, expected_min_tokens):
    count = estimate_token_count(text)
    assert isinstance(count, int), "Token sayısı int tipinde olmalı."
    assert count >= 0, "Token sayısı negatif olmamalı."
    if text.strip():
        assert count >= expected_min_tokens, f"Token sayısı beklenen minimum {expected_min_tokens}'den az."

@pytest.mark.parametrize(
    "text",
    [
        "",                    # Boş string
        "    ",                # Sadece boşluk
        "\t\n\r",              # Kontrol karakterleri
        "kelime\tkelime\nkelime",  # Tab ve newline içeriyor
        "A" * 10000,           # Çok uzun tek karakter
        "\u200b" * 100,        # Zero-width space karakterleri
        "kelime" * 1000 + "\x00",  # Null karakter içeriyor
    ]
)
def test_estimate_token_count_edge_cases(text):
    count = estimate_token_count(text)
    assert isinstance(count, int)
    assert count >= 0

@pytest.mark.parametrize(
    "text, max_tokens",
    [
        ("kelime kelime\tkelime\nkelime", 3),
        ("", 1),
        ("   ", 1),
        ("\t\n\r", 2),
        ("A" * 10000, 5),
        ("\u200b" * 50, 3),
    ]
)
def test_safe_truncate_nlp_edge_cases_with_special_chars(text, max_tokens):
    truncated = safe_truncate_nlp(text, max_tokens=max_tokens)
    assert isinstance(truncated, str)
    count = estimate_token_count(truncated)
    assert count <= max_tokens


@pytest.mark.parametrize(
    "text, max_tokens, expect_truncated",
    [
        ("Kısa cümle.", 100, False),
        ("Bu çok uzun bir metindir ve kesilecek.", 5, True),
        ("", 10, False),
        ("Token sınırında test", 3, True),
        ("A" * 50, 1, True),
    ]
)
def test_safe_truncate_nlp_behavior(text, max_tokens, expect_truncated):
    truncated = safe_truncate_nlp(text, max_tokens=max_tokens)
    assert isinstance(truncated, str), "Dönüş tipi string olmalı."
    if expect_truncated:
        assert truncated != text, "Metin kesilmeli ama farklı kalmalı."
        assert len(truncated) > 0, "Kesilen metin boş olmamalı."
    else:
        assert truncated == text, "Metin kesilmemeli."

def test_safe_truncate_nlp_edge_cases():
    truncated = safe_truncate_nlp("Deneme", max_tokens=0)
    assert truncated == "", "max_tokens=0 ise boş string dönmeli."
    
    with pytest.raises(ValueError):
        safe_truncate_nlp("Deneme", max_tokens=-1)

def test_estimate_token_count_non_string_input():
    for invalid_input in [None, 123, 12.5, [], {}, ()]:
        with pytest.raises(TypeError):
            estimate_token_count(invalid_input)

def test_estimate_token_count_performance_with_tolerance():
    text = "Merhaba dünya! " * 10000
    start_time = time.time()
    count = estimate_token_count(text)
    duration = time.time() - start_time
    assert isinstance(count, int)
    assert duration < 2.0, f"estimate_token_count çok yavaş çalışıyor: {duration:.2f}s"

    # Performans limiti dışarıdan ayarlanabilir, default 2 saniye
    max_duration = float(os.getenv("PERF_TEST_TIMEOUT", "2.0"))
    print(f"[PERF TEST] estimate_token_count çalışma süresi: {duration:.2f}s (limit: {max_duration}s)")
    assert duration < max_duration, f"estimate_token_count çok yavaş çalışıyor: {duration:.2f}s"

@pytest.mark.parametrize(
    "text, max_tokens, expect_truncated",
    [
        ("Token sınırında", 1, True),
        ("Tek kelime", 2, False),
        ("A", 1, False),
        ("", 1, False),
        ("Test metni", 0, True),
    ]
)
def test_safe_truncate_nlp_additional_edges(text, max_tokens, expect_truncated):
    truncated = safe_truncate_nlp(text, max_tokens=max_tokens)
    assert isinstance(truncated, str)
    if expect_truncated:
        assert truncated != text or truncated == "", "Metin kesilmeli."
    else:
        assert truncated == text, "Metin kesilmemeli."

def test_safe_truncate_nlp_invalid_max_tokens_behavior():
    with pytest.raises(ValueError):
        safe_truncate_nlp("Deneme", max_tokens=-5)

def test_estimate_token_count_invalid_input_type():
    for invalid in [None, 123, 5.6, [], {}, set()]:
        with pytest.raises(TypeError):
            estimate_token_count(invalid)

def test_concurrent_calls():
    texts = ["Bu bir test metnidir."] * 1000
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        futures = [executor.submit(estimate_token_count, t) for t in texts]
        results = [f.result() for f in futures]
    assert all(isinstance(r, int) for r in results), "Tüm sonuçlar int olmalı."

@patch("nlp.tiktoken")
def test_estimate_token_count_with_mocked_tiktoken_varied_behavior(mock_tiktoken):
    class DummyEncoder:
        def encode(self, text):
            return [1] * (len(text.split()) * 2)
    mock_tiktoken.encoding_for_model.return_value = DummyEncoder()

    text = "Bu bir deneme metni"
    count = estimate_token_count(text)
    expected = len(text.split()) * 2
    assert count == expected, f"Mock kullanınca {expected} token sayılmalı, {count} bulundu."

    class ErrorEncoder:
        def encode(self, text):
            raise RuntimeError("Encode hatası")
    mock_tiktoken.encoding_for_model.return_value = ErrorEncoder()

    with pytest.raises(RuntimeError):
        estimate_token_count("Hata testi")

@patch("nlp.tiktoken")
def test_estimate_token_count_mock_various_behaviors(mock_tiktoken):
    class DummyEncoderSuccess:
        def encode(self, text):
            return [1] * (len(text.split()) * 3)  # 3 kat token döndür

    class DummyEncoderEmpty:
        def encode(self, text):
            return []

    class DummyEncoderNone:
        def encode(self, text):
            return None

    class DummyEncoderError:
        def encode(self, text):
            raise RuntimeError("Mock encode error")

    # Başarılı durum
    mock_tiktoken.encoding_for_model.return_value = DummyEncoderSuccess()
    text = "Bu bir test metni"
    count = estimate_token_count(text)
    expected = len(text.split()) * 3
    assert count == expected, f"Başarılı mock ile {expected} token bekleniyor, {count} geldi."

    # Boş liste dönmesi
    mock_tiktoken.encoding_for_model.return_value = DummyEncoderEmpty()
    count = estimate_token_count(text)
    assert count == 0, "Boş encode sonucu token sayısı 0 olmalı."

    # None dönmesi (hatalı kullanım)
    mock_tiktoken.encoding_for_model.return_value = DummyEncoderNone()
    with pytest.raises(TypeError):
        estimate_token_count(text)

    # Hata fırlatması
    mock_tiktoken.encoding_for_model.return_value = DummyEncoderError()
    with pytest.raises(RuntimeError):
        estimate_token_count(text)
# ... diğer test fonksiyonları ...

@pytest.mark.parametrize(
    "text",
    [
        "İstanbul'da güzel bir gün ☀️🌳",
        "中文输入测试",
        "👍🏽🔥💯",
        "Emoji test 😊🚀✨",
        "𝓣𝓮𝔁𝓽 𝔀𝓲𝓽𝓱 𝓼𝓽𝔂𝓵𝓮𝓭 𝓬𝓱𝓪𝓻𝓼",
        "अच्छा दिन है",
        "مرحبا بالعالم",
        "Тестирование текста",
    ]
)
def test_estimate_token_count_unicode_and_special_chars(text):
    count = estimate_token_count(text)
    assert isinstance(count, int), "Token sayısı int olmalı."
    assert count > 0, "Token sayısı pozitif olmalı."

@pytest.mark.parametrize(
    "text, max_tokens",
    [
        ("İstanbul'da güzel bir gün ☀️🌳", 5),
        ("中文输入测试", 3),
        ("👍🏽🔥💯", 2),
        ("Emoji test 😊🚀✨", 4),
        ("𝓣𝓮𝔁𝓽 𝔀𝓲𝓽𝓱 𝓼𝓽𝔂𝓵𝓮𝓭 𝓬𝓱𝓪𝓻𝓼", 6),
        ("अच्छा दिन है", 3),
        ("مرحبا بالعالم", 4),
        ("Тестирование текста", 5),
    ]
)
def test_safe_truncate_nlp_unicode(text, max_tokens):
    truncated = safe_truncate_nlp(text, max_tokens=max_tokens)
    assert isinstance(truncated, str), "Truncate sonucu string olmalı."
    count = estimate_token_count(truncated)
    assert count <= max_tokens, f"Truncate sonrası token sayısı max_tokens'dan fazla: {count} > {max_tokens}"



@pytest.mark.parametrize(
    "text",
    [
        "İstanbul'da güzel bir gün ☀️🌳",
        "中文输入测试",
        "👍🏽🔥💯",
        "Emoji test 😊🚀✨",
        "𝓣𝓮𝔁𝓽 𝔀𝓲𝓽𝓱 𝓼𝓽𝔂𝓵𝓮𝓭 𝓬𝓱𝓪𝓻𝓼",
    ]
)

def test_safe_truncate_nlp_large_text_truncation():
    large_text = "kelime " * 100_000
    max_tokens = 50

    truncated = safe_truncate_nlp(large_text, max_tokens=max_tokens)
    count = estimate_token_count(truncated)
    assert count <= max_tokens, "Truncate edilen metin max_tokens sınırını aşmamalı."
    assert isinstance(truncated, str), "Truncate sonucu string olmalı."
    assert len(truncated) < len(large_text), "Truncate edilen metin orijinalden kısa olmalı."

def test_safe_truncate_nlp_large_input():
    large_text = "kelime " * 100000
    truncated = safe_truncate_nlp(large_text, max_tokens=500)
    assert isinstance(truncated, str)
    assert len(truncated) > 0
    token_count = estimate_token_count(truncated)
    assert token_count <= 500, f"Truncated token sayısı max_tokens'dan fazla: {token_count} > 500"


@pytest.mark.asyncio
def test_estimate_token_count_sync_in_async_context():
    text = "Merhaba sync dünya!"
    count = estimate_token_count(text)
    assert isinstance(count, int)
    assert count > 0

@pytest.mark.asyncio
def test_safe_truncate_nlp_sync_in_async_context():
    text = "Bu uzun bir metindir ve kesilecek."
    truncated = safe_truncate_nlp(text, max_tokens=5)
    assert isinstance(truncated, str)
    # Token sayısını kontrol et
    from nlp import estimate_token_count
    count = await estimate_token_count(truncated)
    assert count <= 5
