import pytest
import logging
from unittest.mock import patch, MagicMock, AsyncMock
from learning import celery_audio_to_text_and_save, rate_limiter, transcribe_audio_file

logger = logging.getLogger("learning")


# ===========================
# CELERY AUDIO TASK TESTLERİ
# ===========================

@pytest.mark.asyncio
@patch("learning.sr.Recognizer.recognize_google", new_callable=AsyncMock)
@patch("learning.AudioSegment.from_file")
@patch("learning.save_memory", new_callable=AsyncMock)
async def test_celery_audio_to_text_and_save_success(mock_save, mock_audio_from_file, mock_recognize_google, tmp_path):
    """
    ✅ Ses dosyasından metin başarıyla çıkarılır ve kaydedilir.
    """
    mock_audio = MagicMock()
    mock_audio.__len__.return_value = 1000
    mock_audio_from_file.return_value = mock_audio
    mock_recognize_google.return_value = "konuşma metni"

    audio_path = tmp_path / "audio.mp3"
    audio_path.write_text("dummy")

    await celery_audio_to_text_and_save("user123", str(audio_path))

    mock_save.assert_awaited_once_with("konuşma metni")


@pytest.mark.asyncio
@patch("learning.AudioSegment.from_file", side_effect=Exception("Dosya okunamadı"))
async def test_celery_audio_to_text_and_save_fail_exception(mock_audio_from_file):
    """
    ❌ Dosya okunamazsa exception fırlatılmalı ve loglanmalı.
    """
    with pytest.raises(Exception, match="Dosya okunamadı"):
        await celery_audio_to_text_and_save("user123", "badfile.mp3")


@pytest.mark.asyncio
@patch("learning.AudioSegment.from_file", side_effect=FileNotFoundError())
async def test_celery_audio_to_text_and_save_file_not_exist(mock_audio_from_file):
    """
    ⚠ Dosya yoksa hata vermeden atlanmalı ve log info mesajı olmalı.
    """
    with patch.object(logger, 'info') as mock_log_info:
        await celery_audio_to_text_and_save("user123", "nonexistent.mp3")
        mock_log_info.assert_called_with("Ses dosyası bulunamadı: nonexistent.mp3 - işlem atlandı.")


@pytest.mark.asyncio
@patch("learning.sr.Recognizer.recognize_google", side_effect=TimeoutError("Recognition timeout"))
@patch("learning.AudioSegment.from_file")
async def test_celery_audio_to_text_and_save_recognition_timeout(mock_audio_from_file, mock_recognize_google):
    """
    ⚠ Google recognize timeout durumunda exception fırlatılır ve log error kaydı yapılır.
    """
    mock_audio = MagicMock()
    mock_audio.__len__.return_value = 1000
    mock_audio_from_file.return_value = mock_audio

    with patch.object(logger, "error") as mock_log_error:
        with pytest.raises(TimeoutError, match="Recognition timeout"):
            await celery_audio_to_text_and_save("user123", "audio.mp3")

        mock_log_error.assert_called()


# ===========================
# DIŞ SERVIS HATALARI TESTLERİ
# ===========================

@patch("learning.utils.redis.get", side_effect=ConnectionError("Redis down"))
def test_redis_unavailable_message_logs_and_raises(mock_redis):
    """
    ❌ Redis erişilemezse hata fırlatılır ve log error kaydı yapılır.
    """
    with patch.object(logger, "error") as mock_log_error:
        with pytest.raises(ConnectionError, match="Redis down"):
            rate_limiter("test", limit=1, seconds=1)

        mock_log_error.assert_called_with("Redis error: Redis down")


@pytest.mark.asyncio
@patch("learning.audio.app.send_task", side_effect=ConnectionError("Celery queue unreachable"))
async def test_celery_unavailable_raises_and_logs(mock_send):
    """
    ❌ Celery kuyruğu erişilemezse exception fırlatılır ve error log yazılır.
    """
    with patch.object(logger, "error") as mock_log_error:
        with pytest.raises(ConnectionError, match="Celery queue unreachable"):
            await transcribe_audio_file("fake/path/audio.mp3")

        mock_log_error.assert_called_with("Celery queue unreachable: Celery queue unreachable")


@pytest.mark.asyncio
@patch("learning.audio.app.send_task", side_effect=Exception("Celery genel hata"))
async def test_celery_unexpected_error_logs(mock_send):
    """
    ❌ Celery'den beklenmeyen hata gelirse exception fırlatılır ve log tutulur.
    """
    with patch.object(logger, "error") as mock_log_error:
        with pytest.raises(Exception, match="Celery genel hata"):
            await transcribe_audio_file("fake/path/audio.mp3")

        mock_log_error.assert_called()


# ===========================
# RATE LIMITER DIŞ SERVIS HATA TESTLERİ
# ===========================

@patch("learning.redis_client.pipeline", side_effect=Exception("Pipeline error"))
def test_rate_limiter_pipeline_exception_logs(monkeypatch):
    """
    ⚠ Redis pipeline hatası durumunda rate limiter True dönmeli ve hata loglanmalı.
    """
    from learning import RedisRateLimiter

    limiter = RedisRateLimiter("test", rate=1, per=1)
    with patch.object(logger, "error") as mock_log_error:
        allowed = limiter.allow("userX")
        assert allowed is True
        mock_log_error.assert_called()


# ===========================
# Yardımcı fonksiyonlar ve fixture'lar geliştirilebilir.
# ===========================

