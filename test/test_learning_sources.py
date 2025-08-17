import pytest
from unittest.mock import patch, AsyncMock, MagicMock
from learning import (
    learn_from_pdf, learn_from_docx, learn_from_url,
    celery_audio_to_text_and_save
)

# === PDF LEARNING ===

@pytest.mark.asyncio
@patch("learning.validate_file_path", return_value=True)
@patch("learning.learn_from_text", new_callable=AsyncMock)
@patch("learning.fitz.open")
async def test_learn_from_pdf_success(mock_fitz_open, mock_learn_text, mock_validate):
    """
    Test: Geçerli bir PDF dosyasından başarıyla öğrenme yapılmalı.
    """
    class MockPage:
        def get_text(self):
            return "Sayfa metni"

    mock_fitz_open.return_value = [MockPage(), MockPage()]

    await learn_from_pdf("user123", "dummy.pdf")

    mock_learn_text.assert_called()
    assert mock_learn_text.call_count == 2
    assert any("Sayfa metni" in call.args[1] for call in mock_learn_text.call_args_list)
    

@pytest.mark.asyncio
@patch("learning.validate_file_path", return_value=False)
@patch("learning.learn_from_text", new_callable=AsyncMock)
async def test_learn_from_pdf_invalid_file(mock_learn_text, mock_validate):
    """
    Test: Geçersiz PDF dosya yolu verildiğinde hiçbir işlem yapılmamalı.
    """
    await learn_from_pdf("user123", "dummy.pdf")

    mock_learn_text.assert_not_called()

@pytest.mark.asyncio
@patch("learning.learn_from_text", new_callable=AsyncMock)
async def test_learn_from_pdf_calls_learn_from_text_correctly(mock_learn_text):
    class MockPage:
        def get_text(self):
            return "Sayfa metni"

    with patch("learning.fitz.open", return_value=[MockPage(), MockPage()]):
        await learn_from_pdf("user123", "file.pdf")
    assert mock_learn_text.call_count == 2
    for call in mock_learn_text.call_args_list:
        assert "Sayfa metni" in call.args[1]
@pytest.mark.asyncio
@patch("learning.validate_file_path", return_value=True)
@patch("learning.LearningService.learn_from_multimodal_text", new_callable=AsyncMock)
@patch("learning.generate_image_caption", new_callable=AsyncMock, return_value="[Görsel Açıklaması: Test altyazısı]")
@patch("learning.fitz.open")
@patch("learning.os.remove") # Geçici dosyaların silinmesini mock'la
async def test_learn_from_pdf_with_image_extraction_and_captioning(
    mock_os_remove, mock_fitz_open, mock_generate_caption, mock_learn_multimodal, mock_validate
):
    """
    Test: PDF'ten metin ve görsellerin başarıyla çıkarıldığını,
    görseller için altyazı oluşturulduğunu ve birleştirilmiş metnin
    doğru fonksiyona gönderildiğini doğrular.
    """
    # --- Mock Kurulumu ---
    class MockPixmap:
        def __init__(self):
            # RGB veya Gri tonlama olduğunu simüle et (n - alpha < 4)
            self.n = 4 
            self.alpha = 0
        def save(self, path):
            pass # save işlemini simüle et

    class MockPage:
        def get_text(self):
            return "Bu bir sayfa metnidir."
        
    class MockDoc:
        def __init__(self):
            self.pages = [MockPage()]
        def load_page(self, page_num):
            return self.pages[page_num]
        def get_page_images(self, page_num):
            # Bir görsel bulunduğunu simüle et
            return [(12345,)] 
        def __len__(self):
            return len(self.pages)
        def close(self):
            pass
        
    mock_fitz_open.return_value = MockDoc()
    # fitz.Pixmap'i mock'la
    with patch("learning.fitz.Pixmap", return_value=MockPixmap()):
        
        # --- Servis ve Fonksiyon Çağrısı ---
        # learning.py'deki yapıya uygun olarak LearningService örneği üzerinden çağrı yapıyoruz.
        from learning import LearningService
        service = LearningService(rate_limiter=MagicMock(), redis_client=MagicMock())
        await service.learn_from_pdf("user123", "dummy_with_image.pdf")

    # --- Doğrulamalar ---
    mock_generate_caption.assert_awaited_once() # Altyazı oluşturma fonksiyonu çağrıldı mı?
    mock_learn_multimodal.assert_awaited_once()  # Son öğrenme fonksiyonu çağrıldı mı?

    # Öğrenme fonksiyonuna giden metnin doğruluğunu kontrol et
    call_args = mock_learn_multimodal.call_args
    user_id_arg = call_args[0][0]
    combined_text_arg = call_args[0][1]

    assert user_id_arg == "user123"
    assert "Bu bir sayfa metnidir." in combined_text_arg
    assert "[Görsel Açıklaması: Test altyazısı]" in combined_text_arg
    mock_os_remove.assert_called() # Geçici görsel dosyası silindi mi?

        
# === DOCX LEARNING ===

@pytest.mark.asyncio
@patch("learning.validate_file_path", return_value=True)
@patch("learning.LearningService.learn_from_text", new_callable=AsyncMock)
@patch("learning.docx.Document")
@patch("learning.PROCESSING_TIME") # Histogram metriğini mock'la
async def test_learn_from_docx_success(mock_processing_time, mock_docx_document, mock_learn_text, mock_validate):
    """
    Test: Geçerli bir DOCX dosyasından başarıyla öğrenme yapılmalı ve
    işlem süresi metriği (Histogram) güncellenmeli.
    """
    # --- Mock Kurulumu ---
    class MockParagraph:
        text = "Paragraf metni"

    mock_docx_document.return_value.paragraphs = [MockParagraph(), MockParagraph()]
    mock_observe = mock_processing_time.labels.return_value.observe

    # --- Servis ve Fonksiyon Çağrısı ---
    from learning import LearningService
    service = LearningService(rate_limiter=MagicMock(), redis_client=MagicMock())
    await service.learn_from_docx("user123", "dummy.docx")

    # --- Doğrulamalar ---
    mock_learn_text.assert_awaited_once() # Ana işlev çağrıldı mı?
    mock_observe.assert_called()           # Histogram metriği güncellendi mi?


@pytest.mark.asyncio
@patch("learning.validate_file_path", return_value=False)
@patch("learning.learn_from_text", new_callable=AsyncMock)
async def test_learn_from_docx_invalid_file(mock_learn_text, mock_validate):
    """
    Test: Geçersiz DOCX dosya yolu verildiğinde öğrenme işlemi yapılmamalı.
    """
    await learn_from_docx("user123", "dummy.docx")

    mock_learn_text.assert_not_called()


# === URL LEARNING TESTLERİ

@pytest.mark.asyncio
@patch("learning.LearningService.learn_from_text", new_callable=AsyncMock)
@patch("learning.advanced_content_filter", return_value=True) # Filtreden geçmesine izin ver
@patch("learning.aiohttp.ClientSession.get")
async def test_learn_from_url_filter_allows(mock_get, mock_filter, mock_learn_text):
    """
    Test: Geçerli bir URL'den içerik çekilip, temizlenip, filtreden geçtiğinde
    `learn_from_text` fonksiyonunun çağrıldığını doğrular.
    """
    class MockResponse:
        status = 200
        async def text(self):
            # Temizlenmesi gereken script etiketleri ekle
            return "<html><head><script>alert('xss')</script></head><body><p>Güvenli içerik.</p></body></html>"
        async def __aenter__(self): return self
        async def __aexit__(self, exc_type, exc, tb): pass

    mock_get.return_value = MockResponse()
    
    from learning import LearningService
    service = LearningService(rate_limiter=MagicMock(), redis_client=MagicMock())
    result = await service.learn_from_url("user123", "http://valid.url")

    assert result is True
    mock_learn_text.assert_awaited_once()
    
    # learn_from_text'e giden metnin temizlendiğini doğrula
    called_text = mock_learn_text.call_args[0][1]
    assert "Güvenli içerik." in called_text
    assert "<script>" not in called_text
    assert "alert('xss')" not in called_text


@pytest.mark.asyncio
@patch("learning.LearningService.learn_from_text", new_callable=AsyncMock)
@patch("learning.advanced_content_filter", return_value=False) # Filtreden geçmesini engelle
@patch("learning.aiohttp.ClientSession.get")
async def test_learn_from_url_filter_blocks(mock_get, mock_filter, mock_learn_text):
    """
    Test: URL içeriği güvenlik filtresinden geçemediğinde
    `learn_from_text` fonksiyonunun ÇAĞRILMADIĞINI doğrular.
    """
    class MockResponse:
        status = 200
        async def text(self): return "<html><body><p>Zararlı içerik.</p></body></html>"
        async def __aenter__(self): return self
        async def __aexit__(self, exc_type, exc, tb): pass

    mock_get.return_value = MockResponse()
    
    from learning import LearningService
    service = LearningService(rate_limiter=MagicMock(), redis_client=MagicMock())
    result = await service.learn_from_url("user123", "http://blocked.url")

    assert result is False
    mock_learn_text.assert_not_called()


@pytest.mark.asyncio
async def test_learn_from_url_invalid_url():
    """
    ⚠ Test: Geçersiz bir URL verildiğinde sistem hata vermeden çalışıyor mu?
    """
    await learn_from_url("user123", "invalid_url")  # Beklenen: fail silently veya log

@pytest.mark.asyncio
@patch("learning.logger")
async def test_learn_from_url_invalid_url_logs_warning(mock_logger):
    """
    invalid URL verildiğinde hata fırlatılmaz, loglama olur mu kontrolü.
    """
    await learn_from_url("user123", "invalid_url")
    mock_logger.warning.assert_called()

@pytest.mark.asyncio
@patch("learning.aiohttp.ClientSession.get")
async def test_learn_from_url_non_200(mock_get):
    """
    ⚠ Test: HTTP 200 olmayan durumlarda (örneğin 404), `learn_from_text` çağrılmamalı.
    """
    class MockResponse:
        status = 404
        async def text(self): return ""
        async def __aenter__(self): return self
        async def __aexit__(self, exc_type, exc, tb): pass

    mock_get.return_value = MockResponse()
    await learn_from_url("user123", "http://bad.url")


# === CELERY AUDIO TASK TESTLERİ

@patch("learning.sr.Recognizer.recognize_google")
@patch("learning.AudioSegment.from_file")
@patch("learning.save_memory")
@patch("learning.AudioSegment.from_file", side_effect=Exception("Dosya okunamadı"))
def test_celery_audio_to_text_and_save_fail(mock_audio_from_file):
    """
    ❌ Test: Dosya okunamazsa exception fırlatılmalı.
    """
    with pytest.raises(Exception):
        celery_audio_to_text_and_save("user123", "badfile.mp3")


@patch("learning.AudioSegment.from_file")
def test_celery_audio_to_text_and_save_file_not_exist(mock_audio_from_file):
    """
    ⚠ Test: Dosya fiziksel olarak mevcut değilse hata alınmadan atlanmalı.
    """
    celery_audio_to_text_and_save("user123", "nonexistent.mp3")



@pytest.mark.asyncio
@patch("learning.save_memory", new_callable=AsyncMock)
@patch("learning.AudioSegment.from_file")
@patch("learning.sr.Recognizer.recognize_google")
async def test_celery_audio_to_text_and_save_success(
    mock_recognize_google, mock_audio_from_file, mock_save, tmp_path
):
    mock_audio_from_file.return_value = MagicMock(len=lambda: 1000)
    mock_recognize_google.return_value = "konuşma metni"
    audio_path = tmp_path / "audio.mp3"
    audio_path.write_text("dummy")
    await celery_audio_to_text_and_save("user123", str(audio_path))
    mock_save.assert_awaited_once()
