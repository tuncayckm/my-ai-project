import pytest
import os
import time
import json
import re
import logging
from unittest.mock import patch, MagicMock
from pathlib import Path
from pydantic import ValidationError
from config import (
    AppSettings,
    ContextLoggerAdapter,
    load_translation,
    trace_id_var,
    _
)

# === Fixtures ve Yardımcı Fonksiyonlar ===

@pytest.fixture(scope="session")
def valid_api_key():
    """Geçerli bir OpenAI API anahtarı sağlar."""
    return "sk-testapikey12345678901234567890"

@pytest.fixture
def clean_env(monkeypatch):
    """Her test öncesi ortam değişkenlerini temizler."""
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("ENV_PROFILE", raising=False)
    monkeypatch.delenv("PORT", raising=False)
    monkeypatch.delenv("DATABASE_DSN", raising=False)
    monkeypatch.delenv("ALLOWED_ORIGINS", raising=False)
    monkeypatch.delenv("LOG_LEVEL", raising=False)
    yield

@pytest.fixture
def default_settings(valid_api_key, monkeypatch, clean_env):
    """Geçerli bir API anahtarı ile AppSettings nesnesi sağlar."""
    monkeypatch.setenv("OPENAI_API_KEY", valid_api_key)
    return AppSettings()

# === Birim Testleri: Temel Fonksiyonellik ===

def test_settings_loads_successfully(default_settings):
    """
    ✅ Test: AppSettings nesnesi geçerli bir API anahtarı ile sorunsuz yüklenmeli.
    """
    assert isinstance(default_settings, AppSettings)

def test_default_values_are_correct(default_settings):
    """
    ✅ Test: Ortam değişkenleri verilmediğinde varsayılan değerlerin doğru yüklendiğini doğrular.
    """
    assert default_settings.APP_NAME == "Advanced AIService"
    assert default_settings.DEBUG is False
    assert default_settings.PORT == 8000
    assert default_settings.ALLOWED_ORIGINS == [r"https?://localhost:?.*"]

def test_env_vars_override_defaults(monkeypatch, valid_api_key, clean_env):
    """
    ✅ Test: Ortam değişkenlerinin varsayılan değerleri başarıyla geçersiz kıldığını doğrular.
    """
    monkeypatch.setenv("OPENAI_API_KEY", valid_api_key)
    monkeypatch.setenv("PORT", "9001")
    monkeypatch.setenv("DATABASE_DSN", "postgresql://user:pass@db:5432/ai_prod")
    settings = AppSettings()
    assert settings.PORT == 9001
    assert settings.DATABASE_DSN == "postgresql://user:pass@db:5432/ai_prod"

def test_missing_required_field_raises_error(clean_env):
    """
    ❌ Test: Zorunlu olan OPENAI_API_KEY eksikse ValidationError fırlatılmalı.
    """
    with pytest.raises(ValidationError) as exc_info:
        AppSettings()
    assert "openai_api_key" in str(exc_info.value)
    assert "field required" in str(exc_info.value)

# === Yeni Eklenen Testler: .env Dosyası Yüklemesi ===
@patch("config.load_dotenv")
@patch("config.Path.exists", return_value=True)
def test_load_env_file_with_env_mode(mock_exists, mock_load, monkeypatch):
    """
    ✅ Test: ENV_MODE ortam değişkenine göre doğru .env dosyasını yüklediğini doğrular.
    """
    monkeypatch.setenv("ENV_MODE", "test")
    # _load_env_file fonksiyonu çağrıldığında
    from config import _load_env_file
    _load_env_file()
    
    # load_dotenv'in doğru dosya yolu ile çağrıldığını kontrol et
    mock_load.assert_called_once_with(dotenv_path=str(Path(".env.test")))

@patch("config.find_dotenv", return_value=".env")
@patch("config.load_dotenv")
def test_load_env_file_falls_back_to_default(mock_load, mock_find, monkeypatch):
    """
    ✅ Test: ENV_MODE belirtilmediğinde varsayılan .env dosyasını yüklediğini doğrular.
    """
    monkeypatch.delenv("ENV_MODE", raising=False)
    # _load_env_file fonksiyonu çağrıldığında
    from config import _load_env_file
    _load_env_file()
    
    # load_dotenv'in doğru dosya yolu ile çağrıldığını kontrol et
    mock_find.assert_called_once()
    mock_load.assert_called_once_with(dotenv_path=".env")

@patch("config.find_dotenv", return_value="")
@patch("config.app_logger.warning")
def test_load_env_file_logs_warning_if_no_file_found(mock_warning, monkeypatch):
    """
    ✅ Test: Hiçbir .env dosyası bulunamadığında uyarı logu attığını doğrular.
    """
    monkeypatch.delenv("ENV_MODE", raising=False)
    # _load_env_file fonksiyonu çağrıldığında
    from config import _load_env_file
    _load_env_file()
    
    mock_warning.assert_called_once_with("No .env file found")


# === Birim Testleri: Validasyon ve Uç Durumlar ===

def test_allowed_origins_parsing_from_string(monkeypatch, valid_api_key, clean_env):
    """
    ✅ Test: ALLOWED_ORIGINS virgülle ayrılmış bir string olarak verildiğinde listeye dönüştürülmeli.
    """
    monkeypatch.setenv("OPENAI_API_KEY", valid_api_key)
    monkeypatch.setenv("ALLOWED_ORIGINS", "http://localhost,https://example.com")
    settings = AppSettings()
    assert settings.ALLOWED_ORIGINS == ["http://localhost", "https://example.com"]

def test_allowed_origins_parsing_from_json(monkeypatch, valid_api_key, clean_env):
    """
    ✅ Test: ALLOWED_ORIGINS JSON formatında bir liste olarak verildiğinde doğru şekilde işlenmeli.
    """
    monkeypatch.setenv("OPENAI_API_KEY", valid_api_key)
    monkeypatch.setenv("ALLOWED_ORIGINS", '["http://localhost", "https://example.com"]')
    settings = AppSettings()
    assert settings.ALLOWED_ORIGINS == ["http://localhost", "https://example.com"]

@pytest.mark.parametrize("invalid_value", ["not_a_number", "-1", "9000000"])
def test_invalid_numeric_values_raise_validation_error(monkeypatch, valid_api_key, invalid_value, clean_env):
    """
    ❌ Test: Sayısal alanlara geçersiz değerler (string, negatif, çok büyük) verildiğinde hata fırlatılmalı.
    """
    monkeypatch.setenv("OPENAI_API_KEY", valid_api_key)
    monkeypatch.setenv("PORT", invalid_value)
    with pytest.raises(ValidationError):
        AppSettings()

# === Yeni Eklenen Testler: DSN Validasyonu ===
@pytest.mark.parametrize("invalid_dsn", ["invalid_url", "postgres://user@host", "redis:6379"])
def test_invalid_dsn_raises_validation_error(monkeypatch, valid_api_key, invalid_dsn, clean_env):
    """
    ❌ Test: Geçersiz DSN formatları verildiğinde ValidationError fırlatıldığını doğrular.
    """
    monkeypatch.setenv("OPENAI_API_KEY", valid_api_key)
    monkeypatch.setenv("DATABASE_DSN", invalid_dsn)
    with pytest.raises(ValidationError):
        AppSettings()

def test_valid_dsn_is_accepted(monkeypatch, valid_api_key, clean_env):
    """
    ✅ Test: Geçerli DSN formatlarının sorunsuz kabul edildiğini doğrular.
    """
    monkeypatch.setenv("OPENAI_API_KEY", valid_api_key)
    monkeypatch.setenv("DATABASE_DSN", "postgresql://user:pass@host:5432/db_name")
    settings = AppSettings()
    assert settings.DATABASE_DSN == "postgresql://user:pass@host:5432/db_name"


# === Birim Testleri: Loglama ve Çeviri (i18n) ===

def test_logger_adapter_adds_trace_id(monkeypatch):
    """
    ✅ Test: ContextLoggerAdapter'ın her log kaydına otomatik olarak bir 'trace_id' eklediğini doğrular.
    """
    mock_base_logger = MagicMock()
    adapter = ContextLoggerAdapter(mock_base_logger, {})
    assert trace_id_var.get() is None
    adapter.info("Test mesajı")
    mock_base_logger.info.assert_called_once()
    call_kwargs = mock_base_logger.info.call_args[1]
    assert "extra" in call_kwargs
    assert "trace_id" in call_kwargs["extra"]
    assert len(call_kwargs["extra"]["trace_id"]) == 32

def test_translation_function_default_behavior():
    """
    ✅ Test: Varsayılan _() fonksiyonunun çeviri yapmadan stringi döndürdüğünü doğrular.
    """
    assert _("test message") == "test message"

@patch("config.gettext.translation", side_effect=FileNotFoundError)
def test_load_translation_handles_missing_file(mock_translation):
    """
    ✅ Test: Çeviri dosyası bulunamazsa varsayılan gettext fonksiyonuna geri dönüldüğünü doğrular.
    """
    result = load_translation("locales", "non_existent_locale")
    import gettext
    assert result == gettext.gettext

# === Birim Testleri: Performans ve Gizlilik ===

def test_settings_loading_performance(default_settings):
    """
    ✅ Test: Ayarlar nesnesinin hızlı bir şekilde yüklendiğini doğrular.
    """
    num_loads = 100
    start_time = time.perf_counter()
    for _ in range(num_loads):
        AppSettings()
    duration = time.perf_counter() - start_time
    assert duration < 1.0, f"Settings yüklemesi yavaş: {duration:.3f}s"

def test_secret_value_not_leaked_in_repr(default_settings):
    """
    ✅ Test: Pydantic'in `repr` metodunun SecretStr alanlarını maskelediğini doğrular.
    """
    repr_str = repr(default_settings)
    assert default_settings.OPENAI_API_KEY.get_secret_value() not in repr_str
    assert "**********" in repr_str

def test_export_json_masks_secrets(default_settings):
    """
    ✅ Test: `export_json` metodunun hassas bilgileri (OPENAI_API_KEY) maskelediğini doğrular.
    """
    json_str = default_settings.export_json()
    data = json.loads(json_str)
    assert "OPENAI_API_KEY" in data
    assert data["OPENAI_API_KEY"] == "**********"

# === Yeni Eklenen Test: SecretStr get_secret_value() ===
def test_secret_value_can_be_retrieved(default_settings):
    """
    ✅ Test: SecretStr'in get_secret_value() metodunun hassas değeri doğru döndürdüğünü doğrular.
    """
    secret_key = "sk-testapikey12345678901234567890"
    assert default_settings.OPENAI_API_KEY.get_secret_value() == secret_key


# === Yeni Eklenen Testler: export_json ve Dinamik Güncelleme ===

def test_export_json_structure_and_masking(default_settings):
    """
    ✅ Test: export_json çıktısının JSON formatında olduğunu, 
    tüm beklenen alanları içerdiğini ve secret değerlerin maskelendiğini doğrular.
    """
    json_str = default_settings.export_json()
    data = json.loads(json_str)
    assert "APP_NAME" in data
    assert "OPENAI_API_KEY" in data
    assert "PORT" in data
    assert data["OPENAI_API_KEY"] == "**********"
    assert isinstance(data["PORT"], int)

def test_appsettings_dynamic_update(monkeypatch, valid_api_key, clean_env):
    """
    ✅ Test: AppSettings nesnesi oluşturulduktan sonra alanların güncellenmesi veya yeniden oluşturulması senaryoları.
    """
    monkeypatch.setenv("OPENAI_API_KEY", valid_api_key)
    settings = AppSettings()
    assert settings.PORT == 8000

    # Pydantic modeller default olarak immutable olabilir
    try:
        settings.PORT = 9000
        assert settings.PORT == 9000
    except TypeError:
        new_settings = AppSettings(PORT=9000, OPENAI_API_KEY=valid_api_key)
        assert new_settings.PORT == 9000

    monkeypatch.setenv("PORT", "7000")
    updated_settings = AppSettings()
    assert updated_settings.PORT == 7000

def test_settings_reload_on_env_change(monkeypatch, valid_api_key, clean_env):
    """
    ✅ Test: Ortam değişkenleri değiştiğinde AppSettings nesnesinin yeniden oluşturulması simülasyonu.
    """
    monkeypatch.setenv("OPENAI_API_KEY", valid_api_key)
    monkeypatch.setenv("PORT", "8000")
    settings = AppSettings()
    assert settings.PORT == 8000

    monkeypatch.setenv("PORT", "8500")
    new_settings = AppSettings()
    assert new_settings.PORT == 8500
