# config.py
# Merkezi ayarlar ve logger (önceki 1config.py'nin revizesi)
from pydantic import BaseSettings, Field, SecretStr, validator
from pathlib import Path
from datetime import datetime
import os, sys, json, locale, gettext, uuid, contextvars, logging
from pythonjsonlogger import jsonlogger
from urllib.parse import urlparse

LOCALE_DIR = os.path.join(os.path.dirname(__file__), "locales")
current_locale = os.getenv("APP_LOCALE") or locale.getlocale()[0] or "en"

def load_translation(locale_dir: str, locale_code: str):
    try:
        translation = gettext.translation('messages', localedir=locale_dir, languages=[locale_code])
        return translation.gettext
    except FileNotFoundError:
        return gettext.gettext

_ = load_translation(LOCALE_DIR, current_locale)

# --- Trace/context var for logging ---
trace_id_var = contextvars.ContextVar("trace_id", default=None)

class ContextLoggerAdapter(logging.LoggerAdapter):
    def process(self, msg, kwargs):
        trace_id = trace_id_var.get()
        if not trace_id:
            trace_id = uuid.uuid4().hex
            trace_id_var.set(trace_id)
        extra = self.extra.copy()
        extra.update({"trace_id": trace_id})
        kwargs["extra"] = extra
        return msg, kwargs

# --- Base logger ---
_base_logger = logging.getLogger("ai_assistant")
_base_logger.setLevel(logging.INFO)
if not _base_logger.handlers:
    handler = logging.StreamHandler(sys.stdout)
    fmt = '%(asctime)s %(levelname)s %(name)s %(message)s %(module)s %(trace_id)s'
    handler.setFormatter(jsonlogger.JsonFormatter(fmt))
    _base_logger.addHandler(handler)

app_logger = ContextLoggerAdapter(_base_logger, extra={})

# --- .env loader (simple, uses dotenv if present) ---
def _load_env_file():
    from dotenv import find_dotenv, load_dotenv
    env_mode = os.getenv("ENV_MODE", "local")
    candidate = Path(f".env.{env_mode}")
    if candidate.exists():
        load_dotenv(dotenv_path=str(candidate))
        app_logger.info(f".env loaded: {candidate}")
        return str(candidate)
    default = find_dotenv()
    if default:
        load_dotenv(dotenv_path=default)
        app_logger.info(f".env loaded: {default}")
        return default
    app_logger.warning("No .env file found")
    return None

_env_loaded = _load_env_file()


class AppSettings(BaseSettings):
    """Uygulama konfigürasyon ayarlarını yönetir."""
    # Mimari ayarları
    environment: str = Field("development", description="Uygulama ortamı (development/production).")

    # Redis ayarları
    redis_host: str = Field("localhost", description="Redis sunucu adresi.")
    redis_port: int = Field(6379, description="Redis port numarası.")
    redis_db: int = Field(0, description="Redis veritabanı numarası.")
    
    # JWT ayarları
    jwt_secret: str = Field(..., description="JWT için gizli anahtar. Üretim ortamında zorunludur.")
    jwt_algorithm: str = Field("HS256", description="JWT algoritması.")

    # İşlem ayarları
    max_file_size_mb: int = Field(50, description="İşlenebilecek maksimum dosya boyutu (MB).")
    text_cache_ttl_sec: int = Field(3600, description="Metin önbelleğinin yaşam süresi (saniye).")
    rate_limit: int = Field(5, description="Dakika başına izin verilen maksimum istek sayısı.")
    rate_limit_period: int = Field(1, description="Rate limit periyodu (saniye).")

    # MLOps ayarları
    summary_model_name: str = Field("sshleifer/distilbart-cnn-12-6", description="Özetleme modeli adı.")
    tokenizer_model_name: str = Field("sshleifer/distilbart-cnn-12-6", description="Tokenizer modeli adı.")

    model_config = SettingsConfigDict(env_file='.env', extra='ignore')
    OPENAI_API_KEY: SecretStr = Field(..., env="OPENAI_API_KEY")
    HF_API_KEY: SecretStr | None = Field(None, env="HF_API_KEY")

    VECTOR_DIM: int = Field(384, env="VECTOR_DIM")
    EMBEDDING_MODEL: str = Field("sentence-transformers/all-MiniLM-L6-v2", env="EMBEDDING_MODEL")
    MODEL_NAME: str = Field("gpt-4", env="MODEL_NAME")
    MAX_TOKENS: int = Field(2048, env="MAX_TOKENS")
    MAX_PROMPT_TOKENS: int = Field(1024, env="MAX_PROMPT_TOKENS")
    MAX_HISTORY_LENGTH: int = Field(50, env="MAX_HISTORY_LENGTH")
    CACHE_TTL_SECONDS: int = Field(3600, env="CACHE_TTL_SECONDS")
    PLUGIN_TIMEOUT_SECONDS: int = Field(5, env="PLUGIN_TIMEOUT_SECONDS")
    VECTOR_DB_DSN: str | None = Field(None, env="VECTOR_DB_DSN")
    DATABASE_DSN: str = Field("postgresql://user:pass@localhost:5432/ai_db", env="DATABASE_DSN")
    RATE_LIMIT_PER_MINUTE: int = Field(60, env="RATE_LIMIT_PER_MINUTE")
    ALLOWED_ORIGINS: list[str] = Field(["*"], env="ALLOWED_ORIGINS")

    class Config:
        env_file = _env_loaded
        env_file_encoding = "utf-8"
        case_sensitive = True

    @validator("ALLOWED_ORIGINS", pre=True)
    def parse_origins(cls, v):
        if isinstance(v, str):
            try:
                return json.loads(v)
            except Exception:
                return [s.strip() for s in v.split(",") if s.strip()]
        return v

try:
    app_settings = Settings()
    app_logger.setLevel(app_settings.__getattribute__("LOG_LEVEL") if hasattr(app_settings, "LOG_LEVEL") else logging.INFO)
    app_logger.info(f"Settings loaded. model={app_settings.MODEL_NAME} vector_dim={app_settings.VECTOR_DIM}")
except Exception as e:
    app_logger.critical(f"Failed to load settings: {e}")
    raise

__all__ = ["app_settings", "app_logger", "_env_loaded", "_"]
