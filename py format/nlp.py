import hashlib
import json
from datetime import datetime, timedelta
from typing import Optional
import numpy as np
import faiss
from cachetools import TTLCache
import asyncio
from schemas import ApplyKnowledgeRequest # Merkezi şemayı import et
from core.auth import verify_jwt_token
from config import app_settings, app_logger as logger
from database import DatabaseManager
from langdetect import detect
from nltk.tokenize import sent_tokenize
import tiktoken
from core.embedding import embed_texts
from core.auth import verify_jwt_token
from core.utils import sanitize_input

# Cache ve tokenizer
_context_cache = TTLCache(maxsize=1000, ttl=app_settings.CACHE_TTL_SECONDS)
_tokenizer = tiktoken.encoding_for_model(app_settings.MODEL_NAME)

# Prometheus metrikleri
from prometheus_client import Counter, Histogram

APPLY_KNOWLEDGE_REQUEST_COUNT = Counter('apply_knowledge_requests_total', 'apply_knowledge çağrıları', ['status'])
APPLY_KNOWLEDGE_LATENCY = Histogram('apply_knowledge_latency_seconds', 'apply_knowledge gecikmesi')

# Rate limiter (örnek kullanıcı bazlı)
from rate_limiter import RedisRateLimiter
rate_limiter = RedisRateLimiter(
    key_prefix="apply_knowledge", 
    rate=app_settings.RATE_LIMIT_PER_MINUTE, # config'den gelen değeri kullan
    per_seconds=60

# Audit logger
import logging
audit_logger = logging.getLogger("audit")
audit_logger.setLevel(logging.INFO)
if not audit_logger.hasHandlers():
    fh = logging.FileHandler("audit.log")
    audit_logger.addHandler(fh)

def estimate_token_count(texts):
    if isinstance(texts, str):
        texts = [texts]
    return sum(len(_tokenizer.encode(t)) for t in texts)


def safe_truncate_nlp(text: str, max_tokens: int) -> str:
    tokens = _tokenizer.encode(text)
    if len(tokens) <= max_tokens:
        return text
    sents = sent_tokenize(text)
    out = ""
    used = 0
    for s in sents:
        s_len = len(_tokenizer.encode(s))
        if used + s_len > max_tokens:
            break
        out += s + " "
        used += s_len
    return out.strip()

def invalidate_context_cache_for_user(user_id: str):
    keys = [k for k in list(_context_cache.keys()) if k[0] == user_id]
    for k in keys:
        del _context_cache[k]
    logger.debug(f"Cache invalidated for {user_id}")


async def run_plugin_async(plugin_cls, plugin_data, timeout: int = None):
    plugin_instance = plugin_cls(plugin_data)
    try:
        return await asyncio.wait_for(plugin_instance.run(), timeout=timeout or app_settings.PLUGIN_TIMEOUT_SECONDS)
    except asyncio.TimeoutError:
        logger.warning("Plugin execution timed out")
        return None


async def apply_knowledge(
    db_manager: DatabaseManager,
    user_id: str,
    prompt: str,
    token: str,
    plugin_data: Optional[dict] = None,
    max_context_tokens: Optional[int] = None
) -> str:
    """
    Kullanıcının prompt'una göre LLM için context oluşturur ve cevap üretir.

    Args:
        db_manager (DatabaseManager): Veritabanı bağlantısı ve işlemleri için instance.
        user_id (str): İstek yapan kullanıcı ID'si.
        prompt (str): Kullanıcının sorduğu veya verdiği metin.
        token (str): JWT doğrulama token'ı.
        plugin_data (Optional[dict], optional): Plugin için opsiyonel parametreler. Defaults to None.
        max_context_tokens (Optional[int], optional): Maksimum context token sayısı. Defaults to None.

    Returns:
        str: JSON formatında rol bazlı mesaj dizisi.
    """
    try:
        # Input validasyonu
        req = ApplyKnowledgeRequest(
            user_id=user_id,
            prompt=prompt,
            token=token,
            plugin_data=plugin_data,
            max_context_tokens=max_context_tokens
        )
    except ValidationError as ve:
        logger.error(f"Input validation failed: {ve}")
        APPLY_KNOWLEDGE_REQUEST_COUNT.labels(status='validation_error').inc()
        return json.dumps([{"role":"system","content":"Geçersiz istek verisi."}])

    if not sanitize_input(req.user_id):
        logger.warning("Invalid user_id format")
        APPLY_KNOWLEDGE_REQUEST_COUNT.labels(status='invalid_user_id').inc()
        return json.dumps([{"role":"system","content":"Geçersiz kullanıcı ID'si."}])

    # Rate limiting
    if not rate_limiter.is_allowed(req.user_id):
        logger.warning(f"Rate limit exceeded: {req.user_id}")
        APPLY_KNOWLEDGE_REQUEST_COUNT.labels(status='rate_limited').inc()
        return json.dumps([{"role":"system","content":"Çok fazla istek yaptınız, lütfen biraz bekleyin."}])

    try:
        # JWT doğrulama ve RBAC
        payload = verify_jwt_token(req.token, expected_user_id=req.user_id)
        scopes = payload.get("scopes", []) if isinstance(payload, dict) else []
        roles = payload.get("roles", []) if isinstance(payload, dict) else []

        if "admin" not in roles and plugin_data and plugin_data.get("protected_resource", False):
            logger.warning("RBAC: insufficient role to use plugin on protected_resource")
            APPLY_KNOWLEDGE_REQUEST_COUNT.labels(status='rbac_denied').inc()
            return json.dumps([{"role":"system","content":"Yetersiz yetki."}])

        # Profil yükle
        await db_manager.connect()
        profile = await db_manager.load_profile(req.user_id) or {}

        history = profile.get("history", []) if isinstance(profile.get("history", []), list) else []
        preferences = profile.get("preferences", {}) if isinstance(profile.get("preferences", {}), dict) else {}

        raw_prompt = sanitize_input(req.prompt)

        # Dil tespiti
        lang = preferences.get("language")
        if not lang:
            try:
                lang = detect(raw_prompt)
            except Exception:
                lang = None

        # Geçmiş temizleme ve filtreleme
        cutoff = datetime.utcnow() - timedelta(days=30)
        filtered = []
        for h in history:
            try:
                ts = datetime.fromisoformat(h.get("timestamp"))
            except Exception:
                ts = None
            if ts and ts < cutoff:
                continue
            filtered.append(h)

        filtered.append({"role": "user", "content": raw_prompt, "timestamp": datetime.utcnow().isoformat()})

        # Plugin çağrısı (async)
        plugin_output = None
        if plugin_data and 'plugin_class' in plugin_data:
            plugin_output = await run_plugin_async(plugin_data['plugin_class'], plugin_data)

        # Context oluşturma
        context = await _get_context_from_vector(
            db_manager, raw_prompt, req.user_id,
            lang=lang, time_window_days=30,
            top_n=5,
            max_token_context=max_context_tokens or app_settings.MAX_PROMPT_TOKENS // 2
        )

        messages = [{"role": "system", "content": "You are a helpful assistant."}]
        if context:
            messages.append({"role": "system", "content": context})

        # Tarihe göre sınırlı geçmiş ekle
        trimmed = filtered[-app_settings.MAX_HISTORY_LENGTH:]
        for e in trimmed:
            messages.append({"role": e.get("role", "user"), "content": e.get("content", "")})

        # Plugin sonucu varsa ekle
        if plugin_output:
            messages.append({"role": "assistant", "content": str(plugin_output)})

        messages.append({"role": "user", "content": raw_prompt})

        # Profil güncelle ve cache temizle (await edilip hatalar loglanır)
        await db_manager.save_profile(req.user_id, profile.get("name", "unknown"), preferences, filtered)
        invalidate_context_cache_for_user(req.user_id)

        APPLY_KNOWLEDGE_REQUEST_COUNT.labels(status='success').inc()
        return json.dumps(messages, ensure_ascii=False, indent=2)

    except Exception as e:
        logger.error(f"apply_knowledge error: {e}", exc_info=True)
        APPLY_KNOWLEDGE_REQUEST_COUNT.labels(status='error').inc()
        return json.dumps([{"role":"system","content":"Bir hata oluştu, lütfen daha sonra tekrar deneyin."}])

# _get_context_from_vector fonksiyon aynen kalabilir, gerekirse Prometheus ekleyebilirim.
