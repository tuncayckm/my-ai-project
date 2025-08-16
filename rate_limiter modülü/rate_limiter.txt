import redis
from config import app_settings, app_logger as logger
import time

# RedisRateLimiter'ın merkezi ve standart implementasyonu
class RedisRateLimiter:
    def __init__(self, key_prefix: str, rate: int, per_seconds: int):
        self.key_prefix = key_prefix
        self.rate = rate
        self.per = per_seconds
        try:
            self.redis = redis.Redis(
                host=app_settings.redis_host,
                port=app_settings.redis_port,
                db=app_settings.redis_db,
                decode_responses=True
            )
            self.redis.ping()
        except redis.ConnectionError as e:
            logger.error(f"RedisRateLimiter bağlantı hatası: {e}. Rate limiting devre dışı kalabilir.")
            self.redis = None

    def _get_key(self, user_id: str) -> str:
        return f"{self.key_prefix}:{user_id}"

    def is_allowed(self, user_id: str) -> bool:
        if self.redis is None:
            logger.warning("Redis bağlantısı olmadığından rate limit kontrolü atlanıyor.")
            return True

        key = self._get_key(user_id)
        now = int(time.time())
        
        try:
            with self.redis.pipeline() as pipe:
                # Zaman penceresi dışındaki eski kayıtları sil
                pipe.zremrangebyscore(key, 0, now - self.per)
                # Yeni isteğin zaman damgasını ekle
                pipe.zadd(key, {str(now): now})
                # Mevcut penceredeki toplam istek sayısını al
                pipe.zcard(key)
                # Anahtarın ömrünü uzat
                pipe.expire(key, self.per + 1)
                results = pipe.execute()
            
            current_count = results[2]
            return current_count <= self.rate
        except redis.RedisError as e:
            logger.error(f"Redis rate limiter hatası: {e}. İstek geçici olarak onaylanıyor.")
            return True # Hata durumunda sistemi kilitlememek için isteğe izin ver