# database.py
# Async PostgreSQL DatabaseManager (revize)
from . import utils  # veya `from utils import ...` projenizin yapısına göre değişir
import asyncpg
import json
import logging
from typing import Optional, List, Dict, Any, Union
from pathlib import Path
import functools
import asyncio
from config import app_settings, app_logger

logger = app_logger  # use centralized logger

def async_retry(retries: int = 3, delay: float = 1.0, backoff: float = 2.0):
    """
    Gelişmiş async retry decorator. Hata alınırsa bekleme süresini üstel olarak artırarak tekrar dener.
    """
    def decorator(func: Callable[..., Coroutine[Any, Any, Any]]):
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            last_exc = None
            current_delay = delay
            for attempt in range(1, retries + 1):
                try:
                    return await func(*args, **kwargs)
                except Exception as e:
                    last_exc = e
                    logger.warning(f"Retry {attempt}/{retries} for {func.__name__} due to error: {e}")
                    if attempt < retries:
                        logger.info(f"Waiting for {current_delay:.2f} seconds before next attempt.")
                        await asyncio.sleep(current_delay)
                        current_delay *= backoff
            logger.error(f"All retries failed for {func.__name__}: {last_exc}")
            raise last_exc
        return wrapper
    return decorator

class DatabaseManager:
    def __init__(self, dsn: Optional[str] = None):
        self.dsn = dsn or app_settings.DATABASE_DSN
        self._pool: Optional[asyncpg.Pool] = None

    @async_retry(retries=5, delay=2, backoff=2) # 5 deneme, 2 saniye başlangıç gecikmesi ile
    
    async def connect(self):
        """PostgreSQL veritabanı bağlantı havuzunu oluşturur."""
        if self.pool is None:
            try:
                self.pool = await asyncpg.create_pool(
                    dsn=self.dsn,
                    min_size=1,
                    max_size=20,
                    command_timeout=60,
                    loop=asyncio.get_event_loop()
                )
                self.logger.info("Veritabanı bağlantı havuzu başarıyla oluşturuldu.")
            except Exception as e:
                self.logger.error(f"Veritabanı bağlantısı oluşturulamadı: {e}")
                raise

    async def connect(self):
        if self._pool:
            return
        try:
            self._pool = await asyncpg.create_pool(dsn=self.dsn, min_size=2, max_size=10)
            logger.info("DB pool created")
            await self._run_migrations()
        except Exception as e:
            logger.critical(f"DB connect error: {e}")
            raise

    async def close(self):
        if self._pool:
            await self._pool.close()
            self._pool = None
            logger.info("DB pool closed")

    async def _run_migrations(self):
        """Veritabanı şemasını günceller. `migrations` klasöründeki SQL dosyalarını çalıştırır."""
        migrations_dir = Path("migrations")
        if not migrations_dir.exists():
            self.logger.warning("Migrations klasörü bulunamadı. Migration işlemi atlanıyor.")
            return

        migration_files = sorted([f for f in migrations_dir.glob("*.sql") if f.is_file()])
        if not migration_files:
            self.logger.info("Migrations klasöründe çalıştırılacak dosya bulunamadı.")
            return

        async with self.pool.acquire() as conn:
            async with conn.transaction():
                # Migration geçmişi tablosunu oluştur
                await conn.execute("""
                    CREATE TABLE IF NOT EXISTS alembic_version (
                        version_num VARCHAR(32) NOT NULL PRIMARY KEY
                    );
                """)
                
                # En son migration'ı al
                latest_migration = await conn.fetchval(
                    "SELECT version_num FROM alembic_version ORDER BY version_num DESC LIMIT 1"
                )

                for file in migration_files:
                    migration_version = file.stem.split('_')[0]
                    if latest_migration and migration_version <= latest_migration:
                        self.logger.info(f"Migration {file.name} zaten çalıştırılmış, atlanıyor.")
                        continue

                    self.logger.info(f"Migration {file.name} çalıştırılıyor...")
                    try:
                        with open(file, 'r', encoding='utf-8') as f:
                            sql = f.read()
                        await conn.execute(sql)
                        await conn.execute("INSERT INTO alembic_version (version_num) VALUES ($1)", migration_version)
                        self.logger.info(f"Migration {file.name} başarıyla tamamlandı.")
                    except Exception as e:
                        self.logger.error(f"Migration {file.name} sırasında hata oluştu: {e}")
                        raise # Hata durumunda işlemi geri al

    async def load_profile(self, user_id: str) -> Optional[Dict[str,Any]]:
        if not self._pool:
            raise RuntimeError("DatabaseManager not connected")
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow("SELECT user_id, name, preferences, history FROM user_profiles WHERE user_id = $1", user_id)
            if not row:
                return None
            return {
                "user_id": row["user_id"],
                "name": row["name"],
                "preferences": row["preferences"] or {},
                "history": row["history"] or []
            }

    async def save_profile(self, user_id: str, name: str, preferences: dict, history: list):
        if not self._pool:
            raise RuntimeError("DatabaseManager not connected")
        async with self._pool.acquire() as conn:
            await conn.execute("""
                INSERT INTO user_profiles (user_id, name, preferences, history)
                VALUES ($1, $2, $3::jsonb, $4::jsonb)
                ON CONFLICT (user_id) DO UPDATE SET
                    name = EXCLUDED.name,
                    preferences = EXCLUDED.preferences,
                    history = EXCLUDED.history;
            """, user_id, name, json.dumps(preferences), json.dumps(history))

    async def save_memory(self, user_id: str, text: str, embedding: List[float], source: str):
        if not self._pool:
            raise RuntimeError("DatabaseManager not connected")
        emb_blob = serialize_embedding(embedding)
        async with self._pool.acquire() as conn:
            await conn.execute("""
                INSERT INTO memory_store (user_id, text, embedding, source)
                VALUES ($1, $2, $3, $4)
            """, user_id, text, emb_blob, source)

    async def fetch_all_histories(self) -> List[list]:
        if not self._pool:
            raise RuntimeError("DatabaseManager not connected")
        async with self._pool.acquire() as conn:
            rows = await conn.fetch("SELECT history FROM user_profiles WHERE history IS NOT NULL;")
            lists = []
            for r in rows:
                try:
                    # asyncpg returns already parsed JSONB as Python object
                    history = r["history"]
                    if isinstance(history, list):
                        lists.append(history)
                except Exception:
                    logger.warning("Invalid history row skipped")
            return lists

__all__ = ["DatabaseManager", "serialize_embedding", "deserialize_embedding"]
