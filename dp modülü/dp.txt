import aiosqlite
import json
from datetime import datetime
from typing import Optional, List, Dict, Any, Callable, Coroutine
from aiosqlite import OperationalError, DatabaseError, ProgrammingError
from . import utils  # veya `from utils import ...` projenizin yapısına göre değişir
from config import app_logger as logger
from dp import async_retry, OperationalError
import asyncio
import functools
import os
import contextlib



class AsyncDatabaseManager:
    """
    Asenkron SQLite veritabanı yöneticisi. 
    Bağlantı havuzu, migration, transaction ve retry içerir.
    """

    def __init__(self, db_file: str, pool_size: int = 5, timeout: float = 10.0):
        self.db_file = db_file
        self.pool_size = pool_size
        self.timeout = timeout
        self.pool: List[aiosqlite.Connection] = []
        self.pool_lock = asyncio.Lock()
        self.schema_version = 2

    @async_retry(max_retries=3, delay=5, backoff=2, exceptions=(OperationalError,))
    async def _create_connection(self) -> aiosqlite.Connection:
        conn = await aiosqlite.connect(self.db_file)
        conn.row_factory = aiosqlite.Row
        await conn.execute("PRAGMA journal_mode=WAL;")
        return conn

    async def _init_pool(self):
        if not self.pool:
            for _ in range(self.pool_size):
                conn = await self._create_connection()
                self.pool.append(conn)
            await self._run_migrations()

    async def _get_connection(self) -> aiosqlite.Connection:
        await self._init_pool()
        async with self.pool_lock:
            # Basit round robin
            conn = self.pool.pop(0)
            self.pool.append(conn)
            return conn

    async def _run_migrations(self):
        # Basit migration mekanizması
        conn = await self._get_connection()
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS schema_version (
                version INTEGER PRIMARY KEY
            )
        """)
        await conn.commit()

        async with conn.execute("SELECT version FROM schema_version ORDER BY version DESC LIMIT 1") as cursor:
            row = await cursor.fetchone()
            current_version = row["version"] if row else 0

        if current_version < self.schema_version:
            logger.info(f"Migration gerekiyor: {current_version} -> {self.schema_version}")
            await self._migrate(conn, current_version, self.schema_version)
        else:
            logger.info(f"Schema güncel: {current_version}")
    
    async def _migrate(self, conn: aiosqlite.Connection, old_version: int, new_version: int):
        # Örnek migration
        try:
            if old_version < 1:
                await conn.execute("""
                    CREATE TABLE IF NOT EXISTS memory_store (
                        id TEXT PRIMARY KEY,
                        user_id TEXT,
                        text TEXT,
                        embedding BLOB
                    )
                """)
                await conn.execute("""
                    CREATE INDEX IF NOT EXISTS idx_memory_store_user_id ON memory_store(user_id)
                """)
                await conn.execute("""
                    CREATE TABLE IF NOT EXISTS user_profiles (
                        user_id TEXT PRIMARY KEY,
                        name TEXT,
                        preferences TEXT
                    )
                """)
                await conn.execute("""
                    CREATE TABLE IF NOT EXISTS user_history (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        user_id TEXT,
                        timestamp TEXT,
                        prompt TEXT,
                        response TEXT
                    )
                """)
                await conn.execute("""
                    CREATE INDEX IF NOT EXISTS idx_user_history_user_id ON user_history(user_id)
                """)
                await conn.execute("INSERT INTO schema_version (version) VALUES (?)", (1,))
                await conn.commit()
                logger.info("Migration 1 tamamlandı.")
           # BURADA: Yeni versiyon migration kodunu ekleyin (v1 -> v2)
            if old_version < 2:
                try:
                    await conn.execute("""ALTER TABLE user_profiles ADD COLUMN last_login TEXT""")
                    await conn.execute("UPDATE schema_version SET version = ?", (2,))
                    await conn.commit()
                    logger.info("Migration 2 tamamlandı: user_profiles tablosuna last_login eklendi.")
                except OperationalError as e:
                    # Sütun zaten varsa hata vermesini önler
                    if "duplicate column name" in str(e):
                        logger.warning("Migration 2 çalıştırıldı, ancak sütun zaten var.")
                    else:
                        raise e

                
            # Son olarak, yeni şema versiyonunu kaydedin.
            if new_version > old_version:
                await conn.execute("INSERT OR REPLACE INTO schema_version (version) VALUES (?)", (new_version,))
                await conn.commit()
                logger.info(f"Migrationlar tamamlandı. Yeni versiyon: {new_version}")

        except Exception as e:
            logger.error(f"Migration hatası: {e}")
            raise

    @contextlib.asynccontextmanager
    async def transaction(self):
        """Asenkron veritabanı işlemi (transaction) yöneticisi."""
        conn = await self._get_connection()
        try:
            yield conn
            await conn.commit()
        except Exception:
            await conn.rollback()
            raise

    async def close(self):
        """Havuzdaki tüm bağlantıları kapatır."""
        async with self.pool_lock:
            for conn in self.pool:
                await conn.close()
            self.pool.clear()
            logger.info("Veritabanı bağlantıları kapatıldı.")

    @async_retry()
    async def save_profile(self, user_id: str, name: str, preferences: Dict[str, Any]):
        """
        Kullanıcı profili kaydeder veya günceller.
        """
        if not isinstance(preferences, dict):
            raise ValueError("Preferences dict tipinde olmalı.")
        conn = await self._get_connection()
        prefs_json = json.dumps(preferences, ensure_ascii=False)
        try:
            await asyncio.wait_for(
                conn.execute("""
                    INSERT OR REPLACE INTO user_profiles (user_id, name, preferences)
                    VALUES (?, ?, ?)
                """, (user_id, name, prefs_json)),
                timeout=self.timeout
            )
            await conn.commit()
            logger.info(f"Profil kaydedildi: {user_id}")
        except asyncio.TimeoutError:
            logger.error("Profil kaydetme işlemi timeout oldu.")
            raise

        except (OperationalError, DatabaseError, ProgrammingError) as e:
            logger.error(f"Veritabanı hatası oluştu: {e}")
            raise
            
        except Exception as e:
            logger.error(f"Beklenmedik bir hata oluştu: {e}[cite: 22].")
            raise

        
    @async_retry()
    async def load_profile(self, user_id: str) -> Optional[Dict[str, Any]]:
        """
        Kullanıcı profili yükler.
        """
        conn = await self._get_connection()
        try:
            async with asyncio.wait_for(
                conn.execute("""
                    SELECT user_id, name, preferences FROM user_profiles WHERE user_id = ?
                """, (user_id,)),
                timeout=self.timeout
            ) as cursor:
                row = await cursor.fetchone()
                if not row:
                    logger.warning(f"Profil bulunamadı: {user_id}")
                    return None
                return {
                    "user_id": row["user_id"],
                    "name": row["name"],
                    "preferences": json.loads(row["preferences"])
                }
        except asyncio.TimeoutError:
            logger.error("Profil yükleme işlemi timeout oldu.")
            raise
        except Exception as e:
            logger.error(f"Profil yükleme hatası: {e}")
            raise

    @async_retry()
    async def update_profile_preferences(self, user_id: str, preferences: Dict[str, Any]):
        """
        Kullanıcı tercihlerini günceller.
        """
        if not isinstance(preferences, dict):
            raise ValueError("Preferences dict tipinde olmalı.")
        conn = await self._get_connection()
        prefs_json = json.dumps(preferences, ensure_ascii=False)
        try:
            await asyncio.wait_for(
                conn.execute("""
                    UPDATE user_profiles SET preferences = ? WHERE user_id = ?
                """, (prefs_json, user_id)),
                timeout=self.timeout
            )
            await conn.commit()
            logger.info(f"Preferences güncellendi: {user_id}")
        except asyncio.TimeoutError:
            logger.error("Preferences güncelleme işlemi timeout oldu.")
            raise
        except Exception as e:
            logger.error(f"Preferences güncelleme hatası: {e}")
            raise

    @async_retry()
    async def add_to_history(self, user_id: str, prompt: str, response: str):
        """
        Kullanıcı geçmişine prompt-response ekler.
        """
        timestamp = datetime.utcnow().isoformat()
        conn = await self._get_connection()
        try:
            await asyncio.wait_for(
                conn.execute("""
                    INSERT INTO user_history (user_id, timestamp, prompt, response)
                    VALUES (?, ?, ?, ?)
                """, (user_id, timestamp, prompt, response)),
                timeout=self.timeout
            )
            await conn.commit()
            logger.info(f"History eklendi: {user_id} - {timestamp}")
        except asyncio.TimeoutError:
            logger.error("History ekleme işlemi timeout oldu.")
            raise
        except Exception as e:
            logger.error(f"History ekleme hatası: {e}")
            raise

    @async_retry()
    async def get_history(self, user_id: str, limit: int = 10, start_time: Optional[str] = None, end_time: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Kullanıcı geçmişini veritabanından getirir.
        Tarih aralığı ve limit parametreleri desteklenir.

        Args:
            user_id (str): Geçmişi alınacak kullanıcının ID'si.
            limit (int): Getirilecek kayıt sayısı. Varsayılan: 10.
            start_time (Optional[str]): Başlangıç tarihi (ISO formatında).
            end_time (Optional[str]): Bitiş tarihi (ISO formatında).

        Returns:
            List[Dict[str, Any]]: Geçmiş kayıtlarının bir listesi. Her kayıt bir sözlüktür.

        Raises:
            asyncio.TimeoutError: Veritabanı işlemi zaman aşımına uğrarsa.
            Exception: Beklenmeyen bir hata oluşursa.
        """
        conn = await self._get_connection()
        query = "SELECT timestamp, prompt, response FROM user_history WHERE user_id = ?"
        params = [user_id]

        if start_time and end_time:
            query += " AND timestamp BETWEEN ? AND ?"
            params.extend([start_time, end_time])
        elif start_time:
            query += " AND timestamp >= ?"
            params.append(start_time)
        elif end_time:
            query += " AND timestamp <= ?"
            params.append(end_time)

        query += " ORDER BY timestamp DESC LIMIT ?"
        params.append(limit)

        histories = []
        try:
            async with asyncio.wait_for(conn.execute(query, tuple(params)), timeout=self.timeout) as cursor:
                async for row in cursor:
                    histories.append({
                        "timestamp": row["timestamp"],
                        "prompt": row["prompt"],
                        "response": row["response"]
                    })
            return histories
        except asyncio.TimeoutError:
            logger.error("History çekme işlemi timeout oldu.")
            raise
        except Exception as e:
            logger.error(f"History çekme hatası: {e}")
            raise

    @async_retry()
    async def save_memory(self, mem_id: str, user_id: str, text: str, embedding: List[float]):
        """
        Hafıza kaydı ekler veya günceller. Embedding liste olarak alınır ve bytesa çevrilir.
        """
        if not isinstance(embedding, list):
            raise ValueError("Embedding tipi liste olmalı.")
        embedding_blob = utils.serialize_embedding(embedding)
        conn = await self._get_connection()
        try:
            await asyncio.wait_for(
                conn.execute("""
                    INSERT OR REPLACE INTO memory_store (id, user_id, text, embedding)
                    VALUES (?, ?, ?, ?)
                """, (mem_id, user_id, text, embedding_blob)),
                timeout=self.timeout
            )
            await conn.commit()
            logger.info(f"Hafıza kaydedildi: {mem_id} - {user_id}")
        except asyncio.TimeoutError:
            logger.error("Hafıza kaydetme işlemi timeout oldu.")
            raise
        except Exception as e:
            logger.error(f"Hafıza kaydetme hatası: {e}")
            raise

    @async_retry()
    async def load_memories(self, user_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Hafıza kayıtlarını getirir. Kullanıcı bazında veya tümü.
        """
        conn = await self._get_connection()
        if user_id:
            query = "SELECT * FROM memory_store WHERE user_id = ?"
            params = (user_id,)
        else:
            query = "SELECT * FROM memory_store"
            params = ()
        memories = []
        try:
            async with asyncio.wait_for(conn.execute(query, params), timeout=self.timeout) as cursor:
                async for row in cursor:
                    memories.append({
                        "id": row["id"],
                        "user_id": row["user_id"],
                        "text": row["text"],
                        "embedding": utils.deserialize_embedding(row["embedding"])
                    })
            return memories
        except asyncio.TimeoutError:
            logger.error("Hafıza yükleme işlemi timeout oldu.")
            raise
        except Exception as e:
            logger.error(f"Hafıza yükleme hatası: {e}")
            raise

    @async_retry()
    async def delete_profile(self, user_id: str):
        """
        Profil ve ilişkili geçmiş/hafıza kayıtlarını siler.
        """
        try:
            async with self.transaction() as conn:
                await conn.execute("DELETE FROM user_profiles WHERE user_id = ?", (user_id,))
                await conn.execute("DELETE FROM user_history WHERE user_id = ?", (user_id,))
                await conn.execute("DELETE FROM memory_store WHERE user_id = ?", (user_id,))
            logger.info(f"Profil ve ilişkili veriler silindi: {user_id}")
        except Exception as e:
            logger.error(f"Profil silme hatası: {e}")
            raise

    @async_retry()
    async def delete_memory(self, mem_id: str):
        """
        Belirli bir hafıza kaydını siler.
        """
        conn = await self._get_connection()
        try:
            await asyncio.wait_for(conn.execute("DELETE FROM memory_store WHERE id = ?", (mem_id,)), timeout=self.timeout)
            await conn.commit()
            logger.info(f"Hafıza kaydı silindi: {mem_id}")
        except asyncio.TimeoutError:
            logger.error("Hafıza silme işlemi timeout oldu.")
            raise
        except Exception as e:
            logger.error(f"Hafıza silme hatası: {e}")
            raise


# Test için çalışma örneği
if __name__ == "__main__":

    async def test():
        db_path = os.getenv("DB_FILE_PATH", "test_async_db.sqlite")
        if os.path.exists(db_path):
            os.remove(db_path)

        async with AsyncDatabaseManager(db_path) as db:
            await db.save_profile("user1", "Test User", {"theme": "dark", "lang": "tr"})
            profile = await db.load_profile("user1")
            print("Loaded profile:", profile)

            await db.add_to_history("user1", "Hello?", "Hi!")
            history = await db.get_history("user1")
            print("User history:", history)

            embedding_sample = [0.1, 0.2, 0.3, 0.4]
            await db.save_memory("mem1", "user1", "Sample memory text", embedding_sample)
            memories = await db.load_memories("user1")
            print("User memories:", memories)

            await db.delete_memory("mem1")
            await db.delete_profile("user1")

    asyncio.run(test())