# embedding.py
from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel
from typing import List
import numpy as np
import faiss
import json
import asyncio
from datetime import datetime

from config import app_settings, app_logger as logger
from database import DatabaseManager
from core.embedding import embed_texts
from core.auth import verify_jwt_token
import redis.asyncio as redis_async

# Redis bağlantısı
redis = None
index_lock = asyncio.Lock()
faiss_index = None

# FastAPI router
router = APIRouter(prefix="/memory", tags=["Memory"])

# DB Manager (paylaşımlı kullanılacak)
db_manager = DatabaseManager()

class MemoryIn(BaseModel):
    user_id: str
    text: str
    source: str = "text"

async def init_redis():
    """Redis bağlantısını başlatır."""
    global redis
    redis = redis_async.from_url(app_settings.REDIS__HOST or "redis://localhost:6379", decode_responses=True)
    logger.info("Redis connection initialized.")

def init_faiss():
    """FAISS index oluşturur."""
    global faiss_index
    faiss_index = faiss.IndexFlatIP(app_settings.VECTOR_DIM)
    logger.info("FAISS index initialized (FlatIP).")

async def save_faiss_index(path: str = "faiss_index.index"):
    """FAISS indexi diske yazar."""
    async with index_lock:
        faiss.write_index(faiss_index, path)
        logger.info(f"FAISS index saved to {path}")

async def update_faiss_index(since: datetime = None):
    """Veritabanındaki yeni embeddinglerle FAISS indexini artımlı olarak günceller."""
    async with index_lock:
        logger.info("Checking for new embeddings to update FAISS index...")
        
        # Sadece belirli bir zamandan sonra eklenen kayıtları çek
        records = await db_manager.fetch_new_embeddings(since)
        
        if not records:
            logger.info("No new embeddings found to update.")
            return

        embeddings = []
        ids = []
        for history in records:
            # history, birden fazla kaydı içerebilir
            for h in history:
                if "embedding" in h and "id" in h:
                    embeddings.append(np.array(h["embedding"], dtype=np.float32))
                    # Veritabanından gelen ID'yi, FAISS index'indeki ID'ye dönüştür
                    ids.append(int(h["id"]))

        if not embeddings:
            logger.warning("No valid new embeddings found.")
            return
        
        np_embs = np.array(embeddings).astype("float32")
        faiss.normalize_L2(np_embs)
        
        # Yeni embeddingleri ve ID'lerini indexe ekle
        faiss_index.add_with_ids(np_embs, np.array(ids))

        logger.info(f"FAISS index updated with {len(embeddings)} new embeddings.")

async def remove_embedding_from_index(id_to_remove: int):
    """Belirtilen ID'ye sahip embedding'i FAISS index'inden siler."""
    async with index_lock:
        id_array = np.array([id_to_remove]).astype('int64')
        faiss_index.remove_ids(id_array)
        logger.info(f"Embedding with ID {id_to_remove} removed from FAISS index.")
  

async def save_embedding_to_store(user_id: str, content: str, source: str = "text") -> bool:
    """Metin embedding oluşturup veritabanına kaydeder."""
    try:
        embedding = embed_texts([content], model=app_settings.EMBEDDING_MODEL)[0]
        await db_manager.save_memory(user_id, content, embedding, source)
        logger.info(f"Embedding saved: {source}")
        return True
    except Exception as e:
        logger.error(f"Error saving embedding: {e}")
        return False

# JWT doğrulama dependency
async def get_current_user(token: str) -> str:
    try:
        payload = verify_jwt_token(token)
        user_id = payload.get("sub")
        if not user_id:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token: no user_id")
        return user_id
    except Exception as e:
        logger.warning(f"JWT verification failed: {e}")
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token")

@router.post("/")
async def save_embedding_route(data: MemoryIn, token: str = Depends(get_current_user)):
    """Embedding kaydetme endpointi."""
    ok = await save_embedding_to_store(data.user_id, data.text, data.source)
    if not ok:
        raise HTTPException(status_code=500, detail="Failed to save embedding")
    return {"status": "success"}

@router.get("/search/")
async def search_similar(query: str, top_k: int = 5, token: str = Depends(get_current_user)):
    """Verilen sorguya benzer hafıza kayıtlarını döndürür."""
    cache_key = f"search:{query}:{top_k}"
    try:
        cached = await redis.get(cache_key)
        if cached:
            return json.loads(cached)

        vector = embed_texts([query], model=app_settings.EMBEDDING_MODEL)[0]
        vector = vector / np.linalg.norm(vector) if np.linalg.norm(vector) > 0 else vector
        vector = np.array([vector], dtype=np.float32)

        async with index_lock:
            if faiss_index is None or faiss_index.ntotal == 0:
                return []
            distances, indices = faiss_index.search(vector, top_k)

        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx == -1:
                continue
            # Burada index -> kayıt eşlemesi DB'den yapılır
            results.append({
                "id": idx,
                "score": float(dist)
            })

        await redis.set(cache_key, json.dumps(results), ex=300)
        return results
    except Exception as e:
        logger.error(f"Search error: {e}")
        raise HTTPException(status_code=500, detail="Search error")

# Startup initializer
async def startup_embedding_module():
    await db_manager.connect()
    await init_redis()
    init_faiss()
    logger.info("Embedding module initialized.")

__all__ = ["router", "startup_embedding_module", "save_embedding_to_store", "update_faiss_index"]
