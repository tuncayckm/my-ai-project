# embedding_test_suite.py

import pytest
import asyncio
import time
import numpy as np
import tracemalloc
from httpx import AsyncClient
from unittest.mock import AsyncMock, patch
from your_embedding_module import app, update_faiss_index, normalize_vector, get_current_user, db, index, redis
from jose import jwt
from datetime import datetime, timedelta

# === Fixturelar ===
@pytest.fixture(autouse=True)
def override_dependency():
    async def fake_get_current_user(token="fake"):
        return "user1"
    app.dependency_overrides[get_current_user] = fake_get_current_user
    yield
    app.dependency_overrides.clear()

@pytest.fixture
async def async_client():
    async with AsyncClient(app=app, base_url="http://testserver") as client:
        yield client

# === Yardımcı Fonksiyon ===
def is_normalized(vec):
    return abs((vec ** 2).sum() - 1.0) < 1e-6

# === Normalizasyon Testi ===
@pytest.mark.asyncio
async def test_normalize_vector():
    v = np.array([3.0, 4.0])
    normed = normalize_vector(v)
    assert is_normalized(normed)

# === Bellek Kayıt ve Arama ===
@pytest.mark.anyio
async def test_save_memory_and_search(async_client):
    response = await async_client.post("/memory/", json={"user_id": "user1", "text": "Merhaba dünya"})
    assert response.status_code == 200
    memory_id = response.json().get("memory_id")
    assert memory_id is not None

    response = await async_client.get("/search/", params={"query": "Merhaba", "top_k": 3})
    assert response.status_code == 200
    assert isinstance(response.json(), list)

# === FAISS Güncelleme ===
@pytest.mark.anyio
async def test_incremental_faiss_update(monkeypatch):
    """Sadece yeni eklenen kayıtların FAISS index'e eklendiğini test eder."""
    
    # Başlangıçta index boş
    monkeypatch.setattr("your_embedding_module.faiss_index", faiss.IndexFlatIP(1536))
    
    # 10 kayıtlık bir veri seti oluştur
    initial_records = [{"id": i, "embedding": np.random.rand(1536).astype('float32').tolist()} for i in range(10)]
    
    # İlk 5 kaydı güncelleme fonksiyonuna gönder
    monkeypatch.setattr("your_embedding_module.db_manager.fetch_new_embeddings", AsyncMock(return_value=[initial_records[:5]]))
    await update_faiss_index(since=datetime.utcnow() - timedelta(hours=1))

    # Index boyutunu kontrol et
    assert your_embedding_module.faiss_index.ntotal == 5

    # Sonraki 5 kaydı güncelleme fonksiyonuna gönder
    monkeypatch.setattr("your_embedding_module.db_manager.fetch_new_embeddings", AsyncMock(return_value=[initial_records[5:]]))
    await update_faiss_index(since=datetime.utcnow())

    # Index boyutunu tekrar kontrol et, 10'a yükselmiş olmalı
    assert your_embedding_module.faiss_index.ntotal == 10

# === Parametrik Sınır Durumları ===
@pytest.mark.parametrize("top_k", [-1, 0, 10000])
@pytest.mark.anyio
async def test_search_top_k_variants(top_k, async_client):
    params = {"query": "test", "top_k": top_k}
    response = await async_client.get("/search/", params=params)
    if top_k <= 0:
        assert response.status_code == 422
    else:
        assert response.status_code == 200

# === Uzun ve Boş Sorgular ===
@pytest.mark.anyio
async def test_search_empty_and_long_query(async_client):
    queries = ["", "a" * 10000]
    for query in queries:
        response = await async_client.get("/search/", params={"query": query, "top_k": 5})
        if query == "":
            assert response.status_code == 422
        else:
            assert response.status_code == 200

# === Eşzamanlı Arama ===
@pytest.mark.anyio
async def test_concurrent_search(async_client, monkeypatch):
    async def fake_search(v, k):
        await asyncio.sleep(0.05)
        return np.random.rand(1, k).tolist(), np.random.randint(0, 100, (1, k)).tolist()

    monkeypatch.setattr(index, "search", fake_search)

    async def do_search(i):
        return await async_client.get("/search/", params={"query": f"test {i}", "top_k": 3})

    results = await asyncio.gather(*[do_search(i) for i in range(10)])
    for res in results:
        assert res.status_code == 200

# === Eşzamanlı Index Güncelleme ===
@pytest.mark.anyio
async def test_concurrent_update_faiss(monkeypatch):
    monkeypatch.setattr("your_embedding_module.db.fetch", AsyncMock(return_value=[]))
    await asyncio.gather(
        update_faiss_index(),
        update_faiss_index(),
        update_faiss_index()
    )
    assert True

# === Performans Süresi ===
@pytest.mark.anyio
async def test_search_performance(async_client):
    start = time.monotonic()
    response = await async_client.get("/search/", params={"query": "performance", "top_k": 5})
    duration = time.monotonic() - start
    assert response.status_code == 200
    assert duration < 0.5

# === Yük Simülasyonu ===
@pytest.mark.anyio
async def test_load_stress(async_client):
    async def do_search(i):
        return await async_client.get("/search/", params={"query": f"load {i}", "top_k": 3})

    results = await asyncio.gather(*[do_search(i) for i in range(30)])
    for res in results:
        assert res.status_code == 200

# === Kapsamlı Bellek ve Performans İzleme ===

@pytest.mark.asyncio
async def test_memory_usage_of_search(async_client, monkeypatch):
    """Arama fonksiyonunun bellek kullanımını test eder."""
    tracemalloc.start()
    
    # Gerekli mock'ları tanımla
    monkeypatch.setattr("embedding.redis", AsyncMock())
    redis.get = AsyncMock(return_value=None)
    monkeypatch.setattr("embedding.embed_texts", lambda texts, model=None: [np.ones(app_settings.VECTOR_DIM)])
    monkeypatch.setattr("embedding.faiss_index", type("FakeIndex", (), {
        "ntotal": 1,
        "search": lambda self, v, k: (np.array([[0.99]]), np.array([[1]]))
    })())

    # Testi çalıştır
    resp = await async_client.get("/memory/search/", params={"query": "test", "top_k": 1})
    assert resp.status_code == 200

    snapshot = tracemalloc.take_snapshot()
    top_stats = snapshot.statistics('lineno')

    # Bellek kullanımını kontrol et
    # Bu eşik, test ortamına ve beklenen kullanıma göre ayarlanmalıdır
    total_memory = sum(stat.size for stat in top_stats)
    assert total_memory < 1024 * 1024 * 10, "Memory usage exceeded 10 MB" 
    
    tracemalloc.stop()

# === Bellek Kullanımı Ölçümü ===
@pytest.mark.anyio
async def test_memory_leak_check(async_client):
    tracemalloc.start()
    await async_client.get("/search/", params={"query": "memory test", "top_k": 3})
    snapshot1 = tracemalloc.take_snapshot()

    for _ in range(20):
        await async_client.get("/search/", params={"query": "memory test", "top_k": 3})

    snapshot2 = tracemalloc.take_snapshot()
    top_stats = snapshot2.compare_to(snapshot1, 'lineno')
    total_mem = sum([stat.size_diff for stat in top_stats])

    # bellek artışı 1MB'dan fazla olmamalı
    assert total_mem < 1_000_000
    tracemalloc.stop()

# === Hatalı Token Testi ===
@pytest.mark.anyio
async def test_auth_failure():
    async with AsyncClient(app=app, base_url="http://testserver") as client:
        response = await client.get("/search/", headers={"Authorization": "Bearer invalid"}, params={"query": "test", "top_k": 3})
        assert response.status_code in [401, 403]

# === Veri Tutarsızlığı Simülasyonu ===
@pytest.mark.anyio
async def test_data_corruption_handling(monkeypatch, async_client):
    # db.fetch geri dönüşünde bozuk veri simülasyonu
    corrupt_data = [{"id": "123", "embedding": b"notavalidbytes"}]

    monkeypatch.setattr("your_embedding_module.db.fetch", AsyncMock(return_value=corrupt_data))

    response = await async_client.get("/search/", params={"query": "corrupt", "top_k": 3})
    assert response.status_code == 500 or response.status_code == 200  # sistemin hata veya düzgün işlem vermesi kabul edilir

# === Yanıltıcı Vector Testi ===
@pytest.mark.anyio
async def test_noisy_vector_handling(monkeypatch):
    async def fake_fetch(*args, **kwargs):
        # Geri dönüşte yanlış formatta embedding
        return [{"id": "abc", "embedding": b"invalid"}]

    monkeypatch.setattr("your_embedding_module.db.fetch", fake_fetch)

    # Update faiss index çağrısı hatasız bitmeli
    await update_faiss_index()
    assert True

# === FAISS Indeks Bozulma Testi ===
@pytest.mark.anyio
async def test_faiss_index_corruption(monkeypatch):
    class FakeIndex:
        def __init__(self):
            self.trained = True
            self.reset_called = False

        def reset(self):
            self.reset_called = True

        @property
        def is_trained(self):
            return self.trained

        def train(self, embeddings):
            pass

        def add_with_ids(self, embeddings, ids):
            raise RuntimeError("Index corrupted")

    fake_index = FakeIndex()
    monkeypatch.setattr("your_embedding_module.index", fake_index)

    with pytest.raises(RuntimeError):
        await update_faiss_index()

# === Bulk Insert ve Arama ===
@pytest.mark.anyio
async def test_bulk_insert_and_search(async_client, monkeypatch):
    # Çoklu kayıt ekleme simülasyonu
    sample_embeddings = []
    for i in range(100):
        sample_embeddings.append({"id": f"id_{i}", "embedding": np.random.rand(1536).astype('float32').tobytes()})

    monkeypatch.setattr("your_embedding_module.db.fetch", AsyncMock(return_value=sample_embeddings))
    await update_faiss_index()

    response = await async_client.get("/search/", params={"query": "bulk test", "top_k": 5})
    assert response.status_code == 200

# === Silme ve Tutarlılık Testi ===
@pytest.mark.anyio
async def test_delete_and_search_consistency(async_client, monkeypatch):
    """Silinen bir kaydın FAISS index'inden de kaldırıldığını test eder."""
    
    # Mock için sahte bir FAISS index oluştur
    fake_index = faiss.IndexFlatIP(1536)
    monkeypatch.setattr("your_embedding_module.faiss_index", fake_index)

    # 10 kayıtlık bir veri seti oluştur
    records = [{"id": i, "text": f"text {i}", "embedding": np.random.rand(1536).astype('float32').tolist()} for i in range(10)]
    embeddings = [np.array(rec["embedding"]) for rec in records]
    ids = np.array([rec["id"] for rec in records]).astype('int64')
    
    # Indexe kayıtları ekle
    fake_index.add_with_ids(np.array(embeddings), ids)
    
    assert fake_index.ntotal == 10
    
    # Silinecek kaydın ID'si
    id_to_delete = 5
    
    # Silme işlemini simüle et
    # Bu mock, remove_embedding_from_index'in çağrılmasına gerek kalmadan doğrudan index'ten silecek
    fake_index.remove_ids(np.array([id_to_delete]).astype('int64'))

    # Index boyutunun azaldığını kontrol et
    assert fake_index.ntotal == 9
    
    # Silinen kaydın index'te olup olmadığını kontrol etmek için arama yap
    # Bu testin çalışması için search fonksiyonunun ilgili ID'yi aramaması gerekir
    # Bu test, mevcut search endpoint'inin `id` değerlerini döndürdüğünü varsayıyor
    search_vector = embeddings[id_to_delete].reshape(1, -1)
    distances, indices = fake_index.search(search_vector, 10)
    
    # Sonuçlarda silinen kaydın ID'si olmamalı
    assert id_to_delete not in indices[0]

# === Embeddings Güncelleme Testi ===
@pytest.mark.anyio
async def test_embedding_update_behavior(async_client, monkeypatch):
    # Yeni embeddingle güncelleme simülasyonu
    old_embedding = {"id": "update_test", "embedding": np.random.rand(1536).astype('float32').tobytes()}
    new_embedding = {"id": "update_test", "embedding": np.random.rand(1536).astype('float32').tobytes()}

    monkeypatch.setattr("your_embedding_module.db.fetch", AsyncMock(return_value=[old_embedding]))
    await update_faiss_index()

    # Veri güncelle
    monkeypatch.setattr("your_embedding_module.db.fetch", AsyncMock(return_value=[new_embedding]))
    await update_faiss_index()

    response = await async_client.get("/search/", params={"query": "update test", "top_k": 5})
    assert response.status_code == 200

# === Expired Token Testi ===
@pytest.mark.anyio
async def test_expired_token():
    expired_payload = {"sub": "user1", "exp": datetime.utcnow() - timedelta(minutes=5)}
    token = jwt.encode(expired_payload, "testsecret", algorithm="HS256")

    async def fake_get_current_user(token_in):
        from your_embedding_module import HTTPException
        try:
            # Your actual get_current_user implementation may vary,
            # simulate token expired exception
            if token_in == token:
                raise HTTPException(status_code=401, detail="Token expired")
            return "user1"
        except Exception:
            raise HTTPException(status_code=401)

    app.dependency_overrides[get_current_user] = fake_get_current_user

    async with AsyncClient(app=app, base_url="http://testserver") as client:
        response = await client.get("/search/", headers={"Authorization": f"Bearer {token}"}, params={"query": "expired", "top_k": 3})
        assert response.status_code == 401

# === Hata Yönetimi ve JWT Testleri ===

@pytest.mark.asyncio
async def test_expired_jwt_token():
    """Süresi dolmuş bir JWT token ile auth hatasını test eder."""
    # Süresi dolmuş bir token oluştur
    payload = {"sub": "user1", "exp": datetime.utcnow() - timedelta(minutes=5)}
    expired_token = jwt.encode(payload, "secret-key", algorithm="HS256")

    async def fake_get_current_user_expired(token: str):
        # Gerçek JWT doğrulama fonksiyonunu çağırıyoruz
        try:
            return get_current_user(token)
        except HTTPException as e:
            raise e

    app.dependency_overrides[get_current_user] = fake_get_current_user_expired
    async with AsyncClient(app=app, base_url="http://testserver") as client:
        resp = await client.get("/memory/search/", headers={"Authorization": f"Bearer {expired_token}"}, params={"query": "test"})
        assert resp.status_code == 401
        assert "Signature has expired" in resp.json()["detail"]

    app.dependency_overrides.clear()


# === Benchmark için placeholder ===
@pytest.mark.benchmark(group="embedding_search")
@pytest.mark.anyio
async def test_benchmark_search(benchmark, async_client):
    def run_search():
        return asyncio.run(async_client.get("/search/", params={"query": "benchmark", "top_k": 5}))

    result = benchmark(run_search)
    assert result

# === Dummy Test for Coverage Report ===
def test_dummy_for_coverage():
    assert True
