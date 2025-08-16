DP: Asenkron Veritabanı Yöneticisi
Bu proje, yapay zeka uygulamaları için özel olarak tasarlanmış, asenkron bir SQLite veritabanı yönetim modülüdür. Kullanıcı profillerini, etkileşim geçmişini ve vektör tabanlı hafıza kayıtlarını yüksek performans ve güvenilirlikle yönetmek için geliştirilmiştir.

Özellikler
Asenkron Çalışma: asyncio kullanarak I/O yoğun veritabanı işlemlerini eş zamanlı ve verimli bir şekilde yönetir.

Güvenilirlik: Üstel geri çekilme (exponential backoff) stratejisiyle çalışan async_retry dekoratörü sayesinde, geçici veritabanı hatalarına karşı dayanıklıdır.

Veri Bütünlüğü: asynccontextmanager ile tasarlanmış transaction yöneticisi, çoklu veritabanı işlemlerinin atomik olarak tamamlanmasını veya hatada tamamen geri alınmasını sağlar.

Vektör Desteği: JSON serileştirme ve FAISS entegrasyonu sayesinde vektör gömme (embedding) verilerini verimli bir şekilde saklar ve sorgular.

Kapsamlı Test Kapsamı: pytest, pytest-asyncio, hypothesis ve unittest.mock gibi modern araçlarla oluşturulmuş, yüksek kaliteli bir test paketi ile desteklenir.

Kurulum
Projenin bağımlılıklarını kurmak için pip kullanabilirsiniz.

pip install aiosqlite numpy redis faiss-cpu pytest pytest-asyncio hypothesis tox

Kullanım
Aşağıdaki örnek, AsyncDatabaseManager sınıfının temel kullanımını göstermektedir.

import asyncio
from dp import AsyncDatabaseManager

async def main():
    # Asenkron veritabanı yöneticisi oluşturun.
    # `async with` ifadesi, bağlantı havuzunun otomatik olarak yönetilmesini sağlar.
    async with AsyncDatabaseManager("my_ai_db.sqlite") as db:
        user_id = "user-123"

        # 1. Kullanıcı Profili Kaydetme
        await db.save_profile(user_id, "Alice", {"theme": "light", "language": "en"})
        profile = await db.load_profile(user_id)
        print(f"Loaded Profile: {profile}")

        # 2. Etkileşim Geçmişine Ekleme
        await db.add_to_history(user_id, "Can you help me?", "Yes, I can help.")
        history = await db.get_history(user_id)
        print(f"User History: {history}")

        # 3. Hafıza Vektörü Kaydetme (Embedding)
        embedding_data = [0.12, 0.34, 0.56, 0.78, 0.90] # Örnek embedding
        await db.save_memory("mem-001", user_id, "I like hiking", embedding_data)
        memories = await db.load_memories(user_id)
        print(f"User Memories: {memories}")

if __name__ == "__main__":
    asyncio.run(main())

Testler
Testler, projenin tests/ klasöründe bulunmaktadır. Testleri çalıştırmak için tox veya doğrudan pytest kullanabilirsiniz.

# Tox kullanarak testleri çalıştırma (tüm bağımlılıkları otomatik kurar)
tox

# Alternatif olarak pytest ile doğrudan çalıştırma
pytest --asyncio-mode=auto

Katkıda Bulunma
Geri bildirimler ve katkılarınız her zaman açığız! Lütfen bir pull request (PR) oluşturmaktan veya bir issue açmaktan çekinmeyin.

Lisans
Bu proje MIT Lisansı altında yayınlanmıştır. Daha fazla bilgi için [LİSANS_DOSYASI] inceleyebilirsiniz.