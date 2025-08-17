import asyncio
import logging
from prometheus_client import start_http_server, Counter, Histogram
from your_project.learning import LearningService
from config import app_settings
from database import DatabaseManager
from dataset import prepare_training_data
from fine_tuning import prepare_fine_tuning, train_model_lora
from nlp import apply_knowledge
from rate_limiter import SimpleRateLimiter

# Logger ayarları
logging.basicConfig(level=app_settings.log_level, format='%(asctime)s %(levelname)s: %(message)s')

# Prometheus metrikleri
REQUEST_COUNT = Counter('app_request_count', 'Toplam istek sayısı', ['endpoint', 'method', 'status'])
REQUEST_LATENCY = Histogram('app_request_latency_seconds', 'İstek gecikme süresi', ['endpoint', 'method'])

# Basit rate limiter - örnek olarak kullanıcı bazlı
rate_limiter = SimpleRateLimiter(max_calls=10, period_seconds=60)

# Audit logger (ayrı dosyaya loglama)
audit_logger = logging.getLogger("audit")
audit_logger.setLevel(logging.INFO)
fh = logging.FileHandler("audit.log")
audit_logger.addHandler(fh)

async def main():
    start_http_server(8000)  # Prometheus endpoint

    db_manager = None
    try:
        logging.info("Veritabanı yöneticisi başlatılıyor...")
        db_manager = DatabaseManager(dsn=app_settings.database_dsn)
        await db_manager.connect()

        user_id = "user_123"
        token = "your.jwt.token.here"  # Gerçek token buraya gelecek

        # Profil kontrolü
        profile = await db_manager.load_profile(user_id)
        if not profile:
            logging.info(f"{user_id} için yeni profil oluşturuluyor.")
            await db_manager.save_profile(
                user_id, "Test User", {"language": "tr"}, [{"role": "system", "content": "Merhaba!"}]
            )

        # Rate limiting kontrolü
        if not rate_limiter.is_allowed(user_id):
            logging.warning(f"Rate limit aşıldı: {user_id}")
            return

        # Prometheus metrik ölçümü
        REQUEST_COUNT.labels(endpoint="/apply_knowledge", method="POST", status="started").inc()
        with REQUEST_LATENCY.labels(endpoint="/apply_knowledge", method="POST").time():
            # Audit log
            audit_logger.info(f"user_id={user_id} prompt='Bana yapay zeka hakkında bir şey söyle.' token_valid=True")

            # NLP cevabı al
            response_messages = await apply_knowledge(
                db_manager=db_manager,
                user_id=user_id,
                prompt="Bana yapay zeka hakkında bir şey söyle.",
                token=token,
                plugin_data=None,
                max_context_tokens=512
            )
        REQUEST_COUNT.labels(endpoint="/apply_knowledge", method="POST", status="success").inc()

        logging.info(f"NLP cevabı (JSON formatında): {response_messages}")

        # Fine-tuning verilerini hazırla
        logging.info("Fine-tuning için veriler hazırlanıyor...")
        training_samples = await prepare_training_data(db_manager)
        logging.info(f"Eğitim için {len(training_samples)} örnek bulundu.")

    except Exception as e:
        logging.critical(f"Ana uygulama akışında kritik hata: {e}", exc_info=True)
        REQUEST_COUNT.labels(endpoint="/apply_knowledge", method="POST", status="error").inc()
    finally:
        if db_manager:
            await db_manager.close()
        logging.info("Uygulama sonlandı.")


if __name__ == "__main__":
    asyncio.run(main())
