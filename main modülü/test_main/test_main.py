"""
test_main.py
Main modülü için ileri seviye modül-bazlı unit testler
"""

import asyncio
import logging
import types
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

import main


@pytest.mark.asyncio
async def test_main_success_flow(monkeypatch, caplog):
    """✅ Tüm akışın başarıyla tamamlandığı senaryo"""
    caplog.set_level(logging.INFO)

    # Mock bağımlılıklar
    mock_db = AsyncMock()
    mock_db.connect.return_value = None
    mock_db.load_profile.return_value = {"id": "user_123"}
    mock_db.close.return_value = None
    monkeypatch.setattr(main, "DatabaseManager", lambda dsn: mock_db)

    mock_rate = MagicMock()
    mock_rate.is_allowed.return_value = True
    monkeypatch.setattr(main, "rate_limiter", mock_rate)

    mock_apply = AsyncMock(return_value=[{"role": "assistant", "content": "Merhaba"}])
    monkeypatch.setattr(main, "apply_knowledge", mock_apply)

    monkeypatch.setattr(main, "prepare_training_data", AsyncMock(return_value=[{"sample": 1}]))
    monkeypatch.setattr(main, "start_http_server", lambda port: None)

    await main.main()

    assert "Veritabanı yöneticisi başlatılıyor" in caplog.text
    mock_db.connect.assert_awaited_once()
    mock_apply.assert_awaited_once()
    # finally bloğunun çalışmasını doğrula
    mock_db.close.assert_awaited_once()
    assert "Eğitim için" in caplog.text


@pytest.mark.asyncio
async def test_main_creates_new_profile(monkeypatch, caplog):
    """🆕 Profil bulunmadığında yeni profil oluşturma testi"""
    caplog.set_level(logging.INFO)

    mock_db = AsyncMock()
    mock_db.connect.return_value = None
    mock_db.load_profile.return_value = None
    mock_db.save_profile.return_value = None
    mock_db.close.return_value = None
    monkeypatch.setattr(main, "DatabaseManager", lambda dsn: mock_db)

    monkeypatch.setattr(main, "rate_limiter", MagicMock(is_allowed=lambda _: True))
    monkeypatch.setattr(main, "apply_knowledge", AsyncMock(return_value=[]))
    monkeypatch.setattr(main, "prepare_training_data", AsyncMock(return_value=[]))
    monkeypatch.setattr(main, "start_http_server", lambda port: None)

    await main.main()

    mock_db.save_profile.assert_awaited_once()
    assert "yeni profil oluşturuluyor" in caplog.text


@pytest.mark.asyncio
async def test_main_rate_limit_exceeded(monkeypatch, caplog):
    """🚫 Rate limit aşıldığında NLP çağrısı yapılmamalı"""
    caplog.set_level(logging.WARNING)

    mock_db = AsyncMock()
    mock_db.connect.return_value = None
    mock_db.load_profile.return_value = {}
    mock_db.close.return_value = None
    monkeypatch.setattr(main, "DatabaseManager", lambda dsn: mock_db)

    mock_rate = MagicMock()
    mock_rate.is_allowed.return_value = False
    monkeypatch.setattr(main, "rate_limiter", mock_rate)

    mock_apply = AsyncMock()
    monkeypatch.setattr(main, "apply_knowledge", mock_apply)

    monkeypatch.setattr(main, "prepare_training_data", AsyncMock())
    monkeypatch.setattr(main, "start_http_server", lambda port: None)

    await main.main()

    mock_apply.assert_not_called()
    assert "Rate limit aşıldı" in caplog.text


@pytest.mark.asyncio
async def test_main_handles_exception(monkeypatch, caplog):
    """💥 Akış içinde istisna yakalanmalı ve metrikler güncellenmeli"""
    caplog.set_level(logging.CRITICAL)

    mock_db = AsyncMock()
    mock_db.connect.side_effect = RuntimeError("DB error")
    monkeypatch.setattr(main, "DatabaseManager", lambda dsn: mock_db)

    monkeypatch.setattr(main, "rate_limiter", MagicMock(is_allowed=lambda _: True))
    monkeypatch.setattr(main, "apply_knowledge", AsyncMock())
    monkeypatch.setattr(main, "prepare_training_data", AsyncMock())
    monkeypatch.setattr(main, "start_http_server", lambda port: None)

    mock_counter = MagicMock()
    mock_counter.labels.return_value = mock_counter
    monkeypatch.setattr(main, "REQUEST_COUNT", mock_counter)
    monkeypatch.setattr(main, "REQUEST_LATENCY", MagicMock())

    await main.main()

    assert "kritik hata" in caplog.text.lower()
    mock_counter.labels.assert_any_call(endpoint="/apply_knowledge", method="POST", status="error")
    # Hata durumunda dahi finally bloğunun çalıştığını doğrula
    mock_db.close.assert_awaited_once()


@pytest.mark.asyncio
async def test_main_prometheus_metrics_and_audit_log(monkeypatch, tmp_path):
    """📊 Prometheus metrik ve audit log dosyası yazma testi"""
    audit_file = tmp_path / "audit.log"

    mock_db = AsyncMock()
    mock_db.connect.return_value = None
    mock_db.load_profile.return_value = {}
    mock_db.save_profile.return_value = None
    mock_db.close.return_value = None
    monkeypatch.setattr(main, "DatabaseManager", lambda dsn: mock_db)

    monkeypatch.setattr(main, "rate_limiter", MagicMock(is_allowed=lambda _: True))
    monkeypatch.setattr(main, "apply_knowledge", AsyncMock(return_value=[]))
    monkeypatch.setattr(main, "prepare_training_data", AsyncMock(return_value=[]))
    monkeypatch.setattr(main, "start_http_server", lambda port: None)

    # Audit logger'ı test için yeniden yapılandır
    main.audit_logger.handlers.clear()
    handler = logging.FileHandler(audit_file)
    main.audit_logger.addHandler(handler)

    await main.main()

    handler.close()
    assert audit_file.exists()
    content = audit_file.read_text()
    assert "user_id=" in content
