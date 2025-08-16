import pytest
import asyncio
import json
from unittest.mock import AsyncMock, patch, mock_open
from pathlib import Path
import contextlib
from dp import AsyncDatabaseManager
# utils modülünü içeri aktar
from . import utils  # veya projenizin yapısına göre "import utils"
from aiosqlite import OperationalError, DatabaseError, ProgrammingError

# ---------------- Fixtures ---------------- #

@pytest.fixture(scope="session")
def event_loop():
    """pytest-asyncio için event loop sağlar."""
    loop = asyncio.get_event_loop()
    yield loop
    loop.close()

@pytest.fixture
def mock_db_path():
    return "mock_db.sqlite"

@pytest.fixture
def mock_db_connection():
    """Sahte aiosqlite bağlantısı"""
    mock_conn = AsyncMock()
    mock_conn.commit = AsyncMock()
    mock_conn.rollback = AsyncMock()
    mock_conn.close = AsyncMock()
    mock_conn.execute = AsyncMock()
    mock_conn.fetchall = AsyncMock()
    mock_conn.fetchone = AsyncMock()
    return mock_conn

@pytest.fixture
def mock_db_pool(mock_db_connection):
    """aiosqlite.connect mock’lanmış bağlantı döndürür"""
    @contextlib.asynccontextmanager
    async def mock_connect_context_manager(*args, **kwargs):
        yield mock_db_connection

    with patch('aiosqlite.connect', new=mock_connect_context_manager):
        yield mock_db_connection


# ---------------- AsyncDatabaseManager Testleri ---------------- #

@pytest.mark.asyncio
async def test_save_profile_success(mock_db_pool):
    db = AsyncDatabaseManager("mock.sqlite")
    db._get_connection = AsyncMock(return_value=mock_db_pool)

    await db.save_profile("uid", "User", {"lang": "en"})
    mock_db_pool.execute.assert_called_once()
    sql, params = mock_db_pool.execute.call_args[0]
    assert "INSERT OR REPLACE INTO user_profiles" in sql
    assert params[0] == "uid"

@pytest.mark.asyncio
async def test_save_profile_invalid_preferences(mock_db_pool):
    db = AsyncDatabaseManager("mock.sqlite")
    with pytest.raises(ValueError):
        await db.save_profile("uid", "User", "not-a-dict")

@pytest.mark.asyncio
async def test_load_profile_found(mock_db_pool):
    mock_db_pool.execute.return_value.__aenter__.return_value.fetchone = AsyncMock(
        return_value={"user_id": "uid", "name": "User", "preferences": json.dumps({"lang": "en"})}
    )
    db = AsyncDatabaseManager("mock.sqlite")
    db._get_connection = AsyncMock(return_value=mock_db_pool)
    profile = await db.load_profile("uid")
    assert profile["user_id"] == "uid"
    assert profile["preferences"]["lang"] == "en"

@pytest.mark.asyncio
async def test_load_profile_not_found(mock_db_pool):
    mock_db_pool.execute.return_value.__aenter__.return_value.fetchone = AsyncMock(return_value=None)
    db = AsyncDatabaseManager("mock.sqlite")
    db._get_connection = AsyncMock(return_value=mock_db_pool)
    result = await db.load_profile("unknown")
    assert result is None

@pytest.mark.asyncio
async def test_update_profile_preferences_success(mock_db_pool):
    db = AsyncDatabaseManager("mock.sqlite")
    db._get_connection = AsyncMock(return_value=mock_db_pool)
    await db.update_profile_preferences("uid", {"lang": "tr"})
    mock_db_pool.execute.assert_called_once()

@pytest.mark.asyncio
async def test_update_profile_preferences_invalid(mock_db_pool):
    db = AsyncDatabaseManager("mock.sqlite")
    with pytest.raises(ValueError):
        await db.update_profile_preferences("uid", "not-a-dict")

@pytest.mark.asyncio
async def test_add_to_history_success(mock_db_pool):
    db = AsyncDatabaseManager("mock.sqlite")
    db._get_connection = AsyncMock(return_value=mock_db_pool)
    await db.add_to_history("uid", "prompt", "response")
    mock_db_pool.execute.assert_called_once()
    sql, params = mock_db_pool.execute.call_args[0]
    assert "INSERT INTO user_history" in sql
    assert params[0] == "uid"

@pytest.mark.asyncio
async def test_get_history_basic(mock_db_pool):
    mock_db_pool.execute.return_value.__aenter__.return_value.__aiter__.return_value = [
        {"timestamp": "t", "prompt": "p", "response": "r"}
    ]
    db = AsyncDatabaseManager("mock.sqlite")
    db._get_connection = AsyncMock(return_value=mock_db_pool)
    history = await db.get_history("uid")
    assert len(history) == 1
    assert history[0]["prompt"] == "p"

@pytest.mark.asyncio
async def test_get_history_with_dates(mock_db_pool):
    mock_db_pool.execute.return_value.__aenter__.return_value.__aiter__.return_value = []
    db = AsyncDatabaseManager("mock.sqlite")
    db._get_connection = AsyncMock(return_value=mock_db_pool)
    await db.get_history("uid", limit=5, start_time="2024-01-01", end_time="2024-01-31")
    mock_db_pool.execute.assert_called_once()
    sql, params = mock_db_pool.execute.call_args[0]
    assert "timestamp BETWEEN" in sql

@pytest.mark.asyncio
async def test_get_history_only_start_time(mock_db_pool):
    mock_db_pool.execute.return_value.__aenter__.return_value.__aiter__.return_value = []
    db = AsyncDatabaseManager("mock.sqlite")
    db._get_connection = AsyncMock(return_value=mock_db_pool)

    await db.get_history("uid", start_time="2024-01-01")
    sql, _ = mock_db_pool.execute.call_args[0]
    assert "timestamp >= ?" in sql

@pytest.mark.asyncio
async def test_get_history_only_end_time(mock_db_pool):
    mock_db_pool.execute.return_value.__aenter__.return_value.__aiter__.return_value = []
    db = AsyncDatabaseManager("mock.sqlite")
    db._get_connection = AsyncMock(return_value=mock_db_pool)

    await db.get_history("uid", end_time="2024-01-31")
    sql, _ = mock_db_pool.execute.call_args[0]
    assert "timestamp <= ?" in sql

@pytest.mark.asyncio
async def test_save_memory_success(mock_db_pool):
    db = AsyncDatabaseManager("mock.sqlite")
    db._get_connection = AsyncMock(return_value=mock_db_pool)
    await db.save_memory("mem1", "uid", "text", [0.1, 0.2])
    mock_db_pool.execute.assert_called_once()
    sql, params = mock_db_pool.execute.call_args[0]
    assert "INSERT OR REPLACE INTO memory_store" in sql
    assert params[0] == "mem1"

@pytest.mark.asyncio
async def test_save_memory_invalid_embedding(mock_db_pool):
    db = AsyncDatabaseManager("mock.sqlite")
    with pytest.raises(ValueError):
        await db.save_memory("mem1", "uid", "text", "not-a-list")

@pytest.mark.asyncio
async def test_load_memories_with_user_id(mock_db_pool):
    mock_db_pool.execute.return_value.__aenter__.return_value.__aiter__.return_value = [
        {"id": "m1", "user_id": "uid", "text": "t", "embedding": utils.serialize_embedding([0.1])}
    ]
    db = AsyncDatabaseManager("mock.sqlite")
    db._get_connection = AsyncMock(return_value=mock_db_pool)
    result = await db.load_memories("uid")
    assert result[0]["user_id"] == "uid"

@pytest.mark.asyncio
async def test_load_memories_all(mock_db_pool):
    mock_db_pool.execute.return_value.__aenter__.return_value.__aiter__.return_value = []
    db = AsyncDatabaseManager("mock.sqlite")
    db._get_connection = AsyncMock(return_value=mock_db_pool)
    result = await db.load_memories()
    assert isinstance(result, list)

@pytest.mark.asyncio
async def test_delete_profile_success_with_mocker(mock_db_pool, mocker):
    db = AsyncDatabaseManager("mock.sqlite")
    
    # 'transaction' metodunu mock'lama
    mock_transaction = mocker.AsyncMock()
    # Mock'lanmış transaction, bir async context manager döndürmeli
    mock_transaction.__aenter__.return_value = mock_db_pool
    mocker.patch.object(db, 'transaction', return_value=mock_transaction)
    
    await db.delete_profile("uid")
    
    # execute metodunun beklenen şekilde çağrıldığını kontrol et
    assert mock_db_pool.execute.call_count == 3
    # İlk çağrı user_profiles tablosu için, ikinci user_history, üçüncü memory_store için
    assert "DELETE FROM user_profiles" in mock_db_pool.execute.call_args_list[0][0][0]
    assert "DELETE FROM user_history" in mock_db_pool.execute.call_args_list[1][0][0]
    assert "DELETE FROM memory_store" in mock_db_pool.execute.call_args_list[2][0][0]

@pytest.mark.asyncio
async def test_delete_memory_success(mock_db_pool):
    db = AsyncDatabaseManager("mock.sqlite")
    db._get_connection = AsyncMock(return_value=mock_db_pool)
    await db.delete_memory("mem1")
    mock_db_pool.execute.assert_called_once()
    sql, params = mock_db_pool.execute.call_args[0]
    assert "DELETE FROM memory_store" in sql
    assert params[0] == "mem1"

@pytest.mark.asyncio
async def test_run_migrations_no_update_needed_improved(mock_db_pool):
    db = AsyncDatabaseManager("mock.sqlite")
    # 'transaction' yerine 'aiosqlite.connect' mock'lama
    
    # Mevcut versiyon zaten güncel (schema_version = 2)
    mock_cursor = AsyncMock()
    mock_cursor.fetchone.return_value = {"version": 2}
    mock_db_pool.execute.return_value.__aenter__.return_value = mock_cursor

    with patch.object(db, '_get_connection', return_value=mock_db_pool):
        await db._run_migrations()

    # assert mock_db_pool.execute.call_count == 2
    # Çağrıları daha spesifik kontrol edelim
    calls = [call[0][0] for call in mock_db_pool.execute.call_args_list]
    assert "CREATE TABLE IF NOT EXISTS schema_version" in calls[0]
    assert "SELECT version FROM schema_version" in calls[1]
    # Migration çağrılmadığı için INSERT INTO schema_version'ın olmaması gerekir
    assert "INSERT INTO schema_version" not in calls[2:]

@pytest.mark.asyncio
async def test_migrate_success(mock_db_pool):
    db = AsyncDatabaseManager("mock.sqlite")
    # old_version < 1 durumu simülasyonu
    await db._migrate(mock_db_pool, 0, 2)
    # memory_store ve user_profiles tabloları oluşturulmuş olmalı
    executed_sql = " ".join(call[0][0] for call in mock_db_pool.execute.call_args_list)
    assert "CREATE TABLE IF NOT EXISTS memory_store" in executed_sql
    assert "CREATE TABLE IF NOT EXISTS user_profiles" in executed_sql

@pytest.mark.asyncio
async def test_migrate_raises_error(mock_db_pool):
    db = AsyncDatabaseManager("mock.sqlite")
    mock_db_pool.execute.side_effect = Exception("SQL Error")

    with pytest.raises(Exception) as exc_info:
        await db._migrate(mock_db_pool, 0, 2)
    assert "SQL Error" in str(exc_info.value)

@pytest.mark.asyncio
async def test_transaction_commit_and_rollback(mock_db_pool):
    db = AsyncDatabaseManager("mock.sqlite")
    db._get_connection = AsyncMock(return_value=mock_db_pool)

    # Başarılı commit
    async with db.transaction() as conn:
        await conn.execute("OK")
    mock_db_pool.commit.assert_called_once()

    # Rollback senaryosu
    mock_db_pool.commit.reset_mock()
    mock_db_pool.rollback.reset_mock()
    with pytest.raises(Exception):
        async with db.transaction() as conn:
            await conn.execute("FAIL")
            raise Exception("test")
    mock_db_pool.rollback.assert_called_once()

@pytest.mark.asyncio
async def test_async_retry_on_connection_failure():
    attempts = 0
    async def fail_then_succeed(*args, **kwargs):
        nonlocal attempts
        attempts += 1
        if attempts < 3:
            raise OperationalError("fail")
        return AsyncMock()

    with patch("aiosqlite.connect", side_effect=fail_then_succeed):
        db = AsyncDatabaseManager("mock.sqlite")
        await db._create_connection()
    assert attempts == 3
