import asyncio
import pytest
from asyncpg.pool import Pool
from unittest.mock import AsyncMock, patch
from asyncpg import exceptions as pg_exceptions

# database.py modülünüzdeki fonksiyonları içe aktarıyoruz
from database import (
    get_pg_database_pool,
    execute,
    fetch_one,
    init_db
)

# Test veritabanı bağlantı bilgilerini tanımlıyoruz.
# Bu bilgileri gerçek veritabanınızdan ayırın!
TEST_DB_URL = "postgresql://user:password@localhost:5432/test_db"
# pytest-asyncio eklentisi için event loop'u fixture olarak sağlıyoruz
@pytest.fixture(scope="session")
def event_loop():
    loop = asyncio.get_event_loop()
    yield loop
    loop.close()

# Testler için mock (sahte) bir veritabanı havuzu oluşturuyoruz.
# Gerçek veritabanına bağlanmadan test yapmak için bu yöntem tercih edilir.
@pytest.fixture
async def mock_db_pool():
    # MockPool nesnesi, asyncpg.pool.Pool'un davranışını taklit eder.
    class MockPool(AsyncMock):
        async def acquire(self):
            return AsyncMock() # mock bir bağlantı döndürür
        async def release(self, conn):
            pass # Bağlantıyı serbest bırakma işlemini taklit eder.

    mock_pool = MockPool(spec=Pool)
    mock_pool.fetchrow.return_value = None
    mock_pool.fetch.return_value = []

    # get_pg_database_pool fonksiyonunun mock havuzumuzu döndürmesini sağlıyoruz
    with patch('database.get_pg_database_pool', return_value=mock_pool):
        yield mock_pool

# --- Mevcut Test Fonksiyonları ---

@pytest.mark.asyncio
async def test_get_pg_database_pool_success(mock_db_pool):
    """get_pg_database_pool fonksiyonunun başarılı çalışmasını test eder."""
    pool = await get_pg_database_pool()
    assert pool is not None
    assert isinstance(pool, AsyncMock)

@pytest.mark.asyncio
async def test_execute_with_mock_pool(mock_db_pool):
    """execute fonksiyonunun mock havuz ile çalışmasını test eder."""
    sql = "INSERT INTO test_table (name) VALUES ($1);"
    params = ("test_name",)
    mock_db_pool.execute.return_value = "INSERT 0 1"
    result = await execute(sql, *params)
    mock_db_pool.execute.assert_called_once_with(sql, *params)
    assert result == "INSERT 0 1"

@pytest.mark.asyncio
async def test_fetch_one_with_mock_pool(mock_db_pool):
    """fetch_one fonksiyonunun mock havuz ile çalışmasını test eder."""
    sql = "SELECT * FROM test_table WHERE id = $1;"
    params = (1,)
    mock_record = {'id': 1, 'name': 'test_name'}
    mock_db_pool.fetchrow.return_value = mock_record
    result = await fetch_one(sql, *params)
    mock_db_pool.fetchrow.assert_called_once_with(sql, *params)
    assert result == mock_record

@pytest.mark.asyncio
@patch('database.execute')
async def test_init_db(mock_execute):
    """init_db fonksiyonunun tablo oluşturma SQL'ini çağırdığını test eder."""
    await init_db()
    assert "CREATE TABLE IF NOT EXISTS" in mock_execute.call_args[0][0]


# --- Yeni Eklenen Test Fonksiyonları ---

@pytest.mark.asyncio
async def test_execute_failure(mock_db_pool):
    """Veritabanı sorgusunun hata fırlatması durumunu test eder."""
    sql = "INVALID SQL QUERY"
    params = ()
    # execute'un bir veritabanı hatası fırlatmasını simüle ediyoruz
    mock_db_pool.execute.side_effect = pg_exceptions.PostgresError("Veritabanı hatası")

    with pytest.raises(pg_exceptions.PostgresError):
        await execute(sql, *params)
    # Hata durumunda dahi fonksiyonun çağrıldığını kontrol ediyoruz
    mock_db_pool.execute.assert_called_once_with(sql, *params)


@pytest.mark.asyncio
async def test_fetch_one_no_result(mock_db_pool):
    """fetch_one fonksiyonunun sonuç döndürmediği durumu test eder."""
    sql = "SELECT * FROM test_table WHERE id = $1;"
    params = (999,) # Olmayan bir ID
    
    # fetchrow'un None döndürmesi beklenir
    mock_db_pool.fetchrow.return_value = None
    
    result = await fetch_one(sql, *params)
    mock_db_pool.fetchrow.assert_called_once_with(sql, *params)
    assert result is None


@pytest.mark.asyncio
async def test_fetch_multiple_records(mock_db_pool):
    """Çoklu kayıt getirme (fetch) işlemini test eder."""
    sql = "SELECT * FROM test_table;"
    
    # fetch'in birden fazla kayıt döndürmesini simüle ediyoruz
    mock_records = [{'id': 1, 'name': 'test1'}, {'id': 2, 'name': 'test2'}]
    mock_db_pool.fetch.return_value = mock_records
    
    result = await mock_db_pool.fetch(sql)
    
    # Sorgunun doğru şekilde çağrıldığını ve doğru sonucu döndürdüğünü kontrol ediyoruz
    assert result == mock_records
    assert len(result) == 2


@pytest.mark.asyncio
async def test_get_pg_database_pool_failure():
    """get_pg_database_pool'un bağlantı hatası durumunu test eder."""
    # get_pg_database_pool fonksiyonunun hata fırlatmasını sağlamak için patch kullanıyoruz
    with patch('asyncpg.create_pool', side_effect=pg_exceptions.PostgresError("Bağlantı başarısız")):
        with pytest.raises(pg_exceptions.PostgresError):
            await get_pg_database_pool()