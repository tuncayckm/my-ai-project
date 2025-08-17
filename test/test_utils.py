# test_utils.py

import pytest
import numpy as np
import json

from .utils import serialize_embedding, deserialize_embedding

# ---------------- Pozitif Senaryolar ---------------- #

@pytest.mark.parametrize(
    "data",
    [
        np.array([0.1, -0.5, 1.2]),         # tek boyutlu numpy array
        [0.1, -0.5, 1.2],                   # liste
        np.array([]),                        # boş array
        np.array([[0.1, 0.2], [0.3, 0.4]]), # çok boyutlu array
        ["a", "b", "c"],                     # sayısal olmayan değerler
    ]
)
def test_serialize_and_deserialize_valid_data(data):
    """Farklı tiplerdeki embedding'lerin serialize/deserialize testi."""
    encoded = serialize_embedding(data)
    decoded = deserialize_embedding(encoded)

    assert isinstance(encoded, bytes)
    assert isinstance(decoded, np.ndarray)
    assert decoded.shape == np.array(data).shape
    assert np.array_equal(decoded, np.array(data))


# ---------------- Negatif Senaryolar ---------------- #

@pytest.mark.parametrize(
    "bad_data, expected_exception",
    [
        (b"not-a-json", json.JSONDecodeError),  # geçersiz JSON
        (b"{]", json.JSONDecodeError),          # bozuk JSON
    ]
)
def test_deserialize_with_invalid_bytes(bad_data, expected_exception):
    """Geçersiz JSON formatındaki verilerde hata fırlatılmalı."""
    with pytest.raises(expected_exception):
        deserialize_embedding(bad_data)


# ---------------- Ek Edge Case ve Uyumsuz Tip Testleri ---------------- #

@pytest.mark.parametrize(
    "data, should_raise",
    [
        (None, False),                      # None → numpy array [None]
        ({"a": 1}, True),                   # dict → serialize hata verir
        ("string", False),                   # string → array oluşturulabilir
        (np.random.rand(1000000), False),   # çok büyük array
        (np.array([np.inf, -np.inf, np.nan, 1e-308, 1e308]), False),  # edge float değerleri
    ]
)
def test_serialize_embedding_edge_cases(data, should_raise):
    """Uyumsuz tipler, büyük array ve edge float değerlerinin test edilmesi."""
    if should_raise:
        with pytest.raises(TypeError):
            serialize_embedding(data)
    else:
        encoded = serialize_embedding(data)
        decoded = deserialize_embedding(encoded)
        # shape ve içerik kontrolü
        if isinstance(data, np.ndarray):
            assert decoded.shape == data.shape
            assert np.allclose(decoded, data, equal_nan=True)
        elif data is None:
            assert np.array_equal(decoded, np.array([None]))
        elif isinstance(data, str):
            assert np.array_equal(decoded, np.array([data]))

# ---------------- Memory / Stres Testleri ---------------- #

@pytest.mark.parametrize(
    "size",
    [
        10**6,     # 1 milyon eleman
        10**7,     # 10 milyon eleman
    ]
)
def test_serialize_large_arrays(size):
    """Çok büyük arraylerin serialize/deserialize performans ve memory testi."""
    large_array = np.random.rand(size)
    encoded = serialize_embedding(large_array)
    decoded = deserialize_embedding(encoded)

    assert isinstance(encoded, bytes)
    assert isinstance(decoded, np.ndarray)
    assert decoded.shape == large_array.shape
    # float değerlerini kontrol et (küçük tolerans ile)
    assert np.allclose(decoded, large_array, rtol=1e-7, atol=0)


# ---------------- Ekstrem Edge Float ve Memory Limit Testleri ---------------- #

@pytest.mark.parametrize(
    "array",
    [
        np.array([np.nan, np.inf, -np.inf, 0.0, -0.0, 1e-308, 1e308]),  # ekstrem float değerleri
        np.random.rand(10**5) * 1e308,  # çok büyük float değerleri ile büyük array
        np.random.rand(10**5) * 1e-308, # çok küçük float değerleri ile büyük array
    ]
)
def test_extreme_float_and_large_arrays(array):
    """
    Ekstrem float değerler ve büyük arrayler üzerinde serialize/deserialize testi.
    Memory limit ve doğruluk kontrolü yapılır.
    """
    encoded = serialize_embedding(array)
    decoded = deserialize_embedding(encoded)

    assert isinstance(encoded, bytes)
    assert isinstance(decoded, np.ndarray)
    assert decoded.shape == array.shape
    # NaN ve inf değerlerini de toleranslı kontrol et
    np.testing.assert_array_equal(np.isnan(decoded), np.isnan(array))
    np.testing.assert_array_equal(np.isinf(decoded), np.isinf(array))
    # Finite değerler için yakınlık kontrolü
    finite_mask = np.isfinite(array)
    assert np.allclose(decoded[finite_mask], array[finite_mask], rtol=1e-7, atol=0)


# ---------------- Modül Bazlı Coverage ---------------- #
# Placeholder test. Pytest-cov ile utils modülünün tamamı kapsanabilir.
def test_module_coverage_placeholder():
    """
    Placeholder test. Pytest-cov ile utils modülünün tamamı kapsanabilir.
    Coverage raporu için pytest komutu kullanılacak.
    """
    assert True
