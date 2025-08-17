# utils.py

import numpy as np
import json

def serialize_embedding(embedding: np.ndarray) -> bytes:
    """Numpy embedding dizisini JSON formatına serileştirir."""
    if not isinstance(embedding, np.ndarray):
        embedding = np.array(embedding)
    # Numpy dizisini Python listesine çevirip JSON'a dönüştür ve UTF-8 olarak kodla
    return json.dumps(embedding.tolist()).encode('utf-8')

def deserialize_embedding(serialized_embedding: bytes) -> np.ndarray:
    """JSON formatındaki embedding'i numpy dizisine dönüştürür."""
    # UTF-8 byte dizisini çöz, JSON'dan Python listesine çevir ve numpy dizisi oluştur
    return np.array(json.loads(serialized_embedding.decode('utf-8')))