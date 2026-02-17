import redis

def mark_processed(r: redis.Redis, target_model: str, message_id: str) -> bool:
    """
    Marca un mensaje como procesado para idempotencia.
    Retorna True si no se había procesado antes (puede procesarse),
    False si ya estaba marcado.
    """
    key = f"processed:{target_model}:{message_id}"
    return r.set(key, "1", nx=True) is True
