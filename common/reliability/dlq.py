import logging
import redis

def send_to_dlq(r: redis.Redis, dlq_queue: str, task_json: str):
    """
    Envía un mensaje a la Dead Letter Queue (DLQ).
    """
    try:
        r.lpush(dlq_queue, task_json)
        logging.warning(f"Tarea enviada a DLQ: {dlq_queue}")
    except Exception:
        logging.exception("Error enviando tarea a DLQ")
