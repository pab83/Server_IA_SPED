def should_retry(task, max_retries: int) -> bool:
    """
    Devuelve True si el task aún puede reintentarse según max_retries.
    """
    return task.retry_count < max_retries
