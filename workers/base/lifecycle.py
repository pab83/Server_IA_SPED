import logging
from schemas.result import ResultMessage
from datetime import datetime

logging.basicConfig(level=logging.INFO)

def publish_result(mq_client, queue_name: str, result: ResultMessage):
    """
    Publica el resultado en la cola de resultados.
    Desacoplado del backend de mensajería.
    """
    if not isinstance(result, ResultMessage):
        raise ValueError("El resultado debe ser un ResultMessage")

    try:
        # Añadimos timestamp de publicación
        result_dict = result.model_dump()
        result_dict["published_at"] = datetime.utcnow().isoformat()
        mq_client.publish(queue_name, result_dict)
        logging.info(f"Resultado publicado en {queue_name} para message_id {result.message_id}")
    except Exception:
        logging.exception(f"Error publicando resultado message_id {result.message_id}")

def build_result(task, result_data: dict, model_name: str) -> ResultMessage:
    """
    Construye ResultMessage a partir de TaskMessage y el resultado del worker.
    """
    from schemas.result import Status
    from datetime import datetime

    return ResultMessage(
        message_id=task.message_id,
        correlation_id=task.correlation_id,
        model=model_name,
        status=Status.SUCCESS,
        processing_time_ms=int((datetime.utcnow() - task.timestamp).total_seconds() * 1000),
        **result_data
    )

