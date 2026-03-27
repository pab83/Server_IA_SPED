import logging
from abc import ABC, abstractmethod
from datetime import datetime , timezone
from schemas.task import TaskMessage
from schemas.result import ResultMessage, Status
from lifecycle import publish_result
from reliability.idempotency import mark_processed
from reliability.retry import should_retry
from reliability.dlq import send_to_dlq

logging.basicConfig(level=logging.INFO)

class BaseWorker(ABC):
    """
    Clase base para todos los workers.
    Gestiona:
      - Idempotencia
      - Reintentos
      - Publicación de resultados
      - DLQ
      - Uso de cliente de mensajería desacoplado
    """

    def __init__(self, mq_client, queue_name: str, result_queue: str):
        """
        mq_client: instancia de BaseQueueClient (RedisQueueClient o futuro KafkaClient)
        """
        self.mq_client = mq_client
        self.queue_name = queue_name
        self.result_queue = result_queue

    @abstractmethod
    def process_task(self, task: TaskMessage) -> dict:
        """
        Implementar en cada worker.
        Debe devolver un diccionario serializable para 'result' o 'embedding'.
        """
        pass

    def check_idempotency(self, task: TaskMessage) -> bool:
        """
        Devuelve True si es la primera vez que se procesa el mensaje.
        """
        # RedisQueueClient expone la conexión Redis real en el atributo `r`,
        # que es lo que espera la función `mark_processed`.
        redis_conn = getattr(self.mq_client, "r", self.mq_client)
        return mark_processed(redis_conn, task.target_model, task.message_id)

    def build_result(self, task: TaskMessage, result_data: dict) -> ResultMessage:
            """
            Construye ResultMessage completo.
            Gestiona la resta de datetimes con y sin zona horaria (UTC).
            """
            # Obtenemos la hora actual con zona horaria UTC explícita
            now = datetime.now(timezone.utc)
            
            # Calculamos la diferencia asegurándonos de que ambos tengan tz
            diff = now - task.timestamp
            processing_time_ms = int(diff.total_seconds() * 1000)

            return ResultMessage(
                message_id=task.message_id,
                correlation_id=task.correlation_id,
                model=self.__class__.__name__,
                status=Status.SUCCESS,
                processing_time_ms=processing_time_ms,
                **result_data
            )

    def handle_failure(self, task: TaskMessage):
        """
        Maneja errores: retries o DLQ
        """
        task.retry_count += 1
        if should_retry(task, task.max_retries):
            self.mq_client.publish(self.queue_name, task.model_dump())
            logging.info(f"Tarea {task.message_id} reintentada ({task.retry_count}/{task.max_retries})")
        else:
            dlq_name = f"{self.queue_name}_dlq"
            self.mq_client.send_to_dlq(dlq_name, task.model_dump())
            logging.warning(f"Tarea {task.message_id} enviada a DLQ")

    def run(self, handler=None):
        """
        Inicia el worker. Si se proporciona handler, se usa para procesar cada mensaje;
        si no, se usa process_task estándar.
        """
        logging.info(f"{self.__class__.__name__} iniciado, escuchando cola '{self.queue_name}'")

        def default_handler(task_dict):
            try:
                task = TaskMessage.model_validate(task_dict)
            except Exception:
                logging.exception("Error parseando TaskMessage, descartando")
                return

            if not self.check_idempotency(task):
                logging.info(f"Mensaje {task.message_id} ya procesado, saltando")
                return

            try:
                result_data = self.process_task(task)
                result = self.build_result(task, result_data)
                self.mq_client.publish(self.result_queue, result.model_dump())
            except Exception:
                logging.exception(f"Error procesando tarea {task.message_id}")
                self.handle_failure(task)

        self.mq_client.consume(self.queue_name, handler or default_handler)

