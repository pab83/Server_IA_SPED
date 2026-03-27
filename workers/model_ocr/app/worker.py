import os
import logging
from schemas.task import TaskMessage
from base_worker import BaseWorker
from model import EasyOCRModel
from messaging.redis_client import RedisQueueClient  

logging.basicConfig(level=logging.INFO)

REDIS_HOST = os.getenv("REDIS_HOST", "redis")
QUEUE_NAME = "cola_modelo_ocr"
RESULT_QUEUE = "cola_resultados"


class OCRWorker(BaseWorker):


    def __init__(self, redis_client: RedisQueueClient, queue_name, result_queue):
        super().__init__(redis_client, queue_name, result_queue)

        logging.info("Inicializando modelo EasyOCR...")
        try:
            self.model = EasyOCRModel(lang_list=["en"], gpu=False)
            logging.info("Modelo EasyOCR inicializado correctamente")
        except Exception as e:
            logging.exception(f"Error inicializando EasyOCR: {e}")
            raise

    def process_task(self, task: TaskMessage) -> dict:
        """
        Procesa únicamente la tarea que llega por Redis.
        """
        file_path = task.payload.get("file_path")
        if not file_path:
            raise ValueError("Payload no contiene 'file_path'")

        if not os.path.exists(file_path):
            logging.warning(f"Archivo no encontrado, saltando: {file_path}")
            return {"result": None, "error": "file_not_found"}

        # model.predict() ya devuelve datos normalizados y JSON serializables
        # {"text": "...", "raw": [[{"text": "...", "confidence": 0.9}, ...], ...]}
        ocr_result = self.model.predict(file_path)
        return {"result": ocr_result}