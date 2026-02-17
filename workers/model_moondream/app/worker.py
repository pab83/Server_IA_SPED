import os
import logging
from schemas.task import TaskMessage
from base_worker import BaseWorker
from model import MoondreamModel
from messaging.redis_client import RedisQueueClient 

logging.basicConfig(level=logging.INFO)

REDIS_HOST = os.getenv("REDIS_HOST", "redis")
QUEUE_NAME = "cola_modelo_moondream"
RESULT_QUEUE = "cola_resultados"

class MoondreamWorker(BaseWorker):
    def __init__(self, redis_client: RedisQueueClient, queue_name, result_queue):
        super().__init__(redis_client, queue_name, result_queue) 
        logging.info("Inicializando modelo Moondream...")
        self.model = MoondreamModel(device="cpu")
        logging.info("Modelo Moondream inicializado correctamente")  

    def process_task(self, task: TaskMessage) -> dict:
        file_path = task.payload.get("file_path")
        prompt = task.payload.get("prompt", "")

        if not file_path:
            raise ValueError("Payload no contiene 'file_path'")

        caption = self.model.predict(file_path, prompt)
        return {"result": caption}