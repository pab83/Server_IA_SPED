import os
import logging
from schemas.task import TaskMessage
from base_worker import BaseWorker
from model import MoondreamModel
from messaging.redis_client import RedisQueueClient

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

REDIS_HOST = os.getenv("REDIS_HOST", "redis")
QUEUE_NAME = "cola_modelo_moondream"
RESULT_QUEUE = "cola_resultados_moondream"

class MoondreamWorker(BaseWorker):
    def __init__(self, redis_client: RedisQueueClient, queue_name, result_queue):
        super().__init__(redis_client, queue_name, result_queue)
        self.model = MoondreamModel(device="cpu")
        if not self.model.loaded:
            logging.info("Inicializando modelo Moondream...")
            self.model.load()
        

    def process_task(self, task: TaskMessage) -> dict:
        file_path = task.payload.get("file_path")
        prompt = task.payload.get("prompt", "Describe la imagen en detalle.")

        if not file_path:
            raise ValueError("Payload no contiene 'file_path'")

        # Logging mínimo para no saturar disco
        logging.debug(f"Procesando archivo: {file_path}")

        caption = self.model.predict(file_path, prompt)
        return {"result": caption}

if __name__ == "__main__":
    redis_client = RedisQueueClient(host=REDIS_HOST)
    worker = MoondreamWorker(redis_client, QUEUE_NAME, RESULT_QUEUE)
    worker.run()
