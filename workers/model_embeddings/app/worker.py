import os
import hashlib
import logging
from schemas.task import TaskMessage
from base_worker import BaseWorker
from model import EmbeddingModel
from messaging.redis_client import RedisQueueClient 

logging.basicConfig(level=logging.INFO)

REDIS_HOST = os.getenv("REDIS_HOST", "redis")
QUEUE_NAME = "cola_modelo_embeddings"
RESULT_QUEUE = "cola_resultados"

class EmbeddingsWorker(BaseWorker):
    def __init__(self, redis_client: RedisQueueClient, queue_name, result_queue):
        super().__init__(redis_client, queue_name, result_queue) 
        logging.info("Inicializando modelo Embeddings...")
        self.model = EmbeddingModel(device="cpu")
        logging.info("Modelo Embeddings inicializado correctamente")

    def process_task(self, task: TaskMessage) -> dict:
        document_text = task.payload.get("document_text", "")
        if not document_text:
            raise ValueError("Payload no contiene 'document_text'")

        embedding = self.model.predict(document_text)
        content_hash = hashlib.sha256(document_text.encode()).hexdigest()

        return {
            "embedding": embedding,
            "embedding_dim": len(embedding),
            "content_hash": content_hash
        }

