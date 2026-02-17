import logging
from worker import MoondreamWorker, REDIS_HOST, QUEUE_NAME, RESULT_QUEUE
from messaging.redis_client import RedisQueueClient

logging.basicConfig(level=logging.INFO)

if __name__ == "__main__":
    redis_client = RedisQueueClient(host=REDIS_HOST)

    worker = MoondreamWorker(
        redis_client=redis_client,
        queue_name=QUEUE_NAME,
        result_queue=RESULT_QUEUE,
    )

    logging.info("Worker Moondream GPU iniciado")
    worker.run()
