import logging
import sys
import time

logging.basicConfig(level=logging.INFO)

if __name__ == "__main__":
    logging.info("Iniciando worker OCR...")

    # Queremos importar al arrancar, pero sin que el contenedor se caiga si el import
    # falla temporalmente o tarda: reintentamos y dejamos logs claros.
    while True:
        try:
            logging.info("Importando módulos OCR...")
            from worker import OCRWorker, REDIS_HOST, QUEUE_NAME, RESULT_QUEUE
            from messaging.redis_client import RedisQueueClient
            logging.info("Módulos OCR importados correctamente")
            break
        except Exception as e:
            logging.exception(f"Fallo importando módulos OCR (reintentando): {e}")
            time.sleep(5)

    while True:
        try:
            logging.info("Worker PP-OCRv4 CPU iniciado")

            logging.info("Conectando a Redis...")
            redis_client = RedisQueueClient(host=REDIS_HOST)
            logging.info("Conexión a Redis establecida")

            logging.info("Creando instancia del worker...")
            worker = OCRWorker(
                redis_client=redis_client,
                queue_name=QUEUE_NAME,
                result_queue=RESULT_QUEUE,
            )
            logging.info("Instancia del worker creada correctamente")

            logging.info("Iniciando bucle de consumo...")
            worker.run()
        except KeyboardInterrupt:
            raise
        except Exception as e:
            logging.exception(f"Error fatal en worker OCR (reiniciando): {e}")
            time.sleep(5)