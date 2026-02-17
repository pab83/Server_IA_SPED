import logging
import os
import gc
import numpy as np
import easyocr
from pdf2image import convert_from_path

class EasyOCRModel:
    """
    Wrapper optimizado para EasyOCR en CPU sin OOM.
    Devuelve un dict con:
    - 'text': texto completo
    - 'raw': lista de páginas con dicts {'text', 'confidence'} JSON serializable
    """

    def __init__(self, lang_list=None, gpu=False):
        self.reader = None
        self.lang_list = lang_list or ["en"]
        self.gpu = gpu

    def load(self):
        """Carga el modelo EasyOCR de manera lazy."""
        if self.reader is None:
            logging.info(f"Cargando EasyOCR (langs={self.lang_list}, gpu={self.gpu})")
            try:
                self.reader = easyocr.Reader(self.lang_list, gpu=self.gpu)
            except Exception as e:
                logging.exception("Error cargando EasyOCRModel")
                raise e

    @staticmethod
    def _normalize_results(results):
        """Convierte resultados de EasyOCR a dicts JSON serializables."""
        normalized = []
        for r in results:
            text = str(r[1])
            confidence = float(r[2]) if len(r) > 2 else None
            normalized.append({"text": text, "confidence": confidence})
        return normalized

    def predict(self, file_path: str):
        """Procesa PDFs o imágenes, página por página para evitar OOM."""
        self.load()

        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Archivo no encontrado: {file_path}")

        ext = os.path.splitext(file_path)[1].lower()
        all_pages_normalized = []

        try:
            if ext == ".pdf":
                logging.info(f"Procesando PDF: {file_path}")

                # Convertir PDF → imágenes **una a una**
                pages = convert_from_path(
                    file_path,
                    dpi=150,        # menor DPI para ahorrar RAM
                    fmt="jpeg",
                    thread_count=1  # evitar cargar todas las páginas a la vez
                )

                full_text_list = []

                for i, page in enumerate(pages, start=1):
                    logging.info(f"OCR página {i}/{len(pages)}")
                    image = np.array(page)
                    results = self.reader.readtext(
                        image,
                        detail=1,
                        paragraph=True
                    )
                    del image
                    gc.collect()

                    normalized = self._normalize_results(results)
                    all_pages_normalized.append(normalized)
                    full_text_list.extend([r["text"] for r in normalized])

                    # Liberar página
                    del page
                    gc.collect()

                full_text = "\n".join(full_text_list)
                del pages
                gc.collect()

            else:
                logging.info(f"Procesando imagen: {file_path}")
                results = self.reader.readtext(file_path, detail=1, paragraph=True)
                all_pages_normalized = [self._normalize_results(results)]
                full_text = "\n".join([r["text"] for r in all_pages_normalized[0]])

            return {
                "text": full_text,
                "raw": all_pages_normalized
            }

        except Exception:
            logging.exception("Error en OCR")
            raise

