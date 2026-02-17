import logging
from sentence_transformers import SentenceTransformer
import torch

MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"

class EmbeddingModel:
    """
    Wrapper para generar embeddings con SentenceTransformer.
    
    """

    def __init__(self, device: str = "cpu"):
        self.model = None
        self.device = device

    def load(self):
        """
        Carga el modelo en la GPU o CPU especificada.
        Solo se carga una vez (lazy load).
        """
        if self.model is None:
            logging.info(f"Cargando embeddings {MODEL_ID} en {self.device}")
            self.model = SentenceTransformer(MODEL_ID, device=self.device)

    @torch.no_grad()
    def predict(self, text: str) -> list[float]:
        """
        Genera el embedding del texto.
        Devuelve lista de floats normalizada.
        """
        if not text:
            raise ValueError("El texto no puede estar vacío para generar embeddings")

        self.load()

        embedding = self.model.encode(
            text,
            normalize_embeddings=True
        )

        return embedding.tolist()


