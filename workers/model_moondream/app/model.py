import logging
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from PIL import Image

MODEL_ID = "vikhyatk/moondream2"


class MoondreamModel:
    """
    Wrapper para Moondream2.
    """

    def __init__(self, device: str = None):
        """
        device: "cuda" o "cpu". Detecta automáticamente si no se especifica.
        """
        self.model = None
        self.tokenizer = None
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    def load(self):
        """
        Carga el modelo y tokenizer una sola vez (lazy load).
        """
        if self.model is not None:
            return

        logging.info(f"Cargando Moondream {MODEL_ID} en {self.device}")

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                MODEL_ID,
                trust_remote_code=True
            )

            self.model = AutoModelForCausalLM.from_pretrained(
                MODEL_ID,
                trust_remote_code=True,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                device_map="auto" if self.device == "cuda" else None
            )

            if self.device == "cpu":
                self.model.to("cpu")

            self.model.eval()

            logging.info("Moondream cargado correctamente.")

        except Exception as e:
            logging.exception("Error cargando MoondreamModel")
            raise e

    @torch.no_grad()
    def predict(self, image_path: str, prompt: str) -> str:
        """
        Genera caption para una imagen usando Moondream.
        """
        if not image_path:
            raise ValueError("Debe indicar la ruta del archivo para Moondream")

        if not prompt:
            raise ValueError("Debe proporcionar un prompt para Moondream")

        self.load()

        try:
            image = Image.open(image_path).convert("RGB")

            # Encode image
            image_embeds = self.model.encode_image(image)

            # Generate response
            response = self.model.answer_question(
                image_embeds,
                prompt,
                self.tokenizer
            )

            return response

        except Exception as e:
            logging.exception(
                f"Error generando caption Moondream para {image_path}"
            )
            raise e
