import logging
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from PIL import Image

MODEL_ID = "vikhyatk/moondream2"

class MoondreamModel:
    """
    Wrapper profesional para Moondream2.
    Compatible con BaseWorker.
    """

    def __init__(self, device: str = None):
        """
        device: "cuda" o "cpu". Por defecto detecta automáticamente.
        """
        self.model = None
        self.tokenizer = None
        self.device = device or ("cpu" if torch.cuda.is_available() else "cpu")

    def load(self):
        """
        Carga el modelo y tokenizer en el dispositivo especificado.
        Lazy load: solo se carga la primera vez.
        """
        if self.model is None:
            logging.info(f"Cargando Moondream {MODEL_ID} en {self.device}")

            try:
                self.tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

                self.model = AutoModelForCausalLM.from_pretrained(
                    MODEL_ID,
                    torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                    device_map="auto" if self.device == "cuda" else None
                )

                self.model.eval()
            except Exception as e:
                logging.exception("Error cargando MoondreamModel")
                raise e

    @torch.no_grad()
    def predict(self, image_path: str, prompt: str) -> str:
        """
        Genera un caption/descripción para la imagen dada.
        """
        if not image_path:
            raise ValueError("Debe indicar la ruta del archivo para Moondream")
        if not prompt:
            raise ValueError("Debe proporcionar un prompt para Moondream")

        self.load()

        try:
            image = Image.open(image_path).convert("RGB")

            # Tokenizar prompt
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

            # Generar salida
            output = self.model.generate(**inputs, max_new_tokens=128)

            response = self.tokenizer.decode(output[0], skip_special_tokens=True)

            return response

        except Exception as e:
            logging.exception(f"Error generando caption Moondream para {image_path}")
            raise e

