import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from PIL import Image
import logging

MODEL_ID = "vikhyatk/moondream2"
REVISION = "6b714b26eea5cbd9f31e4edb2541c170afa935ba"

# Limitar threads para CPU
torch.set_num_threads(1)
torch.set_num_interop_threads(1)

logging.basicConfig(level=logging.INFO)

class MoondreamModel:
    def __init__(self, device="cpu"):
        self.device = device
        self.model = None
        self.tokenizer = None
        self.loaded = False

    def load(self):
        if self.model is not None:
            return

        logging.info(f"Cargando Moondream {MODEL_ID} en {self.device}")
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=REVISION, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(MODEL_ID, revision=REVISION, trust_remote_code=True).to(self.device)
        self.model.eval()
        self.loaded = True
        logging.info("Moondream cargado correctamente")

    @torch.no_grad()
    def predict(self, image_path: str, prompt: str = "Describe la imagen en detalle.") -> str:
        if self.model is None or self.tokenizer is None:
            self.load()

        try:
            image = Image.open(image_path).convert("RGB")
            encoded_image = self.model.encode_image(image)
            answer = self.model.answer_question(encoded_image, prompt, self.tokenizer)
            return answer
        except Exception:
            logging.exception(f"Error generando caption para {image_path}")
            raise
