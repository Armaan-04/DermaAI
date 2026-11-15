
import torch
from PIL import Image
from transformers import AutoProcessor

import sys
from pathlib import Path

# Add project root to Python path
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from config import MODEL_NAME, CLASS_NAMES, MODEL_DIR
from model import SkinConditionClassifier


class SkinConditionPredictor:

    def __init__(self):

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        # Load image processor
        self.processor = AutoProcessor.from_pretrained(MODEL_NAME)

        # Load model architecture
        self.model = SkinConditionClassifier().to(self.device)

        # Load trained classification head
        checkpoint_path = MODEL_DIR / "medsiglip_scin_classifier.pth"

        checkpoint = torch.load(
            checkpoint_path,
            map_location=self.device,
            weights_only=True
        )

        self.model.classifier.load_state_dict(checkpoint)

        self.model.eval()

    def predict(self, image):

        # Convert image to RGB
        image = image.convert("RGB")

        # Preprocess image
        inputs = self.processor(
            images=image,
            return_tensors="pt"
        )

        pixel_values = inputs["pixel_values"].to(self.device)

        # Generate prediction
        with torch.no_grad():
            logits = self.model(pixel_values)

            probabilities = torch.softmax(logits, dim=1)

            confidence, predicted_index = probabilities.max(dim=1)

        # Get predicted condition
        condition = CLASS_NAMES[predicted_index.item()]

        return {
            "condition": condition,
            "confidence": round(confidence.item() * 100, 2)
        }


if __name__ == "__main__":

    predictor = SkinConditionPredictor()

    print("DermaAI prediction pipeline loaded successfully!")
    print("Running on:", predictor.device)
