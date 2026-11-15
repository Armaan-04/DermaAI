
import torch
from PIL import Image
from transformers import (
    AutoConfig,
    AutoProcessor,
    AutoModelForZeroShotImageClassification
)

MODEL_NAME = "google/medsiglip-448"

device = "cuda" if torch.cuda.is_available() else "cpu"

# Correct the text configuration
config = AutoConfig.from_pretrained(MODEL_NAME)
config.text_config.bos_token_id = None
config.text_config.eos_token_id = 1
config.text_config.pad_token_id = 1

# Load processor
processor = AutoProcessor.from_pretrained(MODEL_NAME)

# Load model in FP32
model = AutoModelForZeroShotImageClassification.from_pretrained(
    MODEL_NAME,
    config=config,
    dtype=torch.float32
).to(device)

model.eval()

# Create synthetic test image
image = Image.new("RGB", (448, 448), color=(180, 130, 120))

labels = [
    "a clinical photograph of skin",
    "a clinical photograph of healthy skin"
]

# Prepare inputs
inputs = processor(
    text=labels,
    images=image,
    padding="max_length",
    return_tensors="pt"
)

inputs = {k: v.to(device) for k, v in inputs.items()}

# Run inference
with torch.no_grad():
    outputs = model(**inputs)

print("\nIMAGE-TEXT INFERENCE SUCCESSFUL")
print("Logits shape:", outputs.logits_per_image.shape)
print("Image embeddings:", outputs.image_embeds.shape)
print("Text embeddings:", outputs.text_embeds.shape)
print("Precision:", next(model.parameters()).dtype)
print("GPU:", next(model.parameters()).device)
print(
    "GPU memory:",
    round(torch.cuda.memory_allocated(0) / 1024**3, 2),
    "GB"
)

