
"""DermaAI application configuration."""

from pathlib import Path

# Project directories
BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"
MODEL_PATH = MODEL_DIR / "medsiglip_scin_classifier.pth"

# Pretrained model configuration
MODEL_NAME = "google/medsiglip-448"
IMAGE_SIZE = 448
NUM_CLASSES = 8

# Classification labels
CLASS_NAMES = [
    "Eczema",
    "Allergic Contact Dermatitis",
    "Urticaria",
    "Insect Bite",
    "Folliculitis",
    "Psoriasis",
    "Tinea",
    "Herpes Simplex",
]

# Model feature configuration
FEATURE_SIZE = 1152

