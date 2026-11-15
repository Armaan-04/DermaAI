
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoProcessor
from PIL import Image
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)

from config import MODEL_NAME, CLASS_NAMES, NUM_CLASSES, MODEL_DIR
from model import SkinConditionClassifier


# Configuration
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data" / "processed" / "SCIN"
TEST_CSV = DATA_DIR / "splits" / "test.csv"
CHECKPOINT = MODEL_DIR / "medsiglip_scin_classifier.pth"

BATCH_SIZE = 8

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Test dataset
class SCINTestDataset(Dataset):

    def __init__(self, csv_file):
        self.data = pd.read_csv(csv_file)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        row = self.data.iloc[index]

        image_path = BASE_DIR / row["image_path"]
        image = Image.open(image_path).convert("RGB")

        label = int(row["class_index"])

        return image, label


# Load processor
processor = AutoProcessor.from_pretrained(MODEL_NAME)


def collate_fn(batch):
    images, labels = zip(*batch)

    inputs = processor(
        images=list(images),
        return_tensors="pt"
    )

    return inputs["pixel_values"], torch.tensor(labels)


test_dataset = SCINTestDataset(TEST_CSV)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
    collate_fn=collate_fn
)


# Load trained model
model = SkinConditionClassifier().to(device)

checkpoint = torch.load(
    CHECKPOINT,
    map_location=device,
    weights_only=True
)

model.classifier.load_state_dict(checkpoint)

model.eval()


# Perform predictions
all_predictions = []
all_labels = []

print("\nStarting test dataset evaluation...")
print("Device:", device)
print("Test images:", len(test_dataset))

with torch.no_grad():

    for pixel_values, labels in test_loader:

        pixel_values = pixel_values.to(device)

        outputs = model(pixel_values)

        predictions = outputs.argmax(dim=1)

        all_predictions.extend(predictions.cpu().tolist())
        all_labels.extend(labels.tolist())


# Calculate evaluation metrics
accuracy = accuracy_score(all_labels, all_predictions)

print("\n========== TEST RESULTS ==========")
print("Test Accuracy:", round(accuracy * 100, 2), "%")

print("\n========== CLASSIFICATION REPORT ==========")

print(
    classification_report(
        all_labels,
        all_predictions,
        labels=list(range(NUM_CLASSES)),
        target_names=CLASS_NAMES,
        zero_division=0
    )
)


# Generate confusion matrix
cm = confusion_matrix(
    all_labels,
    all_predictions,
    labels=list(range(NUM_CLASSES))
)

print("\nConfusion Matrix:")
print(cm)


# Save confusion matrix visualization
display = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=CLASS_NAMES
)

fig, ax = plt.subplots(figsize=(12, 10))

display.plot(
    ax=ax,
    cmap="Blues",
    xticks_rotation=45,
    values_format="d"
)

plt.title("DermaAI - Test Confusion Matrix")
plt.tight_layout()

output_path = BASE_DIR / "models" / "confusion_matrix.png"

plt.savefig(output_path, dpi=300)
plt.close()

print("\nConfusion matrix saved to:", output_path)
print("\nEvaluation completed successfully!")
