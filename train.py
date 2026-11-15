
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoProcessor
from PIL import Image
import pandas as pd
from pathlib import Path

from config import (
    MODEL_NAME,
    CLASS_NAMES,
    NUM_CLASSES,
    MODEL_DIR
)
from model import SkinConditionClassifier


# Training configuration
BATCH_SIZE = 8
EPOCHS = 10
LEARNING_RATE = 0.001
PATIENCE = 3

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data" / "processed" / "SCIN"
SPLIT_DIR = DATA_DIR / "splits"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class SCINDataset(Dataset):

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


processor = AutoProcessor.from_pretrained(MODEL_NAME)


def collate_fn(batch):
    images, labels = zip(*batch)

    inputs = processor(
        images=list(images),
        return_tensors="pt"
    )

    return inputs["pixel_values"], torch.tensor(labels)


# Load datasets
train_dataset = SCINDataset(SPLIT_DIR / "train.csv")
val_dataset = SCINDataset(SPLIT_DIR / "validation.csv")

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0,
    collate_fn=collate_fn,
    pin_memory=torch.cuda.is_available()
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
    collate_fn=collate_fn,
    pin_memory=torch.cuda.is_available()
)


# Calculate class weights from training data
class_counts = (
    train_dataset.data["class_index"]
    .value_counts()
    .sort_index()
)

class_weights = (
    len(train_dataset) /
    (NUM_CLASSES * class_counts)
).values

class_weights = torch.tensor(
    class_weights,
    dtype=torch.float32,
    device=device
)


# Initialize model
model = SkinConditionClassifier().to(device)

criterion = nn.CrossEntropyLoss(
    weight=class_weights
)

optimizer = torch.optim.AdamW(
    model.classifier.parameters(),
    lr=LEARNING_RATE
)

MODEL_DIR.mkdir(parents=True, exist_ok=True)

best_val_loss = float("inf")
epochs_without_improvement = 0

print("\nStarting DermaAI training...")
print("Device:", device)
print("Training images:", len(train_dataset))
print("Validation images:", len(val_dataset))
print("Classes:", CLASS_NAMES)


for epoch in range(EPOCHS):

    # Training
    model.train()
    model.backbone.eval()
    train_loss = 0.0
    train_correct = 0
    train_total = 0

    for pixel_values, labels in train_loader:

        pixel_values = pixel_values.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(pixel_values)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        train_loss += loss.item() * labels.size(0)

        predictions = outputs.argmax(dim=1)
        train_correct += (predictions == labels).sum().item()
        train_total += labels.size(0)

    train_loss /= train_total
    train_accuracy = train_correct / train_total


    # Validation
    model.eval()
    val_loss = 0.0
    val_correct = 0
    val_total = 0

    with torch.no_grad():

        for pixel_values, labels in val_loader:

            pixel_values = pixel_values.to(device)
            labels = labels.to(device)

            outputs = model(pixel_values)
            loss = criterion(outputs, labels)

            val_loss += loss.item() * labels.size(0)

            predictions = outputs.argmax(dim=1)
            val_correct += (predictions == labels).sum().item()
            val_total += labels.size(0)

    val_loss /= val_total
    val_accuracy = val_correct / val_total

    print(f"\nEpoch {epoch + 1}/{EPOCHS}")
    print(f"Training Loss: {train_loss:.4f}")
    print(f"Training Accuracy: {train_accuracy:.4f}")
    print(f"Validation Loss: {val_loss:.4f}")
    print(f"Validation Accuracy: {val_accuracy:.4f}")


    # Save best classification head
    if val_loss < best_val_loss:

        best_val_loss = val_loss
        epochs_without_improvement = 0

        torch.save(
            model.classifier.state_dict(),
            MODEL_DIR / "medsiglip_scin_classifier.pth"
        )

        print("Best classification head saved!")

    else:
        epochs_without_improvement += 1

    if epochs_without_improvement >= PATIENCE:
        print("\nEarly stopping triggered.")
        break


print("\nTraining completed.")
print("Best validation loss:", round(best_val_loss, 4))
print("Saved model:", MODEL_DIR / "medsiglip_scin_classifier.pth")
