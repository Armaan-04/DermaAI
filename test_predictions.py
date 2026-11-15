
import pandas as pd
from PIL import Image
from pathlib import Path

from utils.predictor import SkinConditionPredictor

# Load 10 test images
df = pd.read_csv(
    "data/processed/SCIN/splits/test.csv"
).head(10)

# Initialize predictor once
predictor = SkinConditionPredictor()

correct = 0

print("\n10-IMAGE PREDICTION TEST\n")

for _, row in df.iterrows():

    image_path = Path(row["image_path"])
    image = Image.open(image_path).convert("RGB")

    result = predictor.predict(image)

    actual = row["label"]
    predicted = result["condition"]

    match = actual == predicted
    correct += int(match)

    status = "Correct" if match else "Incorrect"

    print(
        f"Actual: {actual} | "
        f"Predicted: {predicted} | "
        f"Confidence: {result['confidence']}% | "
        f"{status}"
    )

print(f"\nCorrect predictions: {correct}/10")
