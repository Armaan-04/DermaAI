import ast
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data" / "processed" / "SCIN" / "metadata.csv"
OUTPUT = ROOT / "data" / "processed" / "SCIN" / "training_metadata.csv"

CLASSES = [
    "Eczema",
    "Allergic Contact Dermatitis",
    "Urticaria",
    "Insect Bite",
    "Folliculitis",
    "Psoriasis",
    "Tinea",
    "Herpes Simplex",
]

CLASS_TO_INDEX = {label: index for index, label in enumerate(CLASSES)}

QUALITY_COLUMNS = [
    "gradable_assessment_1",
    "gradable_assessment_2",
    "gradable_assessment_3",
]

df = pd.read_csv(SOURCE)
records = []

for case_id, group in df.groupby("case_id"):
    row = group.iloc[0]

    raw_weights = row["weighted_skin_condition_label"]

    try:
        weights = ast.literal_eval(raw_weights) if isinstance(raw_weights, str) else {}
    except (ValueError, SyntaxError):
        continue

    if not isinstance(weights, dict) or not weights:
        continue

    highest_weight = max(weights.values())
    highest_labels = [
        label for label, weight in weights.items()
        if weight == highest_weight
    ]

    if len(highest_labels) != 1:
        continue

    label = highest_labels[0]

    if label not in CLASS_TO_INDEX:
        continue

    assessments = [
        row[column]
        for column in QUALITY_COLUMNS
        if column in df.columns and pd.notna(row[column])
    ]

    if "NO_IMAGE_QUALITY_INSUFFICIENT" in assessments:
        continue

    if "DEFAULT_YES_IMAGE_QUALITY_SUFFICIENT" not in assessments:
        continue

    for _, image in group.iterrows():
        image_path = SOURCE.parent / image["image_path"]

        if not image_path.is_file():
            continue

        records.append({
            "case_id": case_id,
            "image_number": image["image_number"],
            "image_path": str(image_path.relative_to(ROOT)),
            "label": label,
            "class_index": CLASS_TO_INDEX[label],
        })

result = pd.DataFrame(records, columns=["case_id", "image_number", "image_path", "label", "class_index"])
result.to_csv(OUTPUT, index=False)

print("Label preparation completed.")
print(f"Total images: {len(result)}")
print(f"Unique cases: {result['case_id'].nunique()}")
print("\nImages per condition:")
print(result.groupby("label").size().to_string())
print(f"\nSaved to: {OUTPUT}")

