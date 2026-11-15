from pathlib import Path

import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data" / "processed" / "SCIN" / "training_metadata.csv"
OUTPUT_DIR = ROOT / "data" / "processed" / "SCIN" / "splits"

RANDOM_STATE = 42

df = pd.read_csv(SOURCE)

case_data = df.groupby("case_id").agg(
    label=("label", "first"),
    label_count=("label", "nunique")
).reset_index()

if (case_data["label_count"] != 1).any():
    raise ValueError("Some cases have multiple condition labels.")

# Assign every case to one of ten stratified group folds.
first_split = StratifiedGroupKFold(
    n_splits=10,
    shuffle=True,
    random_state=RANDOM_STATE
)

fold_assignment = pd.Series(index=case_data.index, dtype="int64")

for fold_number, (_, heldout_indices) in enumerate(
    first_split.split(
        case_data,
        y=case_data["label"],
        groups=case_data["case_id"]
    )
):
    fold_assignment.iloc[heldout_indices] = fold_number

# Seven folds for training; three folds for temporary data.
train_cases = set(
    case_data.loc[fold_assignment >= 3, "case_id"]
)

temporary_cases = case_data.loc[fold_assignment < 3].copy()

# Divide temporary cases equally between validation and testing.
second_split = StratifiedGroupKFold(
    n_splits=2,
    shuffle=True,
    random_state=RANDOM_STATE
)

validation_indices, test_indices = next(
    second_split.split(
        temporary_cases,
        y=temporary_cases["label"],
        groups=temporary_cases["case_id"]
    )
)

validation_cases = set(
    temporary_cases.iloc[validation_indices]["case_id"]
)
test_cases = set(
    temporary_cases.iloc[test_indices]["case_id"]
)

train_df = df[df["case_id"].isin(train_cases)].copy()
validation_df = df[df["case_id"].isin(validation_cases)].copy()
test_df = df[df["case_id"].isin(test_cases)].copy()

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

train_df.to_csv(OUTPUT_DIR / "train.csv", index=False)
validation_df.to_csv(OUTPUT_DIR / "validation.csv", index=False)
test_df.to_csv(OUTPUT_DIR / "test.csv", index=False)

assert train_cases.isdisjoint(validation_cases)
assert train_cases.isdisjoint(test_cases)
assert validation_cases.isdisjoint(test_cases)

print("Corrected dataset splitting completed.")
print(f"Training: {len(train_df)} images, {len(train_cases)} cases")
print(f"Validation: {len(validation_df)} images, {len(validation_cases)} cases")
print(f"Testing: {len(test_df)} images, {len(test_cases)} cases")
print("\nCondition distribution by split:")

for name, split_df in [
    ("Training", train_df),
    ("Validation", validation_df),
    ("Testing", test_df)
]:
    print(f"\n{name}:")
    print(split_df.groupby("label")["case_id"].nunique().to_string())

print("\nAll case-level leakage checks passed.")
