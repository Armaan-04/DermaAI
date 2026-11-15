import csv
import io
from pathlib import Path

import pyarrow.parquet as pq
from PIL import Image

PROJECT_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_DIR / "data" / "SCIN" / "data"
OUTPUT_DIR = PROJECT_DIR / "data" / "processed" / "SCIN"
IMAGE_DIR = OUTPUT_DIR / "images"
CSV_PATH = OUTPUT_DIR / "metadata.csv"

IMAGE_COLUMNS = [
    "image_1_path",
    "image_2_path",
    "image_3_path"
]

COLUMNS = [
    "case_id",
    "source",
    "year",
    "weighted_skin_condition_label",
    "dermatologist_skin_condition_on_label_name",
    "dermatologist_skin_condition_confidence",
    "dermatologist_gradable_for_skin_condition_1",
    "dermatologist_gradable_for_skin_condition_2",
    "dermatologist_gradable_for_skin_condition_3",
    "dermatologist_fitzpatrick_skin_type_label_1",
    "dermatologist_fitzpatrick_skin_type_label_2",
    "dermatologist_fitzpatrick_skin_type_label_3",
    "monk_skin_tone_label_india",
    "monk_skin_tone_label_us",
    "image_1_shot_type",
    "image_2_shot_type",
    "image_3_shot_type",
    *IMAGE_COLUMNS
]

def main():
    IMAGE_DIR.mkdir(parents=True, exist_ok=True)

    parquet_files = sorted(RAW_DIR.glob("*.parquet"))

    if len(parquet_files) != 26:
        raise RuntimeError(
            f"Expected 26 Parquet files, found {len(parquet_files)}"
        )

    fields = [
        "case_id",
        "image_number",
        "image_path",
        "original_image_path",
        "source",
        "year",
        "weighted_skin_condition_label",
        "dermatologist_skin_condition_on_label_name",
        "dermatologist_skin_condition_confidence",
        "gradable_assessment_1",
        "gradable_assessment_2",
        "gradable_assessment_3",
        "fitzpatrick_label_1",
        "fitzpatrick_label_2",
        "fitzpatrick_label_3",
        "monk_skin_tone_india",
        "monk_skin_tone_us",
        "shot_type"
    ]

    total_cases = 0
    total_images = 0
    failed_images = 0

    with CSV_PATH.open("w", newline="", encoding="utf-8-sig") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fields)
        writer.writeheader()

        for file_index, parquet_path in enumerate(parquet_files, start=1):
            print(f"Processing file {file_index}/26: {parquet_path.name}")

            parquet_file = pq.ParquetFile(parquet_path)

            for batch in parquet_file.iter_batches(
                batch_size=32,
                columns=COLUMNS
            ):
                for record in batch.to_pylist():
                    total_cases += 1
                    case_id = str(record["case_id"])

                    for number, column in enumerate(IMAGE_COLUMNS, start=1):
                        image_data = record.get(column)

                        if not image_data or not image_data.get("bytes"):
                            continue

                        image_bytes = bytes(image_data["bytes"])
                        original_path = image_data.get("path") or ""
                        extension = Path(original_path).suffix.lower()

                        if extension not in [".jpg", ".jpeg", ".png", ".webp"]:
                            extension = ".jpg"

                        filename = f"{case_id}_image_{number}{extension}"
                        destination = IMAGE_DIR / filename

                        try:
                            with Image.open(io.BytesIO(image_bytes)) as image:
                                image.verify()

                            destination.write_bytes(image_bytes)

                        except Exception as error:
                            failed_images += 1
                            print(
                                f"Image extraction failed: {case_id}, "
                                f"image {number}: {error}"
                            )
                            continue

                        writer.writerow({
                            "case_id": case_id,
                            "image_number": number,
                            "image_path": f"images/{filename}",
                            "original_image_path": original_path,
                            "source": record.get("source"),
                            "year": record.get("year"),
                            "weighted_skin_condition_label":
                                record.get("weighted_skin_condition_label"),
                            "dermatologist_skin_condition_on_label_name":
                                record.get("dermatologist_skin_condition_on_label_name"),
                            "dermatologist_skin_condition_confidence":
                                record.get("dermatologist_skin_condition_confidence"),
                            "gradable_assessment_1":
                                record.get("dermatologist_gradable_for_skin_condition_1"),
                            "gradable_assessment_2":
                                record.get("dermatologist_gradable_for_skin_condition_2"),
                            "gradable_assessment_3":
                                record.get("dermatologist_gradable_for_skin_condition_3"),
                            "fitzpatrick_label_1":
                                record.get("dermatologist_fitzpatrick_skin_type_label_1"),
                            "fitzpatrick_label_2":
                                record.get("dermatologist_fitzpatrick_skin_type_label_2"),
                            "fitzpatrick_label_3":
                                record.get("dermatologist_fitzpatrick_skin_type_label_3"),
                            "monk_skin_tone_india":
                                record.get("monk_skin_tone_label_india"),
                            "monk_skin_tone_us":
                                record.get("monk_skin_tone_label_us"),
                            "shot_type": record.get(f"image_{number}_shot_type")
                        })

                        total_images += 1

    print("\nExtraction completed.")
    print("Parquet files processed:", len(parquet_files))
    print("Cases processed:", total_cases)
    print("Images successfully extracted:", total_images)
    print("Images that failed:", failed_images)
    print("Output image folder:", IMAGE_DIR)
    print("Metadata CSV:", CSV_PATH)


if __name__ == "__main__":
    main()
