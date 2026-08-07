#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import json
import time
import unicodedata
from pathlib import Path
from typing import Any

from paddleocr import TextRecognition
from rapidfuzz.distance import Levenshtein
from tqdm import tqdm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate the pretrained PP-OCRv5 recognition model "
            "on MTH1000 text-line crops."
        )
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("data/processed/MTH1000_B0"),
        help="Root directory containing images/ and the TSV manifest.",
    )
    parser.add_argument(
        "--manifest",
        type=str,
        default="test.tsv",
        help="Manifest filename relative to dataset root.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/B0_PP-OCRv5_server_rec"),
        help="Directory used to save predictions and metrics.",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="PP-OCRv5_server_rec",
        help="PaddleOCR recognition model name.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="gpu:0",
        help='Inference device, for example "gpu:0" or "cpu".',
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Inference batch size.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Evaluate only the first N samples; 0 means all samples.",
    )
    parser.add_argument(
        "--unicode-normalization",
        choices=("none", "NFC"),
        default="NFC",
        help="Unicode normalization applied equally to GT and prediction.",
    )
    return parser.parse_args()


def normalize_text(text: str, mode: str) -> str:
    if mode == "NFC":
        return unicodedata.normalize("NFC", text)
    return text


def load_manifest(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(f"Manifest not found: {path}")

    with path.open("r", encoding="utf-8") as file:
        rows = list(csv.DictReader(file, delimiter="\t"))

    if not rows:
        raise RuntimeError(f"Manifest is empty: {path}")

    required = {"image_path", "text", "page_id", "line_number"}
    missing = required - set(rows[0])

    if missing:
        raise ValueError(
            f"Manifest is missing required columns: {sorted(missing)}"
        )

    return rows


def result_to_dict(result: Any) -> dict[str, Any]:
    if isinstance(result, dict):
        data = result
    elif hasattr(result, "json"):
        json_value = result.json
        if callable(json_value):
            json_value = json_value()
        data = json.loads(json_value) if isinstance(json_value, str) else json_value
    elif hasattr(result, "to_dict"):
        data = result.to_dict()
    else:
        try:
            data = dict(result)
        except Exception as exc:
            raise TypeError(
                f"Unsupported PaddleOCR result type: {type(result).__name__}"
            ) from exc

    if isinstance(data, dict) and isinstance(data.get("res"), dict):
        return data["res"]

    if not isinstance(data, dict):
        raise TypeError(
            f"PaddleOCR result could not be converted to dict: {type(data).__name__}"
        )

    return data


def extract_prediction(result: Any) -> tuple[str, float]:
    data = result_to_dict(result)

    prediction = (
        data.get("rec_text")
        or data.get("text")
        or data.get("label")
        or ""
    )

    score_value = data.get("rec_score")
    if score_value is None:
        score_value = data.get("score", 0.0)

    try:
        score = float(score_value)
    except (TypeError, ValueError):
        score = 0.0

    return str(prediction), score


def main() -> None:
    args = parse_args()

    if args.batch_size <= 0:
        raise ValueError("--batch-size must be greater than zero.")

    manifest_path = args.dataset_root / args.manifest
    rows = load_manifest(manifest_path)

    if args.limit > 0:
        rows = rows[: args.limit]

    args.output_dir.mkdir(parents=True, exist_ok=True)

    model = TextRecognition(
        model_name=args.model_name,
        device=args.device,
    )

    total_distance = 0
    total_gt_chars = 0
    exact_matches = 0
    total_confidence = 0.0
    processed = 0

    predictions: list[dict[str, Any]] = []

    start_time = time.perf_counter()

    for batch_start in tqdm(
        range(0, len(rows), args.batch_size),
        desc="B0 inference",
    ):
        batch_rows = rows[batch_start : batch_start + args.batch_size]

        image_paths = []
        for row in batch_rows:
            image_path = args.dataset_root / row["image_path"]
            if not image_path.is_file():
                raise FileNotFoundError(f"Crop image not found: {image_path}")
            image_paths.append(str(image_path))

        batch_results = list(
            model.predict(
                input=image_paths,
                batch_size=len(image_paths),
            )
        )

        if len(batch_results) != len(batch_rows):
            raise RuntimeError(
                "Prediction count differs from input count: "
                f"{len(batch_results)} != {len(batch_rows)}"
            )

        for row, result in zip(batch_rows, batch_results):
            pred_raw, confidence = extract_prediction(result)

            ground_truth = normalize_text(
                row["text"],
                args.unicode_normalization,
            )
            prediction = normalize_text(
                pred_raw,
                args.unicode_normalization,
            )

            edit_distance = Levenshtein.distance(
                ground_truth,
                prediction,
            )

            gt_length = len(ground_truth)
            pred_length = len(prediction)
            sample_cer = edit_distance / max(gt_length, 1)

            total_distance += edit_distance
            total_gt_chars += gt_length
            exact_matches += int(ground_truth == prediction)
            total_confidence += confidence
            processed += 1

            predictions.append(
                {
                    "image_path": row["image_path"],
                    "page_id": row["page_id"],
                    "line_number": int(row["line_number"]),
                    "ground_truth": ground_truth,
                    "prediction": prediction,
                    "confidence": confidence,
                    "edit_distance": edit_distance,
                    "gt_length": gt_length,
                    "pred_length": pred_length,
                    "sample_cer": sample_cer,
                    "exact_match": ground_truth == prediction,
                }
            )

    elapsed_seconds = time.perf_counter() - start_time

    corpus_cer = total_distance / max(total_gt_chars, 1)
    exact_match_accuracy = exact_matches / max(processed, 1)
    mean_confidence = total_confidence / max(processed, 1)
    lines_per_second = processed / max(elapsed_seconds, 1e-9)
    milliseconds_per_line = 1000.0 * elapsed_seconds / max(processed, 1)

    predictions.sort(
        key=lambda item: (
            item["sample_cer"],
            item["edit_distance"],
        ),
        reverse=True,
    )

    predictions_path = args.output_dir / "predictions.tsv"
    with predictions_path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as file:
        fieldnames = [
            "image_path",
            "page_id",
            "line_number",
            "ground_truth",
            "prediction",
            "confidence",
            "edit_distance",
            "gt_length",
            "pred_length",
            "sample_cer",
            "exact_match",
        ]

        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
            delimiter="\t",
        )
        writer.writeheader()
        writer.writerows(predictions)

    metrics = {
        "experiment": "B0",
        "model": args.model_name,
        "weights": "official_pretrained",
        "fine_tuned_on_mth1000": False,
        "custom_dictionary": False,
        "augmentation": False,
        "postprocessing": False,
        "device": args.device,
        "manifest": str(manifest_path),
        "unicode_normalization": args.unicode_normalization,
        "num_samples": processed,
        "num_ground_truth_characters": total_gt_chars,
        "total_edit_distance": total_distance,
        "cer": corpus_cer,
        "exact_match_accuracy": exact_match_accuracy,
        "mean_model_confidence": mean_confidence,
        "elapsed_seconds": elapsed_seconds,
        "lines_per_second": lines_per_second,
        "milliseconds_per_line": milliseconds_per_line,
    }

    metrics_path = args.output_dir / "metrics.json"
    with metrics_path.open("w", encoding="utf-8") as file:
        json.dump(
            metrics,
            file,
            ensure_ascii=False,
            indent=2,
        )

    print()
    print("=" * 64)
    print("B0 - PP-OCRv5 pretrained recognition baseline")
    print("=" * 64)
    print(f"Samples              : {processed}")
    print(f"GT characters        : {total_gt_chars}")
    print(f"Total edit distance  : {total_distance}")
    print(f"CER                  : {corpus_cer:.6f}")
    print(f"Exact-match accuracy : {exact_match_accuracy:.6f}")
    print(f"Mean confidence      : {mean_confidence:.6f}")
    print(f"Elapsed              : {elapsed_seconds:.2f} seconds")
    print(f"Lines/second         : {lines_per_second:.2f}")
    print(f"Milliseconds/line    : {milliseconds_per_line:.2f}")
    print(f"Predictions          : {predictions_path}")
    print(f"Metrics              : {metrics_path}")


if __name__ == "__main__":
    main()
