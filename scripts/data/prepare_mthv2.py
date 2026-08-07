#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import json
import random
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm


IMAGE_EXTENSIONS = (
    ".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff",
)
SUBSETS = ("MTH1000", "MTH1200", "TKH")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare full MTHv2 for OCR recognition using the official "
            "train/test split. Validation is split only from official train pages."
        )
    )
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--train-split", type=Path, required=True)
    parser.add_argument("--test-split", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("data/processed/MTHv2"))
    parser.add_argument("--val-ratio", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--padding", type=int, default=2)
    parser.add_argument("--min-side", type=int, default=4)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def find_image(img_dir: Path, stem: str) -> Path:
    for extension in IMAGE_EXTENSIONS:
        for suffix in (extension, extension.upper()):
            candidate = img_dir / f"{stem}{suffix}"
            if candidate.exists():
                return candidate

    matches = [
        path for path in img_dir.glob(f"{stem}.*")
        if path.suffix.lower() in IMAGE_EXTENSIONS
    ]
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        raise RuntimeError(f"Multiple images found for {stem}: {matches}")
    raise FileNotFoundError(f"No image found for page {stem} in {img_dir}")


def parse_textline(raw: str) -> tuple[str, np.ndarray]:
    parts = raw.rstrip("\r\n").rsplit(",", 8)
    if len(parts) != 9:
        raise ValueError("Expected transcription plus eight coordinates")

    text = parts[0]
    try:
        coords = np.asarray(
            [float(value) for value in parts[1:]], dtype=np.float32
        ).reshape(4, 2)
    except ValueError as exc:
        raise ValueError("Invalid polygon coordinates") from exc
    return text, coords


def order_quad(points: np.ndarray) -> np.ndarray:
    rect = np.zeros((4, 2), dtype=np.float32)
    point_sum = points.sum(axis=1)
    point_diff = np.diff(points, axis=1).reshape(-1)
    rect[0] = points[np.argmin(point_sum)]
    rect[2] = points[np.argmax(point_sum)]
    rect[1] = points[np.argmin(point_diff)]
    rect[3] = points[np.argmax(point_diff)]
    return rect


def perspective_crop(image: np.ndarray, polygon: np.ndarray, padding: int) -> np.ndarray:
    rect = order_quad(polygon)
    tl, tr, br, bl = rect

    width_top = np.linalg.norm(tr - tl)
    width_bottom = np.linalg.norm(br - bl)
    crop_width = max(int(round(width_top)), int(round(width_bottom)), 1)

    height_left = np.linalg.norm(bl - tl)
    height_right = np.linalg.norm(br - tr)
    crop_height = max(int(round(height_left)), int(round(height_right)), 1)

    destination = np.asarray(
        [[0, 0], [crop_width - 1, 0], [crop_width - 1, crop_height - 1], [0, crop_height - 1]],
        dtype=np.float32,
    )

    matrix = cv2.getPerspectiveTransform(rect, destination)
    crop = cv2.warpPerspective(
        image,
        matrix,
        (crop_width, crop_height),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE,
    )

    if padding > 0:
        crop = cv2.copyMakeBorder(
            crop, padding, padding, padding, padding,
            cv2.BORDER_CONSTANT, value=(255, 255, 255)
        )
    return crop


def normalize_reading_direction(crop: np.ndarray) -> tuple[np.ndarray, bool]:
    height, width = crop.shape[:2]
    if height > width:
        return cv2.rotate(crop, cv2.ROTATE_90_COUNTERCLOCKWISE), True
    return crop, False


def extract_subset_and_page(raw_path: str) -> tuple[str, str]:
    normalized = raw_path.strip().replace("\\", "/")
    if not normalized:
        raise ValueError("Empty split path")

    parts = normalized.split("/")
    subset = next((name for name in SUBSETS if name in parts), None)
    if subset is None:
        raise ValueError(f"Unknown MTHv2 subset in path: {raw_path}")

    page_id = Path(parts[-1]).stem
    if not page_id:
        raise ValueError(f"Could not determine page ID: {raw_path}")
    return subset, page_id


def read_official_split(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(f"Split file not found: {path}")

    records = []
    seen = set()
    with path.open("r", encoding="utf-8-sig") as file:
        for line_number, raw in enumerate(file, start=1):
            raw = raw.strip()
            if not raw:
                continue
            try:
                subset, page_id = extract_subset_and_page(raw)
            except ValueError as exc:
                raise ValueError(f"{path}:{line_number}: {exc}") from exc

            key = (subset, page_id)
            if key in seen:
                raise ValueError(f"Duplicate page in {path}: {subset}/{page_id}")
            seen.add(key)
            records.append({"subset": subset, "page_id": page_id, "official_path": raw})

    if not records:
        raise RuntimeError(f"No pages found in split file: {path}")
    return records


def record_key(record: dict[str, str]) -> str:
    return f'{record["subset"]}/{record["page_id"]}'


def make_train_val_split(official_train, val_ratio: float, seed: int):
    if not 0.0 <= val_ratio < 1.0:
        raise ValueError("--val-ratio must satisfy 0 <= val_ratio < 1")

    records = list(official_train)
    if val_ratio == 0.0:
        return sorted(records, key=record_key), []

    rng = random.Random(seed)
    rng.shuffle(records)
    n_val = max(1, int(round(len(records) * val_ratio)))
    if n_val >= len(records):
        raise ValueError("Validation set would consume all training pages")

    val_records = records[:n_val]
    train_records = records[n_val:]
    return sorted(train_records, key=record_key), sorted(val_records, key=record_key)


def validate_no_leakage(train_records, val_records, test_records) -> None:
    train_keys = {record_key(r) for r in train_records}
    val_keys = {record_key(r) for r in val_records}
    test_keys = {record_key(r) for r in test_records}

    overlaps = {
        "train/val": train_keys & val_keys,
        "train/test": train_keys & test_keys,
        "val/test": val_keys & test_keys,
    }
    bad = {name: values for name, values in overlaps.items() if values}
    if bad:
        details = "; ".join(f"{name}: {sorted(values)[:10]}" for name, values in bad.items())
        raise RuntimeError(f"Page-level split leakage detected: {details}")


def validate_dataset_files(root: Path, records_by_split) -> None:
    missing_images = []
    missing_labels = []
    checked = set()

    for records in records_by_split.values():
        for record in records:
            key = record_key(record)
            if key in checked:
                continue
            checked.add(key)

            subset = record["subset"]
            page_id = record["page_id"]
            img_dir = root / subset / "img"
            label_path = root / subset / "label_textline" / f"{page_id}.txt"

            try:
                find_image(img_dir, page_id)
            except FileNotFoundError:
                missing_images.append(key)

            if not label_path.is_file():
                missing_labels.append(key)

    if missing_images or missing_labels:
        lines = ["Dataset validation failed."]
        if missing_images:
            lines.append(f"Missing images ({len(missing_images)}): {', '.join(missing_images[:20])}")
        if missing_labels:
            lines.append(f"Missing labels ({len(missing_labels)}): {', '.join(missing_labels[:20])}")
        raise FileNotFoundError("\n".join(lines))


def write_manifest(path: Path, rows) -> None:
    fieldnames = [
        "image_path", "text", "subset", "page_id", "line_number", "rotated",
        "original_width", "original_height", "output_width", "output_height",
    ]
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def subset_counts(records) -> dict[str, int]:
    counts = Counter(record["subset"] for record in records)
    return {subset: counts.get(subset, 0) for subset in SUBSETS}


def process_split(
    split_name: str,
    records,
    root: Path,
    output: Path,
    padding: int,
    min_side: int,
    overwrite: bool,
    skipped,
    character_counter: Counter[str],
):
    manifest_rows = []

    for record in tqdm(records, desc=f"Cropping {split_name} pages"):
        subset = record["subset"]
        page_id = record["page_id"]
        img_dir = root / subset / "img"
        label_path = root / subset / "label_textline" / f"{page_id}.txt"
        image_path = find_image(img_dir, page_id)

        pil_image = Image.open(image_path).convert("RGB")
        image = cv2.cvtColor(np.asarray(pil_image), cv2.COLOR_RGB2BGR)

        crop_dir = output / "images" / split_name / subset
        crop_dir.mkdir(parents=True, exist_ok=True)

        with label_path.open("r", encoding="utf-8-sig") as file:
            for line_number, raw in enumerate(file, start=1):
                if not raw.strip():
                    continue

                base = {
                    "split": split_name,
                    "subset": subset,
                    "page_id": page_id,
                    "line_number": line_number,
                }

                try:
                    text, polygon = parse_textline(raw)
                except ValueError as exc:
                    skipped.append({**base, "reason": f"parse_error: {exc}", "raw": raw.rstrip("\r\n")})
                    continue

                if not text.strip():
                    skipped.append({**base, "reason": "empty_transcription", "raw": raw.rstrip("\r\n")})
                    continue

                crop = perspective_crop(image, polygon, padding)
                original_height, original_width = crop.shape[:2]

                if original_width < min_side or original_height < min_side:
                    skipped.append({
                        **base,
                        "reason": f"crop_too_small:{original_width}x{original_height}",
                        "raw": raw.rstrip("\r\n"),
                    })
                    continue

                crop, rotated = normalize_reading_direction(crop)
                output_height, output_width = crop.shape[:2]

                crop_name = f"{page_id}_{line_number:04d}.png"
                crop_path = crop_dir / crop_name

                if overwrite or not crop_path.exists():
                    ok = cv2.imwrite(str(crop_path), crop)
                    if not ok:
                        skipped.append({**base, "reason": "failed_to_write_crop", "raw": raw.rstrip("\r\n")})
                        continue

                relative_path = crop_path.relative_to(output)
                manifest_rows.append({
                    "image_path": relative_path.as_posix(),
                    "text": text,
                    "subset": subset,
                    "page_id": page_id,
                    "line_number": line_number,
                    "rotated": int(rotated),
                    "original_width": original_width,
                    "original_height": original_height,
                    "output_width": output_width,
                    "output_height": output_height,
                })

                if split_name == "train":
                    character_counter.update(text)

    return manifest_rows


def main() -> None:
    args = parse_args()
    args.root = args.root.resolve()
    args.train_split = args.train_split.resolve()
    args.test_split = args.test_split.resolve()
    args.output = args.output.resolve()

    for subset in SUBSETS:
        img_dir = args.root / subset / "img"
        label_dir = args.root / subset / "label_textline"
        if not img_dir.is_dir():
            raise FileNotFoundError(f"Image directory not found: {img_dir}")
        if not label_dir.is_dir():
            raise FileNotFoundError(f"Text-line label directory not found: {label_dir}")

    official_train = read_official_split(args.train_split)
    official_test = read_official_split(args.test_split)

    official_train_keys = {record_key(r) for r in official_train}
    official_test_keys = {record_key(r) for r in official_test}
    overlap = official_train_keys & official_test_keys
    if overlap:
        raise RuntimeError(
            "Official train/test overlap detected: " + ", ".join(sorted(overlap)[:20])
        )

    train_records, val_records = make_train_val_split(
        official_train, args.val_ratio, args.seed
    )
    test_records = sorted(official_test, key=record_key)

    validate_no_leakage(train_records, val_records, test_records)

    records_by_split = {
        "train": train_records,
        "val": val_records,
        "test": test_records,
    }

    print("Validating official split against local dataset...")
    validate_dataset_files(args.root, records_by_split)
    print("Dataset validation passed.")

    args.output.mkdir(parents=True, exist_ok=True)

    manifests = {}
    skipped = []
    character_counter: Counter[str] = Counter()

    for split_name in ("train", "val", "test"):
        manifests[split_name] = process_split(
            split_name,
            records_by_split[split_name],
            args.root,
            args.output,
            args.padding,
            args.min_side,
            args.overwrite,
            skipped,
            character_counter,
        )

    for split_name, rows in manifests.items():
        write_manifest(args.output / f"{split_name}.tsv", rows)

    split_metadata = {
        "source": "HCIILAB/MTHv2_Datasets_Release official train.txt/test.txt",
        "seed": args.seed,
        "val_ratio": args.val_ratio,
        "official_train_pages": len(official_train),
        "official_test_pages": len(official_test),
        "splits": {
            split_name: {
                "num_pages": len(records),
                "subset_counts": subset_counts(records),
                "pages": [
                    {"subset": record["subset"], "page_id": record["page_id"]}
                    for record in records
                ],
            }
            for split_name, records in records_by_split.items()
        },
    }

    with (args.output / "splits.json").open("w", encoding="utf-8") as file:
        json.dump(split_metadata, file, ensure_ascii=False, indent=2)

    with (args.output / "skipped.json").open("w", encoding="utf-8") as file:
        json.dump(skipped, file, ensure_ascii=False, indent=2)

    with (args.output / "train_characters.txt").open("w", encoding="utf-8") as file:
        for char in sorted(character_counter):
            file.write(f"{char}\n")

    print()
    print("MTHv2 preparation completed")
    print("=" * 72)
    print(f"Official train pages: {len(official_train)}")
    print(f"Official test pages : {len(official_test)}")
    print(f"Validation ratio    : {args.val_ratio:.4f}")
    print(f"Seed                : {args.seed}")
    print()

    for split_name in ("train", "val", "test"):
        records = records_by_split[split_name]
        counts = subset_counts(records)
        print(
            f"{split_name:>5}: {len(records):4d} pages, "
            f"{len(manifests[split_name]):6d} crops | "
            + ", ".join(f"{subset}={counts[subset]}" for subset in SUBSETS)
        )

    print()
    print(f"Skipped crops       : {len(skipped)}")
    print(f"Train characters    : {len(character_counter)}")
    print(f"Output              : {args.output}")


if __name__ == "__main__":
    main()
