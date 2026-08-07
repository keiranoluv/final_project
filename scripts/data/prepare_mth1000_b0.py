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
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".tif",
    ".tiff",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Split MTH1000 by page and crop text-line polygons "
            "for the PP-OCRv5 zero-shot baseline."
        )
    )
    parser.add_argument("--root", type=Path, default=Path("mthv2/MTH1000"))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/processed/MTH1000_B0"),
    )
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--train-pages", type=int, default=800)
    parser.add_argument("--val-pages", type=int, default=100)
    parser.add_argument("--test-pages", type=int, default=100)
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
    raise FileNotFoundError(f"No image found for page: {stem}")


def parse_textline(raw: str) -> tuple[str, np.ndarray]:
    parts = raw.rstrip("\r\n").rsplit(",", 8)
    if len(parts) != 9:
        raise ValueError("Expected transcription plus eight coordinates")

    text = parts[0]
    try:
        coords = np.asarray(
            [float(value) for value in parts[1:]],
            dtype=np.float32,
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


def perspective_crop(
    image: np.ndarray,
    polygon: np.ndarray,
    padding: int,
) -> np.ndarray:
    rect = order_quad(polygon)
    tl, tr, br, bl = rect

    width_top = np.linalg.norm(tr - tl)
    width_bottom = np.linalg.norm(br - bl)
    crop_width = max(int(round(width_top)), int(round(width_bottom)), 1)

    height_left = np.linalg.norm(bl - tl)
    height_right = np.linalg.norm(br - tr)
    crop_height = max(int(round(height_left)), int(round(height_right)), 1)

    destination = np.asarray(
        [
            [0, 0],
            [crop_width - 1, 0],
            [crop_width - 1, crop_height - 1],
            [0, crop_height - 1],
        ],
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
            crop,
            padding,
            padding,
            padding,
            padding,
            cv2.BORDER_CONSTANT,
            value=(255, 255, 255),
        )

    return crop


def normalize_reading_direction(crop: np.ndarray) -> tuple[np.ndarray, bool]:
    height, width = crop.shape[:2]
    if height > width:
        # MTH text lines are usually vertical, top-to-bottom.
        # Counter-clockwise rotation maps original top to output left.
        return cv2.rotate(crop, cv2.ROTATE_90_COUNTERCLOCKWISE), True
    return crop, False


def split_pages(
    page_stems: list[str],
    train_pages: int,
    val_pages: int,
    test_pages: int,
    seed: int,
) -> dict[str, list[str]]:
    requested = train_pages + val_pages + test_pages
    if requested != len(page_stems):
        raise ValueError(
            f"Requested {requested} pages, but dataset contains "
            f"{len(page_stems)} pages."
        )

    pages = sorted(page_stems)
    random.Random(seed).shuffle(pages)

    train_end = train_pages
    val_end = train_end + val_pages

    return {
        "train": sorted(pages[:train_end]),
        "val": sorted(pages[train_end:val_end]),
        "test": sorted(pages[val_end:]),
    }


def write_manifest(path: Path, rows: list[dict[str, object]]) -> None:
    fieldnames = [
        "image_path",
        "text",
        "page_id",
        "line_number",
        "rotated",
        "original_width",
        "original_height",
        "output_width",
        "output_height",
    ]

    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()

    img_dir = args.root / "img"
    label_dir = args.root / "label_textline"

    if not img_dir.is_dir():
        raise FileNotFoundError(f"Image directory not found: {img_dir}")
    if not label_dir.is_dir():
        raise FileNotFoundError(f"Label directory not found: {label_dir}")

    label_paths = sorted(label_dir.glob("*.txt"))
    if not label_paths:
        raise RuntimeError(f"No text-line labels found in: {label_dir}")

    page_stems = [path.stem for path in label_paths]
    splits = split_pages(
        page_stems=page_stems,
        train_pages=args.train_pages,
        val_pages=args.val_pages,
        test_pages=args.test_pages,
        seed=args.seed,
    )

    args.output.mkdir(parents=True, exist_ok=True)

    split_lookup = {
        page_id: split_name
        for split_name, pages in splits.items()
        for page_id in pages
    }

    manifests: dict[str, list[dict[str, object]]] = {
        "train": [],
        "val": [],
        "test": [],
    }
    skipped: list[dict[str, object]] = []
    character_counter: Counter[str] = Counter()

    for label_path in tqdm(label_paths, desc="Cropping pages"):
        page_id = label_path.stem
        split_name = split_lookup[page_id]
        image_path = find_image(img_dir, page_id)

        pil_image = Image.open(image_path).convert("RGB")
        image = cv2.cvtColor(np.asarray(pil_image), cv2.COLOR_RGB2BGR)

        crop_dir = args.output / "images" / split_name
        crop_dir.mkdir(parents=True, exist_ok=True)

        with label_path.open("r", encoding="utf-8-sig") as file:
            for line_number, raw in enumerate(file, start=1):
                if not raw.strip():
                    continue

                try:
                    text, polygon = parse_textline(raw)
                except ValueError as exc:
                    skipped.append(
                        {
                            "page_id": page_id,
                            "line_number": line_number,
                            "reason": f"parse_error: {exc}",
                            "raw": raw.rstrip("\r\n"),
                        }
                    )
                    continue

                if not text.strip():
                    skipped.append(
                        {
                            "page_id": page_id,
                            "line_number": line_number,
                            "reason": "empty_transcription",
                            "raw": raw.rstrip("\r\n"),
                        }
                    )
                    continue

                crop = perspective_crop(image, polygon, args.padding)
                original_height, original_width = crop.shape[:2]

                if original_width < args.min_side or original_height < args.min_side:
                    skipped.append(
                        {
                            "page_id": page_id,
                            "line_number": line_number,
                            "reason": f"crop_too_small:{original_width}x{original_height}",
                            "raw": raw.rstrip("\r\n"),
                        }
                    )
                    continue

                crop, rotated = normalize_reading_direction(crop)
                output_height, output_width = crop.shape[:2]

                crop_name = f"{page_id}_{line_number:04d}.png"
                crop_path = crop_dir / crop_name

                if args.overwrite or not crop_path.exists():
                    ok = cv2.imwrite(str(crop_path), crop)
                    if not ok:
                        skipped.append(
                            {
                                "page_id": page_id,
                                "line_number": line_number,
                                "reason": "failed_to_write_crop",
                                "raw": raw.rstrip("\r\n"),
                            }
                        )
                        continue

                relative_path = crop_path.relative_to(args.output)
                manifests[split_name].append(
                    {
                        "image_path": relative_path.as_posix(),
                        "text": text,
                        "page_id": page_id,
                        "line_number": line_number,
                        "rotated": int(rotated),
                        "original_width": original_width,
                        "original_height": original_height,
                        "output_width": output_width,
                        "output_height": output_height,
                    }
                )

                if split_name == "train":
                    character_counter.update(text)

    for split_name, rows in manifests.items():
        write_manifest(args.output / f"{split_name}.tsv", rows)

    with (args.output / "splits.json").open("w", encoding="utf-8") as file:
        json.dump({"seed": args.seed, "splits": splits}, file, ensure_ascii=False, indent=2)

    with (args.output / "skipped.json").open("w", encoding="utf-8") as file:
        json.dump(skipped, file, ensure_ascii=False, indent=2)

    with (args.output / "train_characters.txt").open("w", encoding="utf-8") as file:
        for char in sorted(character_counter):
            file.write(f"{char}\n")

    print()
    print("Preparation completed")
    print("=" * 60)
    for split_name in ("train", "val", "test"):
        print(
            f"{split_name:>5}: "
            f"{len(splits[split_name]):4d} pages, "
            f"{len(manifests[split_name]):6d} crops"
        )
    print(f"Skipped: {len(skipped)}")
    print(f"Train characters: {len(character_counter)}")
    print(f"Output: {args.output.resolve()}")


if __name__ == "__main__":
    main()
