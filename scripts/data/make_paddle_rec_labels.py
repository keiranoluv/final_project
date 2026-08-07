#!/usr/bin/env python3

import csv
from pathlib import Path


DATASET_ROOT = Path("data/processed/MTH1000_B0")
OUTPUT_ROOT = DATASET_ROOT / "paddle_labels"


def convert(manifest_name: str, output_name: str):
    src = DATASET_ROOT / manifest_name
    dst = OUTPUT_ROOT / output_name

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    count = 0

    with src.open("r", encoding="utf-8") as fin, \
         dst.open("w", encoding="utf-8") as fout:

        reader = csv.DictReader(fin, delimiter="\t")

        for row in reader:
            image_path = row["image_path"]
            text = row["text"]

            fout.write(f"{image_path}\t{text}\n")
            count += 1

    print(f"{manifest_name}: {count} samples -> {dst}")


def main():
    convert("train.tsv", "train.txt")
    convert("val.tsv", "val.txt")
    convert("test.tsv", "test.txt")


if __name__ == "__main__":
    main()