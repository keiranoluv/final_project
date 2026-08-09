#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import random
from collections import Counter
from pathlib import Path


TEXT_CANDIDATES = [
    "text",
    "label",
    "ground_truth",
    "transcription",
    "gt",
]


def parse_args():
    p = argparse.ArgumentParser(
        description="Create a rare-character-aware oversampled training TSV."
    )
    p.add_argument("--train-tsv", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--text-column", type=str, default=None)
    p.add_argument("--rare-threshold", type=int, default=50)
    p.add_argument(
        "--oversample-factor",
        type=int,
        default=3,
        help="Total copies for a line containing rare chars. 3 means original + 2 duplicates.",
    )
    p.add_argument(
        "--max-rare-factor",
        type=int,
        default=None,
        help=(
            "Optional stronger cap based on number of rare chars in a line. "
            "If omitted, every rare line uses --oversample-factor."
        ),
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--shuffle",
        action="store_true",
        help="Shuffle final oversampled TSV deterministically using --seed.",
    )
    return p.parse_args()


def detect_text_column(fieldnames, requested=None):
    if requested:
        if requested not in fieldnames:
            raise ValueError(
                f"text column {requested!r} not found. Available: {fieldnames}"
            )
        return requested

    for c in TEXT_CANDIDATES:
        if c in fieldnames:
            return c

    raise ValueError(
        "Could not auto-detect text column. "
        f"Available columns: {fieldnames}. Use --text-column."
    )


def load_tsv(path: Path):
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        if not reader.fieldnames:
            raise ValueError(f"No header found in {path}")
        rows = list(reader)
        return reader.fieldnames, rows


def write_tsv(path: Path, fieldnames, rows):
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
            delimiter="\t",
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    fieldnames, rows = load_tsv(args.train_tsv)
    text_col = detect_text_column(fieldnames, args.text_column)

    char_freq = Counter()
    for row in rows:
        char_freq.update(row[text_col])

    rare_chars = {
        ch for ch, freq in char_freq.items()
        if freq <= args.rare_threshold
    }

    enriched = []
    rare_line_count = 0
    duplicate_rows_added = 0
    rare_char_occurrences_in_selected_lines = Counter()

    for idx, row in enumerate(rows):
        text = row[text_col]
        rare_in_line = [ch for ch in text if ch in rare_chars]
        unique_rare = sorted(set(rare_in_line))

        if unique_rare:
            rare_line_count += 1
            rare_char_occurrences_in_selected_lines.update(rare_in_line)

            factor = args.oversample_factor
            if args.max_rare_factor is not None:
                # Optional dynamic factor:
                # 1 rare char => base factor
                # more rare chars => stronger oversampling, capped.
                factor = min(
                    args.max_rare_factor,
                    args.oversample_factor + max(0, len(unique_rare) - 1),
                )
        else:
            factor = 1

        for copy_idx in range(factor):
            enriched.append(dict(row))
            if copy_idx > 0:
                duplicate_rows_added += 1

    if args.shuffle:
        rng = random.Random(args.seed)
        rng.shuffle(enriched)

    out_tsv = args.output_dir / "train_rare_oversampled.tsv"
    write_tsv(out_tsv, fieldnames, enriched)

    char_rows = []
    for ch, freq in sorted(char_freq.items(), key=lambda kv: (kv[1], kv[0])):
        char_rows.append({
            "char": ch,
            "train_freq": freq,
            "is_rare": freq <= args.rare_threshold,
            "selected_line_occurrences": rare_char_occurrences_in_selected_lines[ch],
        })

    char_csv = args.output_dir / "rare_character_frequencies.csv"
    with char_csv.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "char",
                "train_freq",
                "is_rare",
                "selected_line_occurrences",
            ],
        )
        writer.writeheader()
        writer.writerows(char_rows)

    original_n = len(rows)
    final_n = len(enriched)

    summary = {
        "train_tsv": str(args.train_tsv),
        "text_column": text_col,
        "rare_threshold": args.rare_threshold,
        "oversample_factor": args.oversample_factor,
        "max_rare_factor": args.max_rare_factor,
        "shuffle": args.shuffle,
        "seed": args.seed,
        "original_train_lines": original_n,
        "rare_train_lines": rare_line_count,
        "rare_train_line_percent": 100.0 * rare_line_count / original_n if original_n else 0.0,
        "unique_train_characters": len(char_freq),
        "unique_rare_characters": len(rare_chars),
        "duplicate_rows_added": duplicate_rows_added,
        "final_train_lines": final_n,
        "dataset_expansion_factor": final_n / original_n if original_n else 0.0,
        "output_tsv": str(out_tsv),
    }

    with (args.output_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("=" * 78)
    print("Rare-character-aware training manifest")
    print("=" * 78)
    print(f"Train TSV                 : {args.train_tsv}")
    print(f"Text column               : {text_col}")
    print(f"Rare threshold            : <= {args.rare_threshold}")
    print(f"Oversample factor         : {args.oversample_factor}")
    print(f"Original train lines      : {original_n:,}")
    print(
        f"Rare-containing lines     : {rare_line_count:,} "
        f"({summary['rare_train_line_percent']:.2f}%)"
    )
    print(f"Unique train characters   : {len(char_freq):,}")
    print(f"Unique rare characters    : {len(rare_chars):,}")
    print(f"Duplicate rows added      : {duplicate_rows_added:,}")
    print(f"Final train lines         : {final_n:,}")
    print(
        f"Dataset expansion factor  : {summary['dataset_expansion_factor']:.3f}x"
    )
    print()
    print(f"Output TSV                : {out_tsv}")
    print(f"Summary                   : {args.output_dir / 'summary.json'}")
    print(f"Rare frequency table      : {char_csv}")


if __name__ == "__main__":
    main()
