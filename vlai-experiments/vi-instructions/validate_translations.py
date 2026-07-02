from __future__ import annotations

import csv
from pathlib import Path


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def validate_translations(en_csv: Path, vi_csv: Path) -> None:
    en_rows = _read_rows(en_csv)
    vi_rows = _read_rows(vi_csv)

    if len(en_rows) != len(vi_rows):
        raise ValueError(f"row count mismatch: {en_csv} has {len(en_rows)}, {vi_csv} has {len(vi_rows)}")

    en_keys = [(r["suite"], r["task_id"]) for r in en_rows]
    vi_keys = [(r["suite"], r["task_id"]) for r in vi_rows]
    if en_keys != vi_keys:
        raise ValueError(f"suite/task_id order/content mismatch between {en_csv} and {vi_csv}")

    empty = [r for r in vi_rows if not r["vietnamese"].strip()]
    if empty:
        raise ValueError(f"empty vietnamese translation for {len(empty)} row(s), e.g. {empty[0]}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--en-csv", type=Path, default=Path("vlai-experiments/vi-instructions/data/tasks_en.csv")
    )
    parser.add_argument(
        "--vi-csv", type=Path, default=Path("vlai-experiments/vi-instructions/data/tasks_vi.csv")
    )
    args = parser.parse_args()
    validate_translations(args.en_csv, args.vi_csv)
    print("OK: translations are well-formed.")
