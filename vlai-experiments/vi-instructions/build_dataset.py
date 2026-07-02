from __future__ import annotations

import csv
import shutil
from pathlib import Path

import pandas as pd

from lerobot.datasets.io_utils import load_tasks, write_tasks


def load_translations(csv_path: Path) -> dict[str, str]:
    translations: dict[str, str] = {}
    with csv_path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            english = row["english"]
            vietnamese = row["vietnamese"]
            if not vietnamese.strip():
                raise ValueError(f"Empty Vietnamese translation for: {english!r}")
            translations[english] = vietnamese
    return translations


def translate_tasks_df(tasks_df: pd.DataFrame, translations: dict[str, str]) -> pd.DataFrame:
    missing = [t for t in tasks_df.index if t not in translations]
    if missing:
        raise KeyError(f"No Vietnamese translation for tasks: {missing}")
    return tasks_df.rename(index=translations)


def translate_episode_tasks(task_list: list[str], translations: dict[str, str]) -> list[str]:
    missing = [t for t in task_list if t not in translations]
    if missing:
        raise KeyError(f"No Vietnamese translation for tasks: {missing}")
    return [translations[t] for t in task_list]


def fork_dataset(src_root: Path, dst_root: Path, translations: dict[str, str]) -> None:
    dst_root.mkdir(parents=True, exist_ok=True)
    dst_meta = dst_root / "meta"
    dst_meta.mkdir(exist_ok=True)

    for entry in src_root.iterdir():
        if entry.name == "meta":
            continue
        dst_link = dst_root / entry.name
        if not dst_link.exists():
            dst_link.symlink_to(entry.resolve())

    shutil.copy2(src_root / "meta" / "info.json", dst_meta / "info.json")
    stats_path = src_root / "meta" / "stats.json"
    if stats_path.exists():
        shutil.copy2(stats_path, dst_meta / "stats.json")

    tasks_df = load_tasks(src_root)
    write_tasks(translate_tasks_df(tasks_df, translations), dst_root)

    src_episodes_dir = src_root / "meta" / "episodes"
    dst_episodes_dir = dst_meta / "episodes"
    for src_file in src_episodes_dir.rglob("*.parquet"):
        rel = src_file.relative_to(src_episodes_dir)
        dst_file = dst_episodes_dir / rel
        dst_file.parent.mkdir(parents=True, exist_ok=True)
        ep_df = pd.read_parquet(src_file)
        ep_df["tasks"] = ep_df["tasks"].apply(lambda tasks: translate_episode_tasks(tasks, translations))
        ep_df.to_parquet(dst_file)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--src-root", type=Path, required=True)
    parser.add_argument("--dst-root", type=Path, required=True)
    parser.add_argument(
        "--translations-csv", type=Path, default=Path("vlai-experiments/vi-instructions/data/tasks_vi.csv")
    )
    args = parser.parse_args()

    translations = load_translations(args.translations_csv)
    fork_dataset(args.src_root, args.dst_root, translations)
    print(f"Wrote Vietnamese dataset fork to {args.dst_root}")
