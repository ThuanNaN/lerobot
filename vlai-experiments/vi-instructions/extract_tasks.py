from __future__ import annotations

import csv
from pathlib import Path

LIBERO_SUITES = ["libero_spatial", "libero_object", "libero_goal", "libero_10"]


def extract_libero_tasks(suite_names: list[str]) -> list[dict[str, str | int]]:
    from libero.libero import benchmark

    bench = benchmark.get_benchmark_dict()
    rows: list[dict[str, str | int]] = []
    for suite_name in suite_names:
        suite = bench[suite_name]()
        for task_id, task in enumerate(suite.tasks):
            rows.append({"suite": suite_name, "task_id": task_id, "english": task.language})
    return rows


def write_tasks_csv(rows: list[dict[str, str | int]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["suite", "task_id", "english", "vietnamese"])
        writer.writeheader()
        for row in rows:
            writer.writerow({**row, "vietnamese": ""})


if __name__ == "__main__":
    rows = extract_libero_tasks(LIBERO_SUITES)
    write_tasks_csv(rows, Path("vlai-experiments/vi-instructions/data/tasks_en.csv"))
    print(f"Wrote {len(rows)} rows to vlai-experiments/vi-instructions/data/tasks_en.csv")
