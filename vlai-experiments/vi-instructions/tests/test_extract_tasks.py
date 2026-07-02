import csv
import sys
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from extract_tasks import extract_libero_tasks, write_tasks_csv

pytest_plugins = []


class _FakeTask:
    def __init__(self, language: str):
        self.language = language


class _FakeSuite:
    def __init__(self, languages: list[str]):
        self.tasks = [_FakeTask(lang) for lang in languages]


def test_extract_libero_tasks_builds_rows_per_suite():
    fake_bench = {
        "libero_spatial": lambda: _FakeSuite(["pick up the bowl", "open the drawer"]),
        "libero_object": lambda: _FakeSuite(["turn on the stove"]),
    }
    with patch("libero.libero.benchmark.get_benchmark_dict", return_value=fake_bench):
        rows = extract_libero_tasks(["libero_spatial", "libero_object"])

    assert rows == [
        {"suite": "libero_spatial", "task_id": 0, "english": "pick up the bowl"},
        {"suite": "libero_spatial", "task_id": 1, "english": "open the drawer"},
        {"suite": "libero_object", "task_id": 0, "english": "turn on the stove"},
    ]


def test_write_tasks_csv_adds_empty_vietnamese_column(tmp_path):
    rows = [{"suite": "libero_spatial", "task_id": 0, "english": "pick up the bowl"}]
    out_path = tmp_path / "tasks_en.csv"

    write_tasks_csv(rows, out_path)

    with out_path.open(newline="", encoding="utf-8") as f:
        reader = list(csv.DictReader(f))
    assert reader == [
        {"suite": "libero_spatial", "task_id": "0", "english": "pick up the bowl", "vietnamese": ""}
    ]
