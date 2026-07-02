import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest
from validate_translations import validate_translations


def _write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    import csv

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["suite", "task_id", "english", "vietnamese"])
        writer.writeheader()
        writer.writerows(rows)


def test_validate_translations_passes_for_fully_translated_matching_rows(tmp_path):
    en_rows = [{"suite": "libero_spatial", "task_id": "0", "english": "pick up the bowl", "vietnamese": ""}]
    vi_rows = [
        {
            "suite": "libero_spatial",
            "task_id": "0",
            "english": "pick up the bowl",
            "vietnamese": "nhấc cái bát lên",
        }
    ]
    en_csv, vi_csv = tmp_path / "en.csv", tmp_path / "vi.csv"
    _write_csv(en_csv, en_rows)
    _write_csv(vi_csv, vi_rows)

    validate_translations(en_csv, vi_csv)  # must not raise


def test_validate_translations_rejects_row_count_mismatch(tmp_path):
    en_rows = [
        {"suite": "libero_spatial", "task_id": "0", "english": "pick up the bowl", "vietnamese": ""},
        {"suite": "libero_spatial", "task_id": "1", "english": "open the drawer", "vietnamese": ""},
    ]
    vi_rows = [
        {
            "suite": "libero_spatial",
            "task_id": "0",
            "english": "pick up the bowl",
            "vietnamese": "nhấc cái bát lên",
        }
    ]
    en_csv, vi_csv = tmp_path / "en.csv", tmp_path / "vi.csv"
    _write_csv(en_csv, en_rows)
    _write_csv(vi_csv, vi_rows)

    with pytest.raises(ValueError, match="row count"):
        validate_translations(en_csv, vi_csv)


def test_validate_translations_rejects_empty_vietnamese_cell(tmp_path):
    en_rows = [{"suite": "libero_spatial", "task_id": "0", "english": "pick up the bowl", "vietnamese": ""}]
    vi_rows = [{"suite": "libero_spatial", "task_id": "0", "english": "pick up the bowl", "vietnamese": ""}]
    en_csv, vi_csv = tmp_path / "en.csv", tmp_path / "vi.csv"
    _write_csv(en_csv, en_rows)
    _write_csv(vi_csv, vi_rows)

    with pytest.raises(ValueError, match="empty"):
        validate_translations(en_csv, vi_csv)


def test_validate_translations_rejects_suite_task_id_mismatch(tmp_path):
    en_rows = [{"suite": "libero_spatial", "task_id": "0", "english": "pick up the bowl", "vietnamese": ""}]
    vi_rows = [{"suite": "libero_object", "task_id": "0", "english": "pick up the bowl", "vietnamese": "x"}]
    en_csv, vi_csv = tmp_path / "en.csv", tmp_path / "vi.csv"
    _write_csv(en_csv, en_rows)
    _write_csv(vi_csv, vi_rows)

    with pytest.raises(ValueError, match="suite/task_id"):
        validate_translations(en_csv, vi_csv)
