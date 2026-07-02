import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import pytest
from build_dataset import fork_dataset, load_translations, translate_episode_tasks, translate_tasks_df


def test_translate_tasks_df_renames_index_preserving_task_index():
    tasks_df = pd.DataFrame(
        {"task_index": [0, 1]}, index=pd.Index(["pick up the bowl", "open the drawer"], name="task")
    )
    translations = {"pick up the bowl": "nhấc cái bát lên", "open the drawer": "mở ngăn kéo"}

    translated = translate_tasks_df(tasks_df, translations)

    assert list(translated.index) == ["nhấc cái bát lên", "mở ngăn kéo"]
    assert list(translated["task_index"]) == [0, 1]


def test_translate_tasks_df_raises_on_missing_translation():
    tasks_df = pd.DataFrame({"task_index": [0]}, index=pd.Index(["pick up the bowl"], name="task"))
    with pytest.raises(KeyError):
        translate_tasks_df(tasks_df, {})


def test_translate_episode_tasks_maps_each_entry():
    result = translate_episode_tasks(["pick up the bowl"], {"pick up the bowl": "nhấc cái bát lên"})
    assert result == ["nhấc cái bát lên"]


def test_load_translations_rejects_empty_vietnamese_cell(tmp_path):
    csv_path = tmp_path / "tasks_vi.csv"
    csv_path.write_text(
        "suite,task_id,english,vietnamese\nlibero_spatial,0,pick up the bowl,\n", encoding="utf-8"
    )
    with pytest.raises(ValueError):
        load_translations(csv_path)


def _write_fake_dataset(root: Path) -> None:
    (root / "data").mkdir(parents=True)
    (root / "data" / "dummy.parquet").write_text("dummy")
    meta = root / "meta"
    meta.mkdir()
    (meta / "info.json").write_text(json.dumps({"repo_id": "fake/libero"}))
    (meta / "stats.json").write_text(json.dumps({}))

    from lerobot.datasets.io_utils import write_tasks

    tasks_df = pd.DataFrame({"task_index": [0]}, index=pd.Index(["pick up the bowl"], name="task"))
    write_tasks(tasks_df, root)

    episodes_dir = meta / "episodes" / "chunk-000"
    episodes_dir.mkdir(parents=True)
    ep_df = pd.DataFrame({"episode_index": [0], "tasks": [["pick up the bowl"]]})
    ep_df.to_parquet(episodes_dir / "file-000.parquet")


def test_fork_dataset_symlinks_data_and_translates_tasks(tmp_path):
    src_root, dst_root = tmp_path / "src", tmp_path / "dst"
    _write_fake_dataset(src_root)
    translations = {"pick up the bowl": "nhấc cái bát lên"}

    fork_dataset(src_root, dst_root, translations)

    assert (dst_root / "data").is_symlink()

    from lerobot.datasets.io_utils import load_tasks

    translated_tasks = load_tasks(dst_root)
    assert list(translated_tasks.index) == ["nhấc cái bát lên"]

    ep_df = pd.read_parquet(dst_root / "meta" / "episodes" / "chunk-000" / "file-000.parquet")
    assert ep_df["tasks"].iloc[0] == ["nhấc cái bát lên"]
