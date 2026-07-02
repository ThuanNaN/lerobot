# SmolVLA Vietnamese Instruction-Following Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make SmolVLA understand Vietnamese LIBERO task instructions (input-only — no
Vietnamese text generation) via LoRA finetuning, and make `lerobot-eval` actually able to
exercise that in Vietnamese, not just at training time.

**Architecture:** Stage 0 diagnostics (tokenizer fertility + embedding sanity, no training) →
Stage 1 Vietnamese LIBERO dataset fork (translate task strings, reuse episodes/video via
symlink) → Stage 1b eval-side instruction override (LIBERO's `lerobot-eval` rollout currently
reads task text from the upstream `libero` package, not our dataset — needs a code change) →
Stage 2 widened-LoRA finetune (SmolVLA's default LoRA targets only touch the action expert,
not the VLM's language layers) → Stage 3 eval + comparison report.

**Tech Stack:** Python 3.12, PyTorch, Hugging Face `transformers`/`peft`, `libero` (LIBERO
benchmark), pandas/pyarrow, pytest, `uv`.

## Global Constraints

- All Python execution goes through `uv run` (per `CLAUDE.md`).
- `src/lerobot/envs/` is under CLAUDE.md's strict-mypy list — new/changed code there needs
  full type annotations; run `uv run mypy src/lerobot/envs/libero.py src/lerobot/envs/configs.py`
  after Task 4.
- No new dependencies beyond what `--extra smolvla --extra libero` already provides.
- New standalone research scripts live under `vlai-experiments/vi-instructions/` (root-level,
  sibling to `run.sh`, kept separate from both the `lerobot` package and the `vlai-docs`
  curriculum — per user decision). Because that directory name contains a hyphen it cannot be
  a dotted Python package; test files there import sibling modules via
  `sys.path.insert(0, str(Path(__file__).resolve().parents[1]))` then a plain
  `import <module_name>` (see Task 1 for the exact pattern), not `import vlai_experiments...`.
- Code changes to `src/lerobot/envs/libero.py` / `configs.py` (Task 4) get tests under the
  main `tests/envs/` tree, following existing repo conventions, and must be additive: omitting
  the new parameters must reproduce today's exact behavior (existing English-instruction eval
  runs are unaffected).
- Source spec: `docs/superpowers/specs/2026-07-02-smolvla-vietnamese-instructions-design.md`.

---

### Task 1: Stage 0 — Vietnamese diagnostics (tokenizer fertility + embedding sanity)

**Files:**
- Create: `vlai-experiments/vi-instructions/diagnostics.py`
- Test: `vlai-experiments/vi-instructions/tests/test_diagnostics.py`

**Interfaces:**
- Consumes: nothing (first task, no dependencies).
- Produces: `mean_fertility(tokenizer, sentences: list[str]) -> float`,
  `pooled_embedding(text_model, tokenizer, sentence: str) -> torch.Tensor`,
  `paraphrase_gap(text_model, tokenizer, pairs: list[tuple[str, str]]) -> dict` — reusable by
  nothing downstream in this plan (Stage 0 is a standalone gate a human reads before deciding
  whether to proceed), but keep the functions pure/importable in case Stage 4 escalation needs
  them again later.

- [ ] **Step 1: Write the failing tests**

```python
# vlai-experiments/vi-instructions/tests/test_diagnostics.py
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest
import torch

from diagnostics import mean_fertility, paraphrase_gap, pooled_embedding, tokens_per_word


class _StubTokenizer:
    def __call__(self, sentence: str, return_tensors: str | None = None):
        token_ids = [ord(c) % 50 + 1 for c in sentence.replace(" ", "")]
        if return_tensors == "pt":
            input_ids = torch.tensor([token_ids])
            attention_mask = torch.ones_like(input_ids)
            return {"input_ids": input_ids, "attention_mask": attention_mask}
        return {"input_ids": token_ids}


class _StubOutput:
    def __init__(self, last_hidden_state: torch.Tensor):
        self.last_hidden_state = last_hidden_state


class _StubTextModel:
    hidden_size = 4

    def __call__(self, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        torch.manual_seed(int(input_ids.sum().item()))
        hidden = torch.randn(1, input_ids.shape[1], self.hidden_size)
        return _StubOutput(hidden)


def test_tokens_per_word_counts_ratio():
    tokenizer = _StubTokenizer()
    ratio = tokens_per_word(tokenizer, "hello world")
    assert ratio == len(tokenizer("hello world")["input_ids"]) / 2


def test_tokens_per_word_rejects_empty_sentence():
    with pytest.raises(ValueError):
        tokens_per_word(_StubTokenizer(), "")


def test_mean_fertility_averages_across_sentences():
    tokenizer = _StubTokenizer()
    sentences = ["hello world", "pick up the bowl"]
    expected = sum(tokens_per_word(tokenizer, s) for s in sentences) / len(sentences)
    assert mean_fertility(tokenizer, sentences) == expected


def test_pooled_embedding_has_hidden_size_shape():
    vec = pooled_embedding(_StubTextModel(), _StubTokenizer(), "hello")
    assert vec.shape == (_StubTextModel.hidden_size,)


def test_paraphrase_gap_returns_matched_and_mismatched_means():
    pairs = [("hello", "xin chao"), ("world", "the gioi")]
    result = paraphrase_gap(_StubTextModel(), _StubTokenizer(), pairs)
    assert set(result.keys()) == {"matched_mean", "mismatched_mean", "gap"}
    assert -1.0 <= result["matched_mean"] <= 1.0
    assert -1.0 <= result["mismatched_mean"] <= 1.0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest vlai-experiments/vi-instructions/tests/test_diagnostics.py -v`
Expected: FAIL/ERROR with `ModuleNotFoundError: No module named 'diagnostics'`.

- [ ] **Step 3: Write the implementation**

```python
# vlai-experiments/vi-instructions/diagnostics.py
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

# Representative LIBERO-style imperative instructions, used only for this cheap
# sanity check — NOT the reviewed Stage 1 translations (those come from Task 2/3).
EN_VI_PROBE_PAIRS: list[tuple[str, str]] = [
    ("pick up the black bowl and place it in the tray", "nhấc cái bát đen lên và đặt nó vào khay"),
    ("open the top drawer and put the bowl inside", "mở ngăn kéo trên cùng và đặt cái bát vào bên trong"),
    ("turn on the stove", "bật bếp lên"),
    ("put the red mug on the plate", "đặt cốc màu đỏ lên đĩa"),
    ("close the microwave door", "đóng cửa lò vi sóng lại"),
    ("push the plate to the front of the stove", "đẩy cái đĩa ra phía trước bếp"),
    ("pick up the alphabet soup and put it in the basket", "nhấc hộp súp chữ cái lên và đặt vào giỏ"),
    ("put the wine bottle on the rack", "đặt chai rượu vang lên giá"),
]


def tokens_per_word(tokenizer, sentence: str) -> float:
    n_words = len(sentence.split())
    if n_words == 0:
        raise ValueError("sentence must contain at least one word")
    n_tokens = len(tokenizer(sentence)["input_ids"])
    return n_tokens / n_words


def mean_fertility(tokenizer, sentences: list[str]) -> float:
    return sum(tokens_per_word(tokenizer, s) for s in sentences) / len(sentences)


def pooled_embedding(text_model, tokenizer, sentence: str) -> torch.Tensor:
    encoded = tokenizer(sentence, return_tensors="pt")
    with torch.no_grad():
        out = text_model(input_ids=encoded["input_ids"], attention_mask=encoded["attention_mask"])
    mask = encoded["attention_mask"].unsqueeze(-1).to(out.last_hidden_state.dtype)
    summed = (out.last_hidden_state * mask).sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1)
    return (summed / counts).squeeze(0)


def paraphrase_gap(text_model, tokenizer, pairs: list[tuple[str, str]]) -> dict:
    embeddings_en = [pooled_embedding(text_model, tokenizer, en) for en, _ in pairs]
    embeddings_vi = [pooled_embedding(text_model, tokenizer, vi) for _, vi in pairs]

    matched_sims = [
        torch.nn.functional.cosine_similarity(embeddings_en[i], embeddings_vi[i], dim=0).item()
        for i in range(len(pairs))
    ]
    mismatched_sims = [
        torch.nn.functional.cosine_similarity(embeddings_en[i], embeddings_vi[j], dim=0).item()
        for i in range(len(pairs))
        for j in range(len(pairs))
        if i != j
    ]
    matched_mean = sum(matched_sims) / len(matched_sims)
    mismatched_mean = sum(mismatched_sims) / len(mismatched_sims)
    return {"matched_mean": matched_mean, "mismatched_mean": mismatched_mean, "gap": matched_mean - mismatched_mean}


def main(model_id: str, output_path: Path) -> None:
    from transformers import AutoModelForImageTextToText, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForImageTextToText.from_pretrained(model_id, torch_dtype="bfloat16")
    text_model = model.model.text_model

    en_sentences = [en for en, _ in EN_VI_PROBE_PAIRS]
    vi_sentences = [vi for _, vi in EN_VI_PROBE_PAIRS]

    report = {
        "model_id": model_id,
        "fertility_en": mean_fertility(tokenizer, en_sentences),
        "fertility_vi": mean_fertility(tokenizer, vi_sentences),
        "paraphrase_gap": paraphrase_gap(text_model, tokenizer, EN_VI_PROBE_PAIRS),
    }
    report["fertility_ratio_vi_over_en"] = report["fertility_vi"] / report["fertility_en"]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", default="HuggingFaceTB/SmolVLM2-500M-Video-Instruct")
    parser.add_argument(
        "--output", type=Path, default=Path("vlai-experiments/vi-instructions/data/stage0_diagnostics.json")
    )
    args = parser.parse_args()
    main(args.model_id, args.output)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest vlai-experiments/vi-instructions/tests/test_diagnostics.py -v`
Expected: PASS (5 tests).

- [ ] **Step 5: Run the real diagnostic against the actual backbone (manual, needs `--extra smolvla`)**

Run: `uv run python vlai-experiments/vi-instructions/diagnostics.py`
Expected: downloads `HuggingFaceTB/SmolVLM2-500M-Video-Instruct` (~1GB) and prints/writes
`vlai-experiments/vi-instructions/data/stage0_diagnostics.json` with `fertility_ratio_vi_over_en`
and `paraphrase_gap.gap`. Read these per the spec's Stage 0 gate before proceeding.

- [ ] **Step 6: Commit**

```bash
git add vlai-experiments/vi-instructions/diagnostics.py vlai-experiments/vi-instructions/tests/test_diagnostics.py
git commit -m "feat(vi-instructions): add Stage 0 Vietnamese tokenizer/embedding diagnostics"
```

---

### Task 2: Extract canonical LIBERO English task strings + translation review scaffold

**Files:**
- Create: `vlai-experiments/vi-instructions/extract_tasks.py`
- Create: `vlai-experiments/vi-instructions/validate_translations.py`
- Test: `vlai-experiments/vi-instructions/tests/test_extract_tasks.py`
- Test: `vlai-experiments/vi-instructions/tests/test_validate_translations.py`

**Interfaces:**
- Consumes: nothing.
- Produces: CSV schema `suite,task_id,english,vietnamese` at
  `vlai-experiments/vi-instructions/data/tasks_en.csv` (empty `vietnamese` column) — this
  schema is the contract Task 3 (`load_translations`) and Task 5
  (`load_tasks_vi_csv`/`build_eval_overrides`) both consume once the `vietnamese` column is
  filled in and saved as `tasks_vi.csv`. Also produces
  `validate_translations(en_csv: Path, vi_csv: Path) -> None`.

- [ ] **Step 1: Write the failing tests**

```python
# vlai-experiments/vi-instructions/tests/test_extract_tasks.py
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
    assert reader == [{"suite": "libero_spatial", "task_id": "0", "english": "pick up the bowl", "vietnamese": ""}]
```

```python
# vlai-experiments/vi-instructions/tests/test_validate_translations.py
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
        {"suite": "libero_spatial", "task_id": "0", "english": "pick up the bowl", "vietnamese": "nhấc cái bát lên"}
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
        {"suite": "libero_spatial", "task_id": "0", "english": "pick up the bowl", "vietnamese": "nhấc cái bát lên"}
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest vlai-experiments/vi-instructions/tests/test_extract_tasks.py vlai-experiments/vi-instructions/tests/test_validate_translations.py -v`
Expected: FAIL/ERROR — modules don't exist yet.

- [ ] **Step 3: Write the implementation**

```python
# vlai-experiments/vi-instructions/extract_tasks.py
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
```

```python
# vlai-experiments/vi-instructions/validate_translations.py
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
    parser.add_argument("--en-csv", type=Path, default=Path("vlai-experiments/vi-instructions/data/tasks_en.csv"))
    parser.add_argument("--vi-csv", type=Path, default=Path("vlai-experiments/vi-instructions/data/tasks_vi.csv"))
    args = parser.parse_args()
    validate_translations(args.en_csv, args.vi_csv)
    print("OK: translations are well-formed.")
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest vlai-experiments/vi-instructions/tests/test_extract_tasks.py vlai-experiments/vi-instructions/tests/test_validate_translations.py -v`
Expected: PASS (6 tests).

- [ ] **Step 5: Extract the real English task strings (manual, needs `--extra libero`)**

Run: `uv run python vlai-experiments/vi-instructions/extract_tasks.py`
Expected: `Wrote 40 rows to vlai-experiments/vi-instructions/data/tasks_en.csv` (4 suites × 10 tasks
each — correcting the spec's earlier "~130" estimate, which was wrong; the real count is 40
unique task strings).

- [ ] **Step 6: Manual translation pass (not code — do this before Task 3)**

Copy `tasks_en.csv` to `vlai-experiments/vi-instructions/data/tasks_vi.csv`, fill in the
`vietnamese` column for all 40 rows via machine translation + manual review (per the
approved spec approach), then run:
`uv run python vlai-experiments/vi-instructions/validate_translations.py`
Expected: `OK: translations are well-formed.`

- [ ] **Step 7: Commit**

```bash
git add vlai-experiments/vi-instructions/extract_tasks.py vlai-experiments/vi-instructions/validate_translations.py vlai-experiments/vi-instructions/tests/test_extract_tasks.py vlai-experiments/vi-instructions/tests/test_validate_translations.py
git commit -m "feat(vi-instructions): add LIBERO task extraction and translation validator"
```

---

### Task 3: Vietnamese LIBERO dataset fork

**Files:**
- Create: `vlai-experiments/vi-instructions/build_dataset.py`
- Test: `vlai-experiments/vi-instructions/tests/test_build_dataset.py`

**Interfaces:**
- Consumes: `vlai-experiments/vi-instructions/data/tasks_vi.csv` (Task 2's schema, fully
  translated per Task 2 Step 6), `lerobot.datasets.io_utils.load_tasks`/`write_tasks`
  (existing, verified).
- Produces: `~/.cache/huggingface/lerobot/thuanan/libero-vi/` — a `LeRobotDataset`-compatible root, consumed by Task 6's
  training command.

- [ ] **Step 1: Write the failing tests**

```python
# vlai-experiments/vi-instructions/tests/test_build_dataset.py
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
    csv_path.write_text("suite,task_id,english,vietnamese\nlibero_spatial,0,pick up the bowl,\n", encoding="utf-8")
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest vlai-experiments/vi-instructions/tests/test_build_dataset.py -v`
Expected: FAIL/ERROR — `build_dataset` module doesn't exist yet.

- [ ] **Step 3: Write the implementation**

```python
# vlai-experiments/vi-instructions/build_dataset.py
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
```

`--src-root`/`--dst-root` are required (no default): the source dataset lives in the HF hub
snapshot cache, whose path includes a content-hash directory that can change on re-download —
hardcoding it as a default would go stale silently. Resolve it at call time instead (see
Step 5).

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest vlai-experiments/vi-instructions/tests/test_build_dataset.py -v`
Expected: PASS (5 tests).

- [ ] **Step 5: Build the real fork (manual)**

Resolve the current source snapshot directory (its hash suffix can change across
re-downloads, so don't hardcode it):

```bash
SRC_ROOT=$(find ~/.cache/huggingface/lerobot/hub/datasets--HuggingFaceVLA--libero/snapshots -mindepth 1 -maxdepth 1 -type d | head -1)
DST_ROOT=~/.cache/huggingface/lerobot/thuanan/libero-vi
uv run python vlai-experiments/vi-instructions/build_dataset.py --src-root "$SRC_ROOT" --dst-root "$DST_ROOT"
```

Expected: `Wrote Vietnamese dataset fork to <DST_ROOT>`; verify with
`uv run python -c "from lerobot.datasets.lerobot_dataset import LeRobotDataset; ds = LeRobotDataset(repo_id='thuanan/libero-vi', root='$DST_ROOT'); print(ds[0]['task'])"`
prints a Vietnamese string.

- [ ] **Step 6: Commit**

```bash
git add vlai-experiments/vi-instructions/build_dataset.py vlai-experiments/vi-instructions/tests/test_build_dataset.py
git commit -m "feat(vi-instructions): fork LIBERO dataset with translated Vietnamese task strings"
```

---

### Task 4: `LiberoEnv` instruction-language override (core code change)

**Files:**
- Modify: `src/lerobot/envs/libero.py:110-130` (`LiberoEnv.__init__` signature),
  `src/lerobot/envs/libero.py:169-172` (task extraction body),
  `src/lerobot/envs/libero.py:386-422` (`_make_env_fns`),
  `src/lerobot/envs/libero.py:428-510` (`create_libero_envs`)
- Modify: `src/lerobot/envs/configs.py:322-354` (`LiberoEnv(EnvConfig)` dataclass fields),
  `src/lerobot/envs/configs.py:409-418` (`gym_kwargs` property)
- Test: `tests/envs/test_libero_task_language.py` (new)

**Interfaces:**
- Consumes: nothing from other tasks.
- Produces: `LiberoEnv(..., task_language_override: str | None = None)`;
  `create_libero_envs(..., gym_kwargs={"task_language_overrides": {suite_name: {task_id: str}}})`;
  `LiberoEnv(EnvConfig).task_language_overrides_path: str | None` CLI field
  (`--env.task_language_overrides_path=<path-to-json>`). The JSON schema
  `{suite_name: {task_id_as_string: vietnamese_string}}` is the contract Task 5 must produce
  and Task 7's eval command consumes.

- [ ] **Step 1: Write the failing tests**

```python
# tests/envs/test_libero_task_language.py
import pytest

pytest.importorskip("libero")

from lerobot.envs.libero import create_libero_envs  # noqa: E402
from lerobot.envs.libero import LiberoEnv  # noqa: E402


class _FakeTask:
    def __init__(self, name: str, language: str):
        self.name = name
        self.language = language
        self.problem_folder = "fake_problem"
        self.init_states_file = "fake_init_states.pruned_init"
        self.bddl_file = "fake.bddl"


class _FakeTaskSuite:
    def __init__(self, languages: list[str]):
        self.tasks = [_FakeTask(f"task_{i}", lang) for i, lang in enumerate(languages)]

    def get_task(self, task_id: int) -> _FakeTask:
        return self.tasks[task_id]


def _make_libero_env(task_id: int, languages: list[str], task_language_override: str | None) -> LiberoEnv:
    return LiberoEnv(
        task_suite=_FakeTaskSuite(languages),
        task_id=task_id,
        task_suite_name="libero_spatial",
        init_states=False,
        task_language_override=task_language_override,
    )


def test_task_description_falls_back_to_libero_language_by_default():
    env = _make_libero_env(0, ["pick up the bowl"], task_language_override=None)
    assert env.task_description == "pick up the bowl"


def test_task_description_uses_override_when_provided():
    env = _make_libero_env(0, ["pick up the bowl"], task_language_override="nhấc cái bát lên")
    assert env.task_description == "nhấc cái bát lên"


def test_create_libero_envs_threads_per_task_overrides(monkeypatch):
    fake_suite = _FakeTaskSuite(["pick up the bowl", "open the drawer"])
    monkeypatch.setattr("lerobot.envs.libero._get_suite", lambda name: fake_suite)

    out = create_libero_envs(
        task="libero_spatial",
        n_envs=1,
        init_states=False,
        env_cls=list,
        gym_kwargs={"task_language_overrides": {"libero_spatial": {"0": "nhấc cái bát lên"}}},
    )

    env_task0 = out["libero_spatial"][0][0]()
    assert env_task0.task_description == "nhấc cái bát lên"

    env_task1 = out["libero_spatial"][1][0]()
    assert env_task1.task_description == "open the drawer"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/envs/test_libero_task_language.py -v`
Expected: FAIL — `LiberoEnv.__init__()` raises `TypeError: unexpected keyword argument
'task_language_override'`.

- [ ] **Step 3: Modify `src/lerobot/envs/libero.py`**

In `LiberoEnv.__init__`, add the new parameter after `is_libero_plus: bool = False,`:

```python
        is_libero_plus: bool = False,
        task_language_override: str | None = None,
    ):
```

Then change the task-extraction block (currently at `libero.py:169-172`):

```python
        # Extract task metadata without allocating GPU resources (safe before fork).
        task = task_suite.get_task(task_id)
        self.task = task.name
        self.task_description = task_language_override if task_language_override is not None else task.language
```

In `_make_env_fns`, add a parameter and forward it:

```python
def _make_env_fns(
    *,
    suite,
    suite_name: str,
    task_id: int,
    n_envs: int,
    camera_names: list[str],
    episode_length: int | None,
    init_states: bool,
    gym_kwargs: Mapping[str, Any],
    control_mode: str,
    camera_name_mapping: dict[str, str] | None = None,
    is_libero_plus: bool = False,
    task_language_override: str | None = None,
) -> list[Callable[[], LiberoEnv]]:
    """Build n_envs factory callables for a single (suite, task_id)."""

    def _make_env(episode_index: int, **kwargs) -> LiberoEnv:
        local_kwargs = dict(kwargs)
        return LiberoEnv(
            task_suite=suite,
            task_id=task_id,
            task_suite_name=suite_name,
            camera_name=camera_names,
            init_states=init_states,
            episode_length=episode_length,
            episode_index=episode_index,
            n_envs=n_envs,
            control_mode=control_mode,
            camera_name_mapping=camera_name_mapping,
            is_libero_plus=is_libero_plus,
            task_language_override=task_language_override,
            **local_kwargs,
        )

    fns: list[Callable[[], LiberoEnv]] = []
    for episode_index in range(n_envs):
        fns.append(partial(_make_env, episode_index, **gym_kwargs))
    return fns
```

In `create_libero_envs`, pop the new key alongside the existing `task_ids` pop
(around `libero.py:456`) and slice it per suite/task inside the existing suite loop:

```python
    gym_kwargs = dict(gym_kwargs or {})
    task_ids_filter = gym_kwargs.pop("task_ids", None)  # optional: limit to specific tasks
    task_language_overrides = gym_kwargs.pop("task_language_overrides", None)
```

and inside `for suite_name in suite_names:` (around `libero.py:472`), before the `for tid in
selected:` loop:

```python
        suite_overrides_raw = (task_language_overrides or {}).get(suite_name, {})
        suite_overrides = {int(k): v for k, v in suite_overrides_raw.items()}
```

then inside `for tid in selected:` (around `libero.py:485`), pass it through to `_make_env_fns`:

```python
            fns = _make_env_fns(
                suite=suite,
                episode_length=episode_length,
                suite_name=suite_name,
                task_id=tid,
                n_envs=n_envs,
                camera_names=camera_names,
                init_states=init_states,
                gym_kwargs=gym_kwargs,
                control_mode=control_mode,
                camera_name_mapping=camera_name_mapping,
                is_libero_plus=is_libero_plus,
                task_language_override=suite_overrides.get(tid),
            )
```

- [ ] **Step 4: Modify `src/lerobot/envs/configs.py`**

In `LiberoEnv(EnvConfig)`, add a field after `task_ids: list[int] | None = None`:

```python
    task_ids: list[int] | None = None
    task_language_overrides_path: str | None = None
```

In the `gym_kwargs` property (`configs.py:409-418`), load it if set:

```python
    @property
    def gym_kwargs(self) -> dict:
        kwargs: dict[str, Any] = {
            "obs_type": self.obs_type,
            "render_mode": self.render_mode,
            "observation_height": self.observation_height,
            "observation_width": self.observation_width,
        }
        if self.task_ids is not None:
            kwargs["task_ids"] = self.task_ids
        if self.task_language_overrides_path is not None:
            kwargs["task_language_overrides"] = json.loads(Path(self.task_language_overrides_path).read_text())
        return kwargs
```

Check the top of `configs.py` for existing `import json` / `from pathlib import Path` — add
them at module level if not already present (don't shadow with local imports).

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run pytest tests/envs/test_libero_task_language.py -v`
Expected: PASS (3 tests).

- [ ] **Step 6: Confirm no regression to default (no-override) behavior**

Run: `uv run pytest tests/envs/test_envs.py -k libero -v`
Expected: PASS (existing LIBERO env tests unaffected — the new parameters are additive with
`None`/falsy defaults).

- [ ] **Step 7: Type-check the modified strict-mypy modules**

Run: `uv run mypy src/lerobot/envs/libero.py src/lerobot/envs/configs.py`
Expected: no new errors introduced by this change.

- [ ] **Step 8: Commit**

```bash
git add src/lerobot/envs/libero.py src/lerobot/envs/configs.py tests/envs/test_libero_task_language.py
git commit -m "feat(envs/libero): add per-task Vietnamese instruction override for eval rollouts"
```

---

### Task 5: Eval-side instruction override JSON generator

**Files:**
- Create: `vlai-experiments/vi-instructions/build_eval_overrides.py`
- Test: `vlai-experiments/vi-instructions/tests/test_build_eval_overrides.py`

**Interfaces:**
- Consumes: `vlai-experiments/vi-instructions/data/tasks_vi.csv` (Task 2's schema).
- Produces: `vlai-experiments/vi-instructions/data/eval_overrides.json`, matching exactly the
  `{suite_name: {task_id_as_string: vietnamese_string}}` schema Task 4's
  `gym_kwargs["task_language_overrides"]` expects (loaded verbatim via `json.loads`, so key
  types must be strings — converted back to `int` inside `create_libero_envs`, per Task 4).

- [ ] **Step 1: Write the failing tests**

```python
# vlai-experiments/vi-instructions/tests/test_build_eval_overrides.py
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest

from build_eval_overrides import build_eval_overrides


def test_build_eval_overrides_groups_by_suite():
    rows = [
        {"suite": "libero_spatial", "task_id": "0", "english": "pick up the bowl", "vietnamese": "nhấc cái bát lên"},
        {"suite": "libero_spatial", "task_id": "1", "english": "open the drawer", "vietnamese": "mở ngăn kéo"},
        {"suite": "libero_object", "task_id": "0", "english": "turn on the stove", "vietnamese": "bật bếp lên"},
    ]

    overrides = build_eval_overrides(rows)

    assert overrides == {
        "libero_spatial": {"0": "nhấc cái bát lên", "1": "mở ngăn kéo"},
        "libero_object": {"0": "bật bếp lên"},
    }


def test_build_eval_overrides_rejects_empty_translation():
    rows = [{"suite": "libero_spatial", "task_id": "0", "english": "pick up the bowl", "vietnamese": ""}]
    with pytest.raises(ValueError):
        build_eval_overrides(rows)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest vlai-experiments/vi-instructions/tests/test_build_eval_overrides.py -v`
Expected: FAIL/ERROR — module doesn't exist yet.

- [ ] **Step 3: Write the implementation**

```python
# vlai-experiments/vi-instructions/build_eval_overrides.py
from __future__ import annotations

import csv
import json
from pathlib import Path


def load_tasks_vi_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def build_eval_overrides(rows: list[dict[str, str]]) -> dict[str, dict[str, str]]:
    overrides: dict[str, dict[str, str]] = {}
    for row in rows:
        suite = row["suite"]
        task_id = row["task_id"]
        vietnamese = row["vietnamese"]
        if not vietnamese.strip():
            raise ValueError(f"Empty Vietnamese translation for suite={suite} task_id={task_id}")
        overrides.setdefault(suite, {})[task_id] = vietnamese
    return overrides


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--tasks-vi-csv", type=Path, default=Path("vlai-experiments/vi-instructions/data/tasks_vi.csv")
    )
    parser.add_argument(
        "--output", type=Path, default=Path("vlai-experiments/vi-instructions/data/eval_overrides.json")
    )
    args = parser.parse_args()

    rows = load_tasks_vi_csv(args.tasks_vi_csv)
    overrides = build_eval_overrides(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(overrides, indent=2, ensure_ascii=False))
    print(f"Wrote overrides for {sum(len(v) for v in overrides.values())} tasks across {len(overrides)} suites")
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest vlai-experiments/vi-instructions/tests/test_build_eval_overrides.py -v`
Expected: PASS (2 tests).

- [ ] **Step 5: Build the real overrides file (manual, after Task 2 Step 6)**

Run: `uv run python vlai-experiments/vi-instructions/build_eval_overrides.py`
Expected: `Wrote overrides for 40 tasks across 4 suites`.

- [ ] **Step 6: Commit**

```bash
git add vlai-experiments/vi-instructions/build_eval_overrides.py vlai-experiments/vi-instructions/tests/test_build_eval_overrides.py
git commit -m "feat(vi-instructions): generate per-suite Vietnamese eval-override JSON"
```

---

### Task 6: Stage 2 — widened-LoRA Vietnamese training run

**Files:**
- Create: `run_vi.sh` (repo root, mirrors existing `run.sh`)

**Interfaces:**
- Consumes: `~/.cache/huggingface/lerobot/thuanan/libero-vi` (Task 3), the widened `target_modules` regex (derived directly
  from `modeling_smolvla.py:495-500`'s existing default — not from another task).
- Produces: a LoRA checkpoint under `outputs/train_vi/checkpoints/last/pretrained_model`,
  consumed by Task 7.

- [ ] **Step 1: Write `run_vi.sh`**

```bash
#!/usr/bin/env bash
set -euo pipefail

# SmolVLA + LIBERO Vietnamese-instruction LoRA finetune (Stage 2 of
# docs/superpowers/specs/2026-07-02-smolvla-vietnamese-instructions-design.md)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -f "${SCRIPT_DIR}/.env" ]]; then
  set -a
  source "${SCRIPT_DIR}/.env"
  set +a
fi

HF_USER="${HF_USER:?Set HF_USER (in .env or env) to your Hugging Face username}"
STEPS="${STEPS:-8000}"
BATCH_SIZE="${BATCH_SIZE:-16}"

export MUJOCO_GL="${MUJOCO_GL:-egl}"

WANDB_ARGS=()
if [[ -n "${WANDB_API_KEY:-}" ]]; then
  WANDB_ARGS=(--wandb.enable=true)
fi

# SmolVLA's default LoRA targets (_get_default_peft_targets in modeling_smolvla.py:495-500)
# only adapt the action expert's q/v projections. For Vietnamese instruction comprehension we
# also need the VLM's own text-model attention layers.
TARGET_MODULES='(model\.vlm_with_expert\.lm_expert\..*\.(q|v)_proj|model\.vlm_with_expert\.vlm\.model\.text_model\.layers\..*\.self_attn\.(q|k|v|o)_proj|model\.(state_proj|action_in_proj|action_out_proj|action_time_mlp_in|action_time_mlp_out))'

uv sync --locked --extra smolvla --extra libero

uv run lerobot-train \
  --policy.type=smolvla \
  --policy.repo_id="${HF_USER}/libero-vi" \
  --policy.load_vlm_weights=true \
  --policy.train_expert_only=false \
  --dataset.repo_id="${HF_USER}/libero-vi" \
  --dataset.root="${HOME}/.cache/huggingface/lerobot/${HF_USER}/libero-vi" \
  --peft.method_type=LORA \
  --peft.r="${LORA_R:-16}" \
  --peft.lora_alpha="${LORA_ALPHA:-32}" \
  --peft.target_modules="${TARGET_MODULES}" \
  --env.type=libero \
  --env.task=libero_10 \
  --output_dir=./outputs/train_vi/ \
  --steps="${STEPS}" \
  --batch_size="${BATCH_SIZE}" \
  --eval.batch_size=1 \
  --eval.n_episodes=1 \
  --env_eval_freq=1000 \
  "${WANDB_ARGS[@]}"
```

Run: `chmod +x run_vi.sh`

- [ ] **Step 2: Smoke-test the pipeline (manual, needs GPU + `--extra smolvla --extra libero`)**

Run: `STEPS=10 BATCH_SIZE=2 ./run_vi.sh`
Expected: completes without error and writes
`outputs/train_vi/checkpoints/last/pretrained_model/{config.json,model.safetensors}`.

- [ ] **Step 3: Full training run (manual)**

Run: `STEPS=8000 ./run_vi.sh` (adjust `STEPS`/`BATCH_SIZE` per the single-GPU budget from the
spec, starting small per Stage 2's guidance and scaling up only after the smoke test and an
initial eval look reasonable).

- [ ] **Step 4: Commit**

```bash
git add run_vi.sh
git commit -m "feat(vi-instructions): add Stage 2 widened-LoRA Vietnamese training script"
```

---

### Task 7: Stage 3 — Vietnamese eval + comparison report

**Files:**
- Create: `run_eval_vi.sh` (repo root)
- Create: `vlai-experiments/vi-instructions/compare_eval.py`
- Test: `vlai-experiments/vi-instructions/tests/test_compare_eval.py`

**Interfaces:**
- Consumes: `--env.task_language_overrides_path` (Task 4), `eval_overrides.json` (Task 5), the
  checkpoint from Task 6.
- Produces: `eval_info.json` per run (existing `lerobot-eval` output format,
  `{"overall": {"pc_success": float, ...}, "<suite>": {"pc_success": float, ...}, ...}`, per
  `scripts/lerobot_eval.py:979-998`) plus a markdown comparison table.

- [ ] **Step 1: Write `run_eval_vi.sh`**

```bash
#!/usr/bin/env bash
set -euo pipefail

# Stage 3 eval: run a checkpoint against LIBERO with Vietnamese task instructions
# substituted in via the Stage 1b override mechanism (envs/libero.py).

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -f "${SCRIPT_DIR}/.env" ]]; then
  set -a
  source "${SCRIPT_DIR}/.env"
  set +a
fi

CHECKPOINT_PATH="${1:?Usage: run_eval_vi.sh <checkpoint_path> <output_dir>}"
OUTPUT_DIR="${2:?Usage: run_eval_vi.sh <checkpoint_path> <output_dir>}"

export MUJOCO_GL="${MUJOCO_GL:-egl}"

uv run lerobot-eval \
  --policy.path="${CHECKPOINT_PATH}" \
  --env.type=libero \
  --env.task=libero_spatial,libero_object,libero_goal,libero_10 \
  --env.task_language_overrides_path="${SCRIPT_DIR}/vlai-experiments/vi-instructions/data/eval_overrides.json" \
  --eval.batch_size=1 \
  --eval.n_episodes=10 \
  --env.max_parallel_tasks=1 \
  --output_dir="${OUTPUT_DIR}"
```

Run: `chmod +x run_eval_vi.sh`

- [ ] **Step 2: Write the failing comparison-script tests**

```python
# vlai-experiments/vi-instructions/tests/test_compare_eval.py
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from compare_eval import build_comparison_table, render_markdown_table


def test_build_comparison_table_aligns_suites_across_runs():
    runs = {
        "vi_lora_on_vi": {"overall": {"pc_success": 40.0}, "libero_spatial": {"pc_success": 50.0}},
        "en_baseline_on_en": {"overall": {"pc_success": 70.0}, "libero_spatial": {"pc_success": 80.0}},
    }

    rows = build_comparison_table(runs)

    assert rows == [
        {"suite": "libero_spatial", "vi_lora_on_vi": 50.0, "en_baseline_on_en": 80.0},
        {"suite": "overall", "vi_lora_on_vi": 40.0, "en_baseline_on_en": 70.0},
    ]


def test_render_markdown_table_produces_expected_header_and_rows():
    rows = [{"suite": "overall", "run_a": 40.0}]
    table = render_markdown_table(rows, ["run_a"])
    lines = table.splitlines()
    assert lines[0] == "| suite | run_a |"
    assert lines[1] == "|---|---|"
    assert lines[2] == "| overall | 40.0 |"
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `uv run pytest vlai-experiments/vi-instructions/tests/test_compare_eval.py -v`
Expected: FAIL/ERROR — module doesn't exist yet.

- [ ] **Step 4: Write the implementation**

```python
# vlai-experiments/vi-instructions/compare_eval.py
from __future__ import annotations

import json
from pathlib import Path


def load_eval_info(path: Path) -> dict:
    return json.loads(path.read_text())


def build_comparison_table(runs: dict[str, dict]) -> list[dict[str, str | float]]:
    suites = sorted({suite for info in runs.values() for suite in info if suite != "overall"})
    rows: list[dict[str, str | float]] = []
    for suite in [*suites, "overall"]:
        row: dict[str, str | float] = {"suite": suite}
        for run_name, info in runs.items():
            row[run_name] = info.get(suite, {}).get("pc_success", float("nan"))
        rows.append(row)
    return rows


def render_markdown_table(rows: list[dict[str, str | float]], run_names: list[str]) -> str:
    header = "| suite | " + " | ".join(run_names) + " |"
    separator = "|---" * (len(run_names) + 1) + "|"
    lines = [header, separator]
    for row in rows:
        cells = [str(row["suite"])] + [
            f"{row[name]:.1f}" if isinstance(row[name], float) else str(row[name]) for name in run_names
        ]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--vi-lora-on-vi", type=Path, required=True)
    parser.add_argument("--en-baseline-on-en", type=Path, required=True)
    parser.add_argument("--en-baseline-zero-shot-vi", type=Path, required=True)
    args = parser.parse_args()

    runs = {
        "vi_lora_on_vi": load_eval_info(args.vi_lora_on_vi),
        "en_baseline_on_en": load_eval_info(args.en_baseline_on_en),
        "en_baseline_zero_shot_vi": load_eval_info(args.en_baseline_zero_shot_vi),
    }
    rows = build_comparison_table(runs)
    print(render_markdown_table(rows, list(runs.keys())))
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run pytest vlai-experiments/vi-instructions/tests/test_compare_eval.py -v`
Expected: PASS (2 tests).

- [ ] **Step 6: Run the three real eval passes (manual, needs GPU + Task 6's checkpoint)**

```bash
./run_eval_vi.sh outputs/train_vi/checkpoints/last/pretrained_model outputs/eval_vi_lora_on_vi
./run_eval_vi.sh <english-baseline-checkpoint-from-run.sh> outputs/eval_en_baseline_zero_shot_vi
# English baseline on English instructions: reuse run.sh's existing eval invocation
# (no --env.task_language_overrides_path) against outputs/eval/ from the original run.sh.
```

- [ ] **Step 7: Generate the comparison report**

Run:
```bash
uv run python vlai-experiments/vi-instructions/compare_eval.py \
  --vi-lora-on-vi outputs/eval_vi_lora_on_vi/eval_info.json \
  --en-baseline-on-en outputs/eval/eval_info.json \
  --en-baseline-zero-shot-vi outputs/eval_en_baseline_zero_shot_vi/eval_info.json
```
Expected: a markdown table with per-suite and overall `pc_success` for all three runs — this
is the spec's Stage 3 deliverable.

- [ ] **Step 8: Commit**

```bash
git add run_eval_vi.sh vlai-experiments/vi-instructions/compare_eval.py vlai-experiments/vi-instructions/tests/test_compare_eval.py
git commit -m "feat(vi-instructions): add Stage 3 Vietnamese eval script and comparison report"
```
