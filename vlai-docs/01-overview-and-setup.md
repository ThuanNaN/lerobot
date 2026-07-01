# Bài 01 — Tổng quan kiến trúc LeRobot & cài đặt môi trường

## Mục tiêu
- Hiểu LeRobot được tổ chức thế nào và **luồng dữ liệu chạy qua đâu**.
- Dựng được môi trường dev chạy được test.
- Biết các "điểm cắm" (extension points) mà sau này bạn sẽ đụng tới cho VQA/VLM.

## 1. Bức tranh tổng thể

LeRobot là thư viện PyTorch cho robot học thật, nhưng về bản chất kỹ thuật nó là một
**framework training multimodal**: dữ liệu (ảnh/state/ngôn ngữ/hành động) → processor →
policy (model) → loss/optimizer → checkpoint → eval.

Luồng huấn luyện rút gọn:

```
LeRobotDataset  ──>  DataLoader  ──>  PreProcessor (pipeline)  ──>  Policy.forward()  ──>  loss
   (parquet+video)      (batch)        (normalize, tokenize...)       (VLM + head)
                                                                          │
                                              Optimizer/Scheduler  <──────┘
                                                                          │
                                                              Checkpoint + WandB
```

Luồng inference rút gọn:

```
Observation  ──>  PreProcessor  ──>  Policy.select_action()  ──>  PostProcessor  ──>  Action
```

## 2. Cây thư mục cốt lõi (`src/lerobot/`)

| Thư mục | Vai trò | Bài liên quan |
|---|---|---|
| `scripts/` | CLI entry points: `lerobot-train`, `lerobot-eval`, `lerobot-record`, `lerobot-annotate`... | 06, 07, 11 |
| `configs/` | Dataclass config + draccus CLI. `train.py`, `policies.py`, `recipe.py`, `dataset.py` | 03, 10 |
| `policies/` | Mỗi policy 1 subdir, kế thừa `PreTrainedPolicy`. `factory.py` tạo policy | 05, 08, 09, 13 |
| `processor/` | Pipeline biến đổi dữ liệu: normalize, tokenizer, rename, device... | 04 |
| `datasets/` | `LeRobotDataset`, metadata, video decode, `language.py` (VQA) | 02, 10 |
| `annotations/` | `steerable_pipeline` sinh annotation (gồm `general_vqa`) | 11 |
| `envs/` | Môi trường mô phỏng (Gymnasium): libero, metaworld, robocasa... | 07 |
| `robots/ motors/ cameras/ teleoperators/` | Lớp phần cứng (ít liên quan tới VQA tiếng Việt) | — |
| `optim/` | Optimizer & scheduler config | 06 |
| `model/` | Khối model dùng chung | 09, 13 |

Các file gốc khác: `pyproject.toml` (nguồn chân lý duy nhất cho deps, extras, scripts),
`Makefile` (E2E test), `uv.lock`.

> **Mánh đọc code nhanh:** mọi script CLI trong `pyproject.toml` ở mục `[project.scripts]`
> đều trỏ tới một hàm trong `src/lerobot/scripts/`. Ví dụ `lerobot-train` → `scripts/lerobot_train.py`.

## 3. Các "điểm cắm" bạn sẽ dùng cho dự án VQA/VLM

Ghi nhớ 4 điểm này — toàn bộ Module 4 xoay quanh chúng:

1. **`PreTrainedConfig`** (`configs/policies.py:41`) — base config policy, dùng
   `draccus.ChoiceRegistry`. Thêm model mới = thêm một subclass `@PreTrainedConfig.register_subclass("ten")`.
2. **`PreTrainedPolicy`** (`policies/pretrained.py:107`) — base model. Bắt buộc cài
   `forward()` và `select_action()`.
3. **`ProcessorStep`** (`processor/pipeline.py:143`) + `ProcessorStepRegistry` — thêm bước
   xử lý dữ liệu mới (ví dụ tokenizer tiếng Việt).
4. **`LeRobotDataset` + language columns** (`datasets/language.py`) — nơi chứa dữ liệu
   VQA tiếng Việt của bạn.

## 4. Thực hành — cài môi trường

```bash
# Base
uv sync --locked

# Thêm test + dev tools (cần cho việc học/chỉnh code)
uv sync --locked --extra test --extra dev

# Nếu muốn chạy VLA models (smolvla...), cài thêm extra tương ứng
uv sync --locked --extra smolvla     # hoặc --extra all cho mọi thứ

# Tải artifact test (LFS)
git lfs install && git lfs pull
```

Kiểm tra cài đặt:

```bash
# Liệt kê policy/env/dataset đã đăng ký
uv run lerobot-info --help

# Chạy thử một phần test nhanh để chắc môi trường OK
uv run pytest tests -k "dataset" -q --maxfail=5
```

Xem các extra có sẵn (để biết policy nào cần gói gì):

```bash
uv run python -c "import tomllib,pathlib; d=tomllib.loads(pathlib.Path('pyproject.toml').read_text()); print(list(d['project']['optional-dependencies']))"
```

## 5. Tự kiểm tra
1. Khi gõ `lerobot-train`, file Python nào được gọi? (Gợi ý: `pyproject.toml` → `[project.scripts]`.)
2. Dữ liệu ngôn ngữ (text) đi vào model qua bước nào trong luồng training?
3. Để thêm một policy mới, bạn cần subclass 2 lớp base nào?

➡️ Tiếp theo: [02 — LeRobotDataset](./02-dataset.md)
