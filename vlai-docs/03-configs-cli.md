# Bài 03 — Configs & draccus CLI (ChoiceRegistry)

## Mục tiêu
- Hiểu cách LeRobot biến **dataclass thành CLI** bằng `draccus`.
- Hiểu cơ chế **polymorphism qua `ChoiceRegistry`** — nền tảng để thêm policy/env/processor mới.
- Đọc được `TrainPipelineConfig` và biết mọi flag `--abc.xyz` đến từ đâu.

## 1. draccus: dataclass ⇄ CLI ⇄ JSON

LeRobot không dùng argparse thủ công. Mỗi config là một `@dataclass`, và `draccus` tự sinh
CLI từ nó. Quy tắc: **đường dẫn field = đường dẫn flag**.

```
TrainPipelineConfig.batch_size            -> --batch_size=64
TrainPipelineConfig.dataset.repo_id       -> --dataset.repo_id=user/ds
TrainPipelineConfig.policy.type           -> --policy.type=smolvla
TrainPipelineConfig.wandb.enable          -> --wandb.enable=true
```

Config gốc cho training là `TrainPipelineConfig` (`src/lerobot/configs/train.py:78`).
Một vài field quan trọng (đọc trực tiếp file để thấy hết):

| Field | Ý nghĩa |
|---|---|
| `dataset: DatasetConfig` | Dataset nào (repo_id, splits, delta_timestamps...) |
| `policy: PreTrainedConfig` | Model nào + hyperparams |
| `env: EnvConfig \| None` | Môi trường eval (nếu có) |
| `batch_size`, `steps`, `num_workers` | Cấu hình train cơ bản |
| `save_freq`, `output_dir`, `resume` | Checkpoint |
| `optimizer`, `scheduler` | Tối ưu (mặc định lấy từ preset của policy nếu `use_policy_training_preset=True`) |
| `eval`, `eval_steps`, `env_eval_freq` | Đánh giá |
| `wandb` | Logging |
| `peft: PeftConfig \| None` | LoRA/PEFT (Bài 14) |

## 2. ChoiceRegistry — trái tim của tính mở rộng

`PreTrainedConfig` (`src/lerobot/configs/policies.py:41`) khai báo:

```python
class PreTrainedConfig(draccus.ChoiceRegistry, HubMixin, abc.ABC):
    ...
```

`ChoiceRegistry` cho phép một field "đa hình": `--policy.type=smolvla` sẽ khiến draccus
chọn đúng subclass `SmolVLAConfig`. Mỗi policy đăng ký tên của nó bằng decorator, đại loại:

```python
@PreTrainedConfig.register_subclass("smolvla")
@dataclass
class SmolVLAConfig(PreTrainedConfig):
    ...
```

➡️ **Đây chính là cách bạn thêm model mới** (Bài 13): viết một `@dataclass` config, đăng ký
một tên, và draccus + factory lo phần còn lại.

Xem danh sách tên policy đã đăng ký:

```bash
uv run python -c "from lerobot.configs.policies import PreTrainedConfig as P; print(sorted(P.get_known_choices()))"
```

Cùng cơ chế áp dụng cho `EnvConfig` (`src/lerobot/envs/configs.py`) và các config khác.

## 3. input_features / output_features

`PreTrainedConfig` có hai field then chốt (`policies.py:59`):

```python
input_features:  dict[str, PolicyFeature] | None
output_features: dict[str, PolicyFeature] | None
```

- Nếu để `None`, chúng được **suy ra từ dataset** lúc train.
- Có các helper lọc theo loại, ví dụ `image_features` lọc các feature `FeatureType.VISUAL`
  (`policies.py:152`). Policy VLA dùng cái này để biết "có mấy camera, ảnh kích thước bao nhiêu".

## 4. Config từ file JSON / từ Hub

`TrainPipelineConfig` là `HubMixin` → có thể lưu/khôi phục từ `train_config.json`
(`TRAIN_CONFIG_NAME`, `train.py:35`). Khi `--resume=true`, config được nạp lại từ checkpoint.
Bạn cũng có thể `--config_path=...` để nạp một config có sẵn rồi override vài flag trên CLI.

## 5. Thực hành

In ra cấu trúc config train (giúp biết flag nào tồn tại):

```bash
uv run lerobot-train --help 2>&1 | head -60
```

Tạo nhanh một policy config bằng factory (không cần train):

```bash
uv run python - <<'PY'
from lerobot.policies.factory import make_policy_config
cfg = make_policy_config("act")          # thử "smolvla" nếu đã cài extra
print(type(cfg).__name__)
print("chunk_size:", getattr(cfg, "chunk_size", None))
PY
```

## 6. Tự kiểm tra
1. Flag `--policy.type=smolvla` khiến draccus làm gì dưới gầm?
2. Nếu không khai báo `input_features`, policy lấy schema đầu vào ở đâu?
3. Thêm một model mới cần đăng ký gì, ở đâu?

➡️ Tiếp theo: [04 — Processor pipeline](./04-processor-pipeline.md)
