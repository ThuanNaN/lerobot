# Bài 05 — Policies: PreTrainedPolicy, factory, forward/select_action

## Mục tiêu
- Hiểu "policy" trong LeRobot = một `nn.Module` + giao diện chuẩn để train và infer.
- Nắm hợp đồng (contract) bắt buộc của `PreTrainedPolicy`.
- Biết `factory.py` tạo policy ra sao — nền tảng để cắm model mới (Bài 13).

## 1. Policy là gì?

Một policy ánh xạ **quan sát → hành động**. Trong LeRobot, mọi policy kế thừa
`PreTrainedPolicy` (`src/lerobot/policies/pretrained.py:107`):

```python
class PreTrainedPolicy(nn.Module, HubMixin, abc.ABC):
    config_class: None        # mỗi policy phải gán config_class của mình
```

`HubMixin` cho khả năng `from_pretrained`/`push_to_hub` (`:171`). Lúc subclass mà quên gán
`config_class`, repo sẽ báo lỗi ngay (`:127`).

## 2. Hợp đồng bắt buộc (abstract methods)

Mọi policy phải cài (xem các `@abc.abstractmethod` quanh `pretrained.py:255–294`):

| Method | Khi nào gọi | Ý nghĩa |
|---|---|---|
| `forward(batch) -> (loss, dict\|None)` (`:272`) | Training | Tính loss từ một batch |
| `select_action(batch, **kwargs) -> Tensor` (`:294`) | Inference | Trả về 1 action cho bước hiện tại |
| `reset()` | Đầu episode | Xóa state/queue nội bộ |
| `get_optimizer_params()` / preset | Setup train | Tham số cho optimizer |

Nhiều policy (ACT, SmolVLA...) dự đoán cả **chuỗi action** rồi đẩy vào một hàng đợi nội bộ;
`select_action` lấy dần ra từng action. Đó là lý do có `reset()` và `populate_queues`
(`policies/utils.py`).

## 3. Factory — tạo policy từ tên/cfg

`src/lerobot/policies/factory.py` là điểm trung tâm:

- `get_policy_class(name)` (`:~86`) — ánh xạ tên → class. Hỗ trợ:
  `tdmpc, diffusion, act, multi_task_dit, vqbet, pi0, pi05, gaussian_actor, smolvla, wall_x, molmoact2`...
- `make_policy_config(policy_type, **kwargs)` (`:172`) — tạo config tương ứng.
- `make_policy(cfg, ...)` (`:466`) — dựng policy hoàn chỉnh (load weights nếu có `path`).

Quan trọng: factory dùng **lazy import** (chỉ import policy khi cần) để không bắt buộc cài
mọi extra. Khi thêm model mới, bạn sẽ thêm một nhánh ở đây (Bài 13).

## 4. Từ config tới model: input/output features

Policy đọc `cfg.input_features` / `cfg.output_features` (Bài 03) để dựng các đầu vào/đầu ra.
Với VLA: `image_features` (các camera) + state + ngôn ngữ → backbone VLM; `output_features`
(thường là `action`) → "action head"/"action expert".

## 5. Bản đồ các policy có sẵn

| Nhóm | Policy | Ghi chú |
|---|---|---|
| Non-VLM (cổ điển) | `act`, `diffusion`, `tdmpc`, `vqbet`, `multi_task_dit` | Tốt để học cơ chế, train nhanh, máy yếu vẫn chạy |
| **VLA (VLM-based)** | `smolvla`, `pi0`, `pi05`, `pi0_fast`, `pi_gemma`, `eo1`, `groot`, `molmoact2`, `xvla`, `wall_x`, `vla_jepa` | **Trọng tâm cho dự án VQA/VLM** |
| RL | `gaussian_actor`, `rtc` | Liên quan HIL-SERL/async |

Lời khuyên học: **bắt đầu với `act`** để nắm vòng đời policy (nhẹ, dễ chạy), rồi mới sang
`smolvla` (Bài 09) để hiểu phần VLM.

## 6. Thực hành

```bash
# Liệt kê policy đã đăng ký
uv run python -c "from lerobot.configs.policies import PreTrainedConfig as P; print(sorted(P.get_known_choices()))"

# Soi giao diện base
sed -n '107,130p;255,300p' src/lerobot/policies/pretrained.py
```

Tạo một policy ACT nhỏ và chạy forward giả lập (hiểu shape vào/ra):

```bash
uv run python - <<'PY'
from lerobot.policies.factory import make_policy_config
cfg = make_policy_config("act")
print("config:", type(cfg).__name__)
print("output_features keys:", list((cfg.output_features or {}).keys()))
PY
```

## 7. Tự kiểm tra
1. Hai method nào *bắt buộc* phải cài cho mọi policy? Cái nào dùng lúc train, cái nào lúc infer?
2. Vì sao factory dùng lazy import?
3. Một VLA policy nhận ngôn ngữ vào ở "nhánh" nào của model?

➡️ Tiếp theo: [06 — Training loop](./06-training.md)
