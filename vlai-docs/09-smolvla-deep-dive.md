# Bài 09 — SmolVLA deep dive (code-level)

## Mục tiêu
- Đọc hiểu SmolVLA ở mức code: config, model, luồng `forward`/`select_action`.
- Thấy chính xác **ngôn ngữ đi vào model ở đâu** — chìa khóa cho VQA tiếng Việt.
- Nắm các nút bấm (config) để finetune hiệu quả.

> Paper: SmolVLA (arXiv 2506.01844). Code: `src/lerobot/policies/smolvla/`.

## 1. Hai lớp chính

- `SmolVLAPolicy(PreTrainedPolicy)` (`modeling_smolvla.py:226`) — wrapper train/infer,
  `config_class = SmolVLAConfig` (`:229`).
- `VLAFlowMatching(nn.Module)` (`modeling_smolvla.py:541`) — model lõi, dùng **flow matching**
  để sinh action chunk. Bên trong nó là `SmolVLMWithExpertModel`
  (`smolvlm_with_expert.py`) = **VLM backbone + action expert**.

Kiến trúc khái niệm:

```
ảnh(nhiều cam) ─ resize_with_pad ─┐
task text ─ tokenizer ────────────┼─> SmolVLM (VLM)  ──(prefix/cross-attn)──> Action Expert ──> action chunk
state ─ Linear(state_proj) ───────┘                                              ▲
                                                  noise + time (flow matching) ──┘
```

## 2. Config quan trọng (`configuration_smolvla.py`)

`@PreTrainedConfig.register_subclass("smolvla")` (`:24`). Các field đáng nhớ:

| Field | Mặc định | Ý nghĩa |
|---|---|---|
| `vlm_model_name` | `HuggingFaceTB/SmolVLM2-500M-Video-Instruct` | **Backbone VLM** — đổi cái này để dùng VLM hỗ trợ tiếng Việt tốt hơn |
| `load_vlm_weights` | `False` | True khi init từ SmolVLA pretrained; False khi train expert từ đầu |
| `chunk_size` / `n_action_steps` | 50 / 50 | Độ dài action chunk |
| `tokenizer_max_length` | 48 | Giới hạn token ngôn ngữ — **tăng nếu câu hỏi VQA tiếng Việt dài** |
| `num_steps` | 10 | Số bước denoise (flow matching) lúc sinh action |
| `freeze_vision_encoder` | True | Đóng băng vision encoder khi finetune |
| `train_expert_only` | True | Chỉ train action expert (rẻ) — **lưu ý cho VQA bạn sẽ cần mở phần ngôn ngữ** |
| `num_vlm_layers` | 16 | Số layer VLM dùng |
| `attention_mode` | `cross_attn` | Cách expert đọc context từ VLM |
| `optimizer_lr`, `scheduler_*` | | Preset tối ưu (Bài 06) |

> Cho VQA tiếng Việt: 3 nút bạn sẽ chỉnh đầu tiên là `vlm_model_name`,
> `tokenizer_max_length`, và việc bỏ `train_expert_only` để học phần ngôn ngữ.

## 3. Ngôn ngữ đi vào đâu? (đoạn quan trọng nhất)

Cả train lẫn infer đều đọc token ngôn ngữ từ batch dưới key chuẩn:

```python
# forward (training)  -> modeling_smolvla.py:377
lang_tokens = batch[OBS_LANGUAGE_TOKENS]
lang_masks  = batch[OBS_LANGUAGE_ATTENTION_MASK]

# _get_action_chunk (inference) -> :290
lang_tokens = batch[OBS_LANGUAGE_TOKENS]
lang_masks  = batch[OBS_LANGUAGE_ATTENTION_MASK]
```

Hai key này do **tokenizer step** trong processor tạo ra (Bài 04), từ cột `task` hoặc từ
`messages` đã render bằng recipe (Bài 10). Vậy chuỗi liên kết là:

```
dataset(text tiếng Việt) → RenderMessagesStep → TokenizerStep → OBS_LANGUAGE_TOKENS → SmolVLM
```

➡️ Để model "hiểu tiếng Việt", điểm tác động hiệu quả nhất là **tokenizer + LLM của
`vlm_model_name`**, không phải sửa SmolVLA logic.

## 4. Luồng training: `forward` (`:358`)

1. `prepare_images` (`:415`) — resize/pad ảnh nhiều camera.
2. `prepare_state` (`:484`) — chiếu state về không gian ẩn.
3. Lấy `lang_tokens/lang_masks`, `actions`, sinh `noise` + `time`.
4. `self.model.forward(images, img_masks, lang_tokens, lang_masks, state, actions, noise, time)`
   → trả về `losses` (mục tiêu flow matching).

## 5. Luồng inference: `select_action` (`:325`)

- `select_action` quản một **hàng đợi action**: chỉ gọi sinh chunk khi hàng đợi cạn, rồi nhả
  từng action mỗi bước. `reset()` (`:251`) xóa hàng đợi đầu episode.
- `_get_action_chunk` (`:276`) gọi `self.model.sample_actions(...)` để denoise ra chunk.
- Có hỗ trợ **RTC** (real-time chunking) cho inference độ trễ thấp (`init_rtc_processor`,
  `predict_action_chunk`).

## 6. PEFT/LoRA đã được "đi dây" sẵn

`_get_default_peft_targets` (`:495`) đặt target LoRA vào q/v projection của
`lm_expert` và vài projection chung (`:500`). Nghĩa là SmolVLA **đã sẵn sàng cho LoRA**
(Bài 14) — rất tiện khi GPU hạn chế.

## 7. Thực hành
```bash
# Soi các điểm ngôn ngữ đi vào model
grep -n "OBS_LANGUAGE\|sample_actions\|def forward\|def select_action\|vlm_model_name" \
  src/lerobot/policies/smolvla/modeling_smolvla.py | head

# Xem config đầy đủ
sed -n '1,160p' src/lerobot/policies/smolvla/configuration_smolvla.py
```

Thử khởi tạo (cần `uv sync --extra smolvla`):
```bash
uv run python - <<'PY'
from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
c = SmolVLAConfig()
print("backbone:", c.vlm_model_name)
print("tok_max_len:", c.tokenizer_max_length, "| chunk:", c.chunk_size)
PY
```

## 8. Tự kiểm tra
1. Hai key batch nào mang ngôn ngữ vào SmolVLA? Ai tạo ra chúng?
2. Muốn câu hỏi VQA tiếng Việt dài hơn, chỉnh field nào?
3. Vì sao đổi `vlm_model_name` là đòn bẩy lớn nhất cho tiếng Việt?

➡️ Tiếp theo: [10 — Language columns, recipes & VQA](./10-language-recipes-vqa.md)
