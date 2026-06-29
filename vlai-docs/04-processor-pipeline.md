# Bài 04 — Processor pipeline (ProcessorStep, normalize, tokenizer)

## Mục tiêu
- Hiểu **processor pipeline** đứng giữa dataset và policy, làm gì và tại sao tách riêng.
- Biết các step quan trọng: normalize, tokenizer, rename, device, render messages.
- Biết cách viết một `ProcessorStep` mới (kỹ năng cốt lõi cho tokenizer tiếng Việt).

## 1. Tại sao có processor?

Dataset trả về dữ liệu "thô" (ảnh, state, text). Model cần dữ liệu "chín" (đã normalize,
text đã tokenize, đúng device/dtype, đúng tên key). LeRobot tách phần biến đổi này thành
một **pipeline các bước** có thể tái sử dụng, lưu/khôi phục cùng model, và chạy cả khi
inference. Nhờ vậy training và inference dùng *đúng cùng một* tiền xử lý → tránh lệch phân phối.

Tài liệu chính thức: `docs/source/introduction_processors.mdx`,
`docs/source/implement_your_own_processor.mdx`, `docs/source/env_processor.mdx`.

## 2. Các lớp nền (`src/lerobot/processor/pipeline.py`)

- `ProcessorStep` (`:143`) — base ABC. Mỗi step nhận một `EnvTransition` và trả về một
  `EnvTransition` đã biến đổi (`__call__` ở `:171`). Có các base con tiện dụng như
  `ObservationProcessorStep`, `ActionProcessorStep`.
- `ProcessorStepRegistry` (`:59`) — đăng ký step theo tên để khôi phục từ config/Hub.
- `DataProcessorPipeline[TInput, TOutput]` (`:254`) — chuỗi nhiều step lại; gọi `pipeline(data)`
  để chạy tuần tự (`__call__` ở `:289`).

Đăng ký một step để nó portable (lưu được ra config):

```python
@ProcessorStepRegistry.register("my_vi_tokenizer")
class MyViTokenizerStep(ProcessorStep):
    ...
```

## 3. Tiền xử lý vs hậu xử lý

Một policy thường đi kèm **hai** pipeline (xem `processor/factory.py` và
`PolicyProcessorPipeline`):
- **PreProcessor**: dataset/observation → batch sẵn sàng cho `forward`/`select_action`.
- **PostProcessor**: output model (ví dụ action đã normalize) → action thật để gửi cho robot
  (un-normalize, đổi tên, đưa về CPU...).

## 4. Các step bạn sẽ gặp nhiều (`src/lerobot/processor/`)

| File | Step | Vai trò |
|---|---|---|
| `normalize_processor.py` | Normalize/Unnormalize | Chuẩn hóa bằng `stats.json` của dataset |
| `tokenizer_processor.py` | Tokenizer | Biến cột `task`/text → token ids + attention mask |
| `rename_processor.py` | Rename | Đổi tên key cho khớp model |
| `device_processor.py` | Device/Dtype | Đưa tensor lên GPU, ép dtype |
| `render_messages_processor.py` | `RenderMessagesStep` | Dựng `messages` chat từ recipe (Bài 10 — **quan trọng cho VQA**) |
| `newline_task_processor.py` | | Chuẩn hóa chuỗi task |
| `batch_processor.py` | | Gom/biến đổi theo batch |
| `delta_action_processor.py`, `relative_action_processor.py` | | Biểu diễn action |

Với dự án VQA tiếng Việt, hai step bạn quan tâm nhất là **tokenizer** (token tiếng Việt) và
**render messages** (dựng cặp hỏi-đáp VQA thành chat turns).

## 5. Tokenizer step — cận cảnh ý tưởng

VLM trong LeRobot (SmolVLA/pi0...) dùng tokenizer của chính backbone VLM (ví dụ SmolVLM).
`tokenizer_processor.py` lấy text (task hoặc messages đã render), tokenize, rồi đặt vào batch
dưới các key chuẩn như `OBS_LANGUAGE_TOKENS`, `OBS_LANGUAGE_ATTENTION_MASK`
(xem `src/lerobot/utils/constants.py`, và cách SmolVLA dùng chúng ở
`policies/smolvla/modeling_smolvla.py`).

➡️ Hệ quả thực tế: nếu backbone VLM bạn chọn **đã hỗ trợ tiếng Việt** trong tokenizer của nó,
phần lớn việc "xử lý tiếng Việt" được lo sẵn. Nếu không, bạn cần đổi backbone hoặc mở rộng
tokenizer (Bài 13/15).

## 6. Thực hành — soi pipeline của một policy

```bash
uv run python - <<'PY'
from lerobot.processor.factory import make_pre_post_processors  # tên hàm có thể khác, xem factory.py
print("Xem các hàm dựng pipeline trong processor/factory.py")
PY
ls src/lerobot/processor
uv run python -c "import inspect, lerobot.processor.factory as f; print([n for n in dir(f) if not n.startswith('_')])"
```

Đọc kỹ một step làm mẫu trước khi tự viết:

```bash
sed -n '1,80p' src/lerobot/processor/tokenizer_processor.py
```

## 7. Tự kiểm tra
1. Vì sao dùng cùng processor cho train và inference lại quan trọng?
2. `RenderMessagesStep` biến cái gì thành cái gì? (Liên hệ Bài 10.)
3. Để tokenizer hiểu tiếng Việt tốt, bạn tác động ở backbone VLM hay ở processor step? Vì sao?

➡️ Tiếp theo: [05 — Policies cơ bản](./05-policies-co-ban.md)
