# Bài 10 — Language columns, recipes & VQA (bài lõi cho dự án)

## Mục tiêu
- Hiểu hệ thống ngôn ngữ 3 tầng của LeRobot: **cột dữ liệu → recipe → messages**.
- Biết style `vqa` đã được hỗ trợ tận răng và cách dùng nó cho **VQA tiếng Việt**.
- Viết được một recipe VQA và hiểu output mà policy nhận.

> Tài liệu gốc: `docs/source/language_and_recipes.mdx`. Code:
> `datasets/language.py`, `datasets/language_render.py`, `configs/recipe.py`,
> `processor/render_messages_processor.py`.

## 1. Vì sao bài này quan trọng nhất với bạn

LeRobot **đã thiết kế sẵn** đường đi cho dữ liệu hỏi-đáp trên ảnh (VQA). Bạn gần như không
phải sửa lõi — chỉ cần (a) đổ dữ liệu VQA tiếng Việt vào đúng cột, và (b) khai một recipe.
Toàn bộ phần render thành chat + tokenize + đưa vào VLM đã có.

## 2. Tầng 1 — hai cột ngôn ngữ trong dataset

`LeRobotDataset` có thể mang thêm 2 cột (tùy chọn) cạnh frame
(`datasets/language.py`):

- `language_persistent` — trạng thái **duy trì** qua nhiều frame (`subtask`, `plan`, `memory`,
  `motion`, `task_aug`). Mỗi row có `timestamp` riêng.
- `language_events` — sự kiện **chỉ tại đúng 1 frame** (`interjection`, **`vqa`**, `trace`).

Schema một row (rút gọn):

```text
role: str            # "user" | "assistant" | "system" | "tool"
content: str | null  # nội dung text
style: str | null    # "vqa", "subtask", ...
timestamp: float32   # chỉ cột persistent
camera: str | null   # key observation.images.* (BẮT BUỘC cho vqa/trace)
tool_calls: list | null
```

Các tập style (đọc trực tiếp `language.py`):
- `CORE_STYLES` gồm `vqa` (`language.py:34`...).
- `EVENT_ONLY_STYLES = {"interjection", "vqa", "trace"}` (`language.py:50`).
- `VIEW_DEPENDENT_STYLES = {"vqa", "trace"}` (`language.py:62`) → **row `vqa` BẮT BUỘC có
  `camera` trỏ tới một `observation.images.*`** (vì câu hỏi gắn với một góc nhìn).

➡️ Một mẫu VQA tiếng Việt = **một cặp 2 event row cùng timestamp, cùng camera**:
```python
# language_events tại frame t (ví dụ camera "observation.images.top")
{"role": "user",      "style": "vqa", "camera": "observation.images.top",
 "content": "Vật màu đỏ đang ở bên nào của cái bát?"}
{"role": "assistant", "style": "vqa", "camera": "observation.images.top",
 "content": "Bên trái cái bát."}
```

## 3. Tầng 2 — recipe (YAML, không cần code)

Recipe (`configs/recipe.py`: `TrainingRecipe`, `MessageTurn`) khai báo **lấy row nào** (bindings)
và **dựng thành chat turn nào** (messages). Có sẵn binding mặc định cho VQA
(`recipe.py:33`):

```python
"vqa":       "emitted_at(t, style=vqa, role=assistant)"
"vqa_query": "emitted_at(t, style=vqa, role=user)"
```

Recipe VQA cho một camera (theo đúng mẫu trong `language_and_recipes.mdx`):

```yaml
# vqa_vi.yaml — một recipe VQA tiếng Việt cho camera "top"
ask_vqa_top:
  bindings:
    vqa_query: "emitted_at(t, style=vqa, role=user, camera=observation.images.top)"
    vqa:       "emitted_at(t, style=vqa, role=assistant, camera=observation.images.top)"
  messages:
    - role: user
      stream: high_level
      if_present: vqa_query
      content:
        - { type: image, feature: observation.images.top }
        - { type: text,  text: "${vqa_query}" }
    - role: assistant
      content: "${vqa}"
      stream: high_level
      target: true          # đây là phần model phải HỌC SINH RA
      if_present: vqa
```

Điểm cần nhớ:
- `target: true` đánh dấu turn là **mục tiêu huấn luyện** (model học sinh ra câu trả lời).
- `camera=...` là **bắt buộc** để tránh lỗi mơ hồ khi có nhiều camera; thêm một sub-recipe cho
  mỗi camera.
- `${task}` luôn đọc từ `meta/tasks.parquet`, độc lập với annotation.
- Recipe có thể **blend** nhiều mục tiêu (VQA / subtask / memory...) theo trọng số; mỗi frame
  được chọn một nhánh xác định theo sample index → train nhiều objective cùng lúc.

## 4. Tầng 3 — output cho policy

`RenderMessagesStep` (`render_messages_processor.py:34`, đã đăng ký tên
`render_messages_processor`) chuyển sample thành:

```python
sample["messages"]                 # HF-style chat messages (đa phương thức)
sample["message_streams"]          # nhãn high_level/low_level để lọc
sample["target_message_indices"]   # turn nào là mục tiêu (tính loss)
```

Sau đó tokenizer step (Bài 04) biến `messages` thành `OBS_LANGUAGE_TOKENS` cho VLM (Bài 09).
Toàn bộ chuỗi:

```
language_events(vqa, tiếng Việt) ─► recipe(vqa_vi.yaml) ─► RenderMessagesStep ─► messages
        ─► TokenizerStep ─► OBS_LANGUAGE_TOKENS ─► VLM backbone
```

## 5. Thực hành
```bash
# Đọc đầy đủ thiết kế 3 tầng
sed -n '1,200p' docs/source/language_and_recipes.mdx

# Xem registry style + binding mặc định
grep -n "CORE_STYLES\|EVENT_ONLY_STYLES\|VIEW_DEPENDENT_STYLES" src/lerobot/datasets/language.py
grep -n "vqa" src/lerobot/configs/recipe.py
```

Tự dựng một mẫu VQA trong bộ nhớ và thử resolver (khái niệm — đọc `language_render.py` để có
API chính xác):
```bash
sed -n '1,60p' src/lerobot/datasets/language_render.py
```

## 6. Tự kiểm tra
1. Một mẫu VQA cần mấy row, ở cột nào, ràng buộc gì về `camera`?
2. `target: true` trong recipe có ý nghĩa gì khi tính loss?
3. Ba key sidecar mà `RenderMessagesStep` sinh ra là gì, dùng để làm gì?

➡️ Tiếp theo: [11 — Annotation pipeline & sinh VQA tự động](./11-annotation-pipeline.md)
