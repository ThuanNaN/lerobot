# Bài 12 — Thêm dataset VQA tiếng Việt

## Mục tiêu
- Biết các cách đưa dữ liệu VQA tiếng Việt vào định dạng LeRobot.
- Tự tạo dataset bằng code và đính kèm cột ngôn ngữ `vqa`.
- Hiểu khi nào dùng dataset robot thật, khi nào dùng dataset ảnh-tĩnh.

> Đây là bước "addons dataset" trong kế hoạch của bạn. Code tham chiếu:
> `datasets/lerobot_dataset.py`, `datasets/language.py`, `examples/dataset/`,
> `examples/port_datasets/`.

## 1. Hai kịch bản dữ liệu

LeRobot vốn cho dữ liệu robot (ảnh + state + action theo thời gian). Dự án VQA của bạn có thể
thuộc một trong hai dạng:

- **A. VQA gắn robot/episode** — bạn có episode robot và muốn hỏi-đáp về cảnh trong khi robot
  hoạt động. Khớp tự nhiên với format LeRobot (dùng cột `language_events` style `vqa`).
- **B. VQA ảnh-tĩnh thuần** (giống VQAv2/ViVQA) — chỉ ảnh + câu hỏi + câu trả lời, không có
  action. Bạn vẫn nhúng được vào LeRobot bằng cách coi mỗi ảnh là một "episode 1 frame" với
  feature ảnh + cột `language_events`, **không có** `action`.

> Quyết định kiến trúc quan trọng: nếu mục tiêu nghiên cứu của bạn là **VQA thuần** (không
> điều khiển robot), cân nhắc dùng LeRobot chủ yếu cho *backbone VLM + training loop + processor*,
> còn dataloader có thể là dataset HF thường. Nhưng đi theo format LeRobot giúp tái dùng recipe
> + RenderMessagesStep + tokenizer sẵn có (Bài 10). Chọn có chủ đích.

## 2. Cấu trúc tối thiểu cần khai báo

Một dataset cần `features` (Bài 02). Cho VQA ảnh-tĩnh tối thiểu:

```python
features = {
    "observation.images.main": {        # ảnh
        "dtype": "video",               # hoặc "image"
        "shape": (3, H, W),
        "names": ["channels", "height", "width"],
    },
    # KHÔNG cần "action" nếu là VQA thuần.
    # (Nếu muốn giữ schema robot, có thể thêm observation.state/action giả.)
}
```

Câu hỏi-đáp tiếng Việt **không** nằm trong `features` mà nằm trong cột ngôn ngữ
`language_events` (style `vqa`) — xem Bài 10.

## 3. Tạo dataset bằng code (khung)

Pattern chuẩn dùng `add_frame` + `save_episode` (Bài 02):

```python
from lerobot.datasets.lerobot_dataset import LeRobotDataset

ds = LeRobotDataset.create(
    repo_id="<user>/vivqa_lerobot",
    fps=1,                       # ảnh tĩnh: 1 frame/episode là đủ
    features=features,
    # ... xem chữ ký create() trong lerobot_dataset.py để biết tham số chính xác
)

for sample in your_vietnamese_vqa_iterable:      # ảnh + câu hỏi + câu trả lời (tiếng Việt)
    ds.add_frame({
        "observation.images.main": sample.image,        # numpy/tensor HWC
        "task": sample.task or "Trả lời câu hỏi về ảnh", # task string (tasks.parquet)
    })
    # đính cặp VQA tiếng Việt vào language_events (style=vqa, kèm camera!)
    # -> xem datasets/language.py để biết API ghi cột ngôn ngữ chính xác
    ds.save_episode()
```

> Chữ ký `LeRobotDataset.create(...)`, cách add language rows, và tên tham số chính xác phải
> đọc trực tiếp trong `src/lerobot/datasets/lerobot_dataset.py` (hàm `create`, `add_frame`,
> `save_episode`) và `datasets/language.py` (`column_for_style`, schema row). Đừng đoán — API
> có thể đổi theo version.

## 4. Cách dễ hơn: annotate dataset có sẵn

Nếu bạn đã có episode/ảnh ở format LeRobot mà chưa có VQA, **đừng tự ghi tay** — dùng
`lerobot-annotate` (Bài 11) với prompt tiếng Việt để sinh cột `vqa` tự động, rồi soát lại.

## 5. Port từ dataset công khai

`examples/port_datasets/` có script chuyển dữ liệu ngoài → format LeRobot. Dùng làm khuôn để
viết script port một bộ VQA tiếng Việt có sẵn (ví dụ ViVQA, UIT-... ) vào LeRobot:

```bash
ls examples/port_datasets examples/dataset
sed -n '1,60p' examples/dataset/*.py 2>/dev/null | head -80
```

## 6. Kiểm thử dataset của bạn

```bash
uv run python - <<'PY'
from lerobot.datasets.lerobot_dataset import LeRobotDataset
ds = LeRobotDataset("<user>/vivqa_lerobot")   # hoặc đường dẫn local
s = ds[0]
print("keys:", list(s.keys()))
# kiểm tra có cột ngôn ngữ + ảnh
PY
```

Sau đó thử recipe VQA (Bài 10) + RenderMessagesStep để chắc dữ liệu render ra `messages` đúng.

## 7. Checklist
- [ ] Quyết định dạng A (gắn robot) hay B (ảnh tĩnh).
- [ ] Khai `features` (ảnh, optional state/action).
- [ ] Ghi cặp VQA tiếng Việt vào `language_events` (style=`vqa`, có `camera`).
- [ ] Viết `vqa_vi.yaml` recipe (Bài 10).
- [ ] Verify một sample render ra `messages` hợp lệ.
- [ ] (Khuyến nghị) tách train (tự sinh) / eval (thủ công, người Việt soát).

## 8. Tự kiểm tra
1. Câu hỏi-đáp tiếng Việt lưu trong `features` hay trong cột ngôn ngữ? Vì sao?
2. Hai hàm nào dùng để ghi dataset bằng code?
3. Khi nào nên dùng `lerobot-annotate` thay vì ghi tay?

➡️ Tiếp theo: [13 — Thêm policy/model VLM mới](./13-them-policy-model-moi.md)
