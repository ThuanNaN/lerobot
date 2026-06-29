# Bài 02 — LeRobotDataset: format, episode, video, features

## Mục tiêu
- Hiểu một `LeRobotDataset` gồm những gì trên đĩa và khi load vào RAM.
- Hiểu khái niệm **episode**, **feature**, **delta timestamps**, **video decoding**.
- Biết một sample (`__getitem__`) trả về cấu trúc gì — vì đó chính là đầu vào model.

## 1. LeRobotDataset là gì?

`LeRobotDataset` (`src/lerobot/datasets/lerobot_dataset.py:46`) kế thừa
`torch.utils.data.Dataset`. Nó là lớp trung tâm: gói dữ liệu robot (ảnh nhiều camera,
state, action, ngôn ngữ) theo dạng **episode-aware** + giải mã video on-the-fly.

Hai lớp chính:
- `LeRobotDataset` — dataset để train/đọc.
- `LeRobotDatasetMetadata` — metadata (features, fps, số episode, thống kê...).

## 2. Cấu trúc trên đĩa (format v3)

Tham khảo `docs/source/lerobot-dataset-v3.mdx`. Đại khái một dataset gồm:

```
<repo_id>/
├── meta/
│   ├── info.json            # fps, features, codebase version, số episode/frame
│   ├── stats.json           # mean/std/min/max cho normalize
│   ├── episodes/...         # metadata theo episode
│   └── tasks.parquet        # ánh xạ task string <-> task index
├── data/
│   └── chunk-*/file-*.parquet   # các cột frame: state, action, timestamp, index,
│                                # (tùy chọn) language_persistent, language_events
└── videos/
    └── chunk-*/<camera_key>/file-*.mp4   # ảnh quan sát lưu dạng video
```

Điểm cần nhớ cho VQA: **text/ngôn ngữ nằm trong các cột parquet** (`task`, và tùy chọn
`language_persistent`/`language_events`), còn **ảnh nằm trong video**. Xem Bài 10.

## 3. Features — "schema" của dataset

Mỗi dataset khai báo `features`: một dict mô tả từng kênh dữ liệu (tên, dtype, shape, loại).
Loại feature (`FeatureType`) gồm `VISUAL`, `STATE`, `ACTION`, `ENV`,... (xem
`src/lerobot/configs/types.py`). Đây là cầu nối giữa dataset và policy: policy đọc
`input_features`/`output_features` để biết nó nhận/sinh gì (Bài 05).

Ví dụ feature key thường gặp: `observation.images.<camera>`, `observation.state`, `action`.

## 4. Episode, index và delta timestamps

- Dataset được chia theo **episode** (một lần demo liên tục). Sampler tôn trọng ranh giới
  episode (`src/lerobot/datasets/sampler.py`).
- `delta_timestamps`: cho phép một sample lấy **nhiều khung thời gian** quanh frame hiện tại
  (ví dụ lịch sử quan sát, hoặc chuỗi action tương lai — "action chunk"). Đây là lý do
  policy như ACT/SmolVLA dự đoán cả một *chuỗi* action.

## 5. Một sample trông như thế nào

`__getitem__` (`lerobot_dataset.py:475`) trả về một `dict[str, Tensor]` (cộng vài metadata).
Khái niệm: với mỗi key trong `features`, bạn nhận tensor tương ứng; video được decode thành
tensor ảnh; cột `task` thành chuỗi (sau đó được tokenize ở processor).

## 6. Tạo / ghi dataset

Hai phương thức quan trọng:
- `add_frame(frame: dict)` (`:398`) — thêm một frame vào episode đang ghi.
- `save_episode(...)` (`:414`) — chốt và lưu episode (kèm encode video).

Trong thực tế, dữ liệu robot được ghi bằng `lerobot-record`. Nhưng cho dự án VQA tiếng Việt,
bạn sẽ **tự tạo dataset bằng code** (Bài 12) qua đúng 2 hàm này.

## 7. Thực hành

Xem nhanh metadata một dataset công khai (không cần tải full):

```bash
uv run lerobot-info --repo-id lerobot/svla_so100_stacking 2>/dev/null || \
uv run python - <<'PY'
from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
m = LeRobotDatasetMetadata("lerobot/svla_so100_stacking")
print("fps:", m.fps)
print("episodes:", m.total_episodes, "frames:", m.total_frames)
print("features:", list(m.features.keys()))
PY
```

Load và soi một sample:

```bash
uv run python - <<'PY'
from lerobot.datasets.lerobot_dataset import LeRobotDataset
ds = LeRobotDataset("lerobot/svla_so100_stacking")
sample = ds[0]
for k, v in sample.items():
    shape = getattr(v, "shape", None)
    print(f"{k:35s} {type(v).__name__:10s} {shape}")
PY
```

Trực quan hóa dataset (script có sẵn):

```bash
uv run lerobot-dataset-viz --help
```

## 8. Tự kiểm tra
1. Ảnh quan sát được lưu ở đâu trên đĩa, text task lưu ở đâu?
2. `delta_timestamps` dùng để làm gì? Vì sao policy cần nó để xuất "action chunk"?
3. Hai hàm nào dùng để tự tạo dataset bằng code?

➡️ Tiếp theo: [03 — Configs & draccus CLI](./03-configs-cli.md)
