# Bài 11 — Annotation pipeline & sinh VQA tự động

## Mục tiêu
- Hiểu `lerobot-annotate`: dùng một VLM để **sinh annotation tự động** (gồm VQA) ghi vào dataset.
- Biết module `vqa` sinh cặp hỏi-đáp thế nào, để **bản địa hóa sang tiếng Việt**.
- Nắm pipeline staging → validate → write để mở rộng an toàn.

> Tài liệu gốc: `docs/source/annotation_pipeline.mdx`. Code:
> `src/lerobot/annotations/steerable_pipeline/`. Script: `scripts/lerobot_annotate.py`.

## 1. `lerobot-annotate` làm gì

Trỏ nó vào một LeRobot dataset; nó "xem" video từng episode bằng một VLM và **ghi annotation
ngôn ngữ trở lại** hai cột `language_persistent` / `language_events` (Bài 10), trực tiếp vào
`data/chunk-*/file-*.parquet`. Nó tạo ra: `plan`, `subtask`, `memory`, `interjection`,
`speech`, và **`vqa`**.

Kiến trúc (rút gọn từ doc):

```
dataset → read episodes → [ plan | interjections | vqa ] → staging(JSONL) → validator → writer → parquet
                                       ▲
                          một VLM server dùng chung (Qwen-VL qua vLLM/OpenAI API)
```

Ba module nằm ở `annotations/steerable_pipeline/modules/`:
`plan_subtasks_memory.py`, `interjections_and_speech.py`, **`general_vqa.py`**.

## 2. Module `vqa` (`modules/general_vqa.py`)

Cách hoạt động (đọc docstring đầu file):
- Cứ mỗi `1/hz` giây có một "emission tick"; mỗi tick neo `K` frame liên tiếp; **mỗi frame
  được neo sinh một cặp VQA**.
- Với dataset nhiều camera: **mỗi frame neo sinh một cặp `(vqa, user)` + `(vqa, assistant)`
  cho từng camera**, và đóng dấu trường `camera` tương ứng (đúng ràng buộc view-dependent ở
  Bài 10).
- Loại câu hỏi bao phủ: **bbox, keypoint, count, attribute, spatial**. Câu trả lời của
  assistant là một **chuỗi JSON** có schema tùy loại câu hỏi; JSON hỏng sẽ retry 1 lần
  (`VlmClient.generate_json`).

Prompt điều khiển VLM nằm ở `annotations/steerable_pipeline/prompts/vqa.txt`. **Đây là nơi
bản địa hóa tiếng Việt**: chỉnh prompt để VLM sinh câu hỏi/câu trả lời bằng tiếng Việt.

## 3. Các mảnh phụ trợ trong pipeline
- `config.py` — `AnnotationPipelineConfig`, `VqaConfig` (hz, K, loại câu hỏi...).
- `vlm_client.py` — `make_vlm_client` (kết nối VLM server, ví dụ `Qwen/Qwen2.5-VL-7B-Instruct`).
- `frames.py` — lấy frame/ảnh từ video (`make_frame_provider`, `to_image_blocks`).
- `staging.py` / `validator.py` / `writer.py` — ghi tạm JSONL → kiểm tra → ghi vào parquet.
- `executor.py` — điều phối toàn bộ.

## 4. Chạy thử (khái niệm)

```bash
uv run lerobot-annotate \
  --root=/path/to/your_dataset \
  --vlm.model_id=Qwen/Qwen2.5-VL-7B-Instruct
# Phân tán: xem examples/annotations/run_hf_job.py
```

Xem các tham số:
```bash
uv run lerobot-annotate --help
sed -n '1,80p' src/lerobot/annotations/steerable_pipeline/config.py
```

## 5. Chiến lược cho VQA tiếng Việt

Bạn có 3 con đường, tăng dần công sức:

1. **Sinh tự động bằng VLM đa ngôn ngữ + prompt tiếng Việt.** Đổi `prompts/vqa.txt` sang
   tiếng Việt và chọn `--vlm.model_id` là VLM hỗ trợ tiếng Việt tốt (Qwen-VL...). Nhanh, nhưng
   cần kiểm định chất lượng câu sinh ra.
2. **Sinh tự động rồi dịch/hiệu đính.** Dùng pipeline có sẵn (tiếng Anh), sau đó dịch máy +
   người soát. Cân bằng chất lượng/chi phí.
3. **Annotate thủ công / bán tự động.** Tự ghi cặp VQA tiếng Việt vào cột `language_events`
   bằng code (Bài 12) — chất lượng cao nhất, hợp cho tập eval/benchmark chuẩn.

> Gợi ý nghiên cứu: tách rõ **train set** (có thể sinh tự động, nhiều) và **eval/benchmark set**
> (thủ công, chuẩn, có người Việt soát) để báo cáo trung thực.

## 6. Thực hành
1. Đọc `general_vqa.py` và liệt kê 5 loại câu hỏi + schema JSON mỗi loại.
2. Mở `prompts/vqa.txt`, hình dung bản dịch tiếng Việt.
3. Đọc `validator.py` để biết ràng buộc nào sẽ bị từ chối (tránh ghi dữ liệu sai schema).

## 7. Tự kiểm tra
1. Module `vqa` ghi dữ liệu vào cột nào, kèm trường bắt buộc nào?
2. Vì sao mỗi camera lại có cặp VQA riêng?
3. Ba chiến lược tạo VQA tiếng Việt khác nhau ở chất lượng/chi phí thế nào?

➡️ Tiếp theo: [12 — Thêm dataset VQA tiếng Việt](./12-them-dataset-tieng-viet.md)
