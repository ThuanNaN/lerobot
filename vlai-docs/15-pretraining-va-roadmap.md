# Bài 15 — Pretraining & roadmap dự án VQA/VLM tiếng Việt

## Mục tiêu
- Hiểu khi nào cần pretraining (vs chỉ finetune) và pretrain cái gì.
- Có một **roadmap nghiên cứu** cụ thể, theo từng giai đoạn, dùng LeRobot làm nền.
- Biết các rủi ro/đánh đổi và cách báo cáo kết quả trung thực.

## 1. Pretraining nghĩa là gì ở đây?

Với VQA/VLM, "pretraining" có thể là một trong ba mức (tăng dần chi phí):

1. **Adapter/projection pretraining** — chỉ huấn luyện lớp nối vision↔language trên cặp
   ảnh-caption tiếng Việt (rẻ, thường đủ để "mở khóa" tiếng Việt cho một backbone đa ngôn ngữ).
2. **Continued pretraining backbone** — tiếp tục huấn luyện LLM/VLM backbone trên **corpus
   tiếng Việt** lớn (đắt, nhưng nâng nền tảng ngôn ngữ).
3. **From-scratch** — gần như không nên cho một dự án đơn lẻ; tốn kém và khó cạnh tranh.

> Nguyên tắc: **đừng pretrain nếu finetune (Bài 14) đã đủ.** Chỉ pretrain phần nào là nút thắt
> thực sự (thường là tokenizer/ngôn ngữ tiếng Việt của backbone).

## 2. Hai nút thắt tiếng Việt cần đo trước

Làm 2 thí nghiệm chẩn đoán nhỏ **trước khi** quyết định pretrain:

- **Tokenizer fertility**: số token trung bình / từ tiếng Việt. Nếu quá cao → cân nhắc
  mở rộng/đổi tokenizer (LeRobot có `lerobot-train-tokenizer` →
  `scripts/lerobot_train_tokenizer.py`, hữu ích cho action tokenizer kiểu pi0-fast; với text
  tiếng Việt bạn thường dùng/đổi tokenizer của chính backbone VLM).
- **Zero-shot tiếng Việt**: cho backbone trả lời vài câu VQA tiếng Việt chưa finetune. Nếu nó
  đã "hiểu" kha khá → chỉ cần finetune; nếu vỡ hoàn toàn → cần continued pretraining.

## 3. Roadmap nghiên cứu đề xuất (theo giai đoạn)

### Giai đoạn 0 — Làm chủ repo (Bài 01–11)
- Chạy được train/eval ACT và SmolVLA.
- Hiểu rõ luồng `dataset → recipe → RenderMessages → tokenizer → VLM` (Bài 10).
- **Sản phẩm**: một notebook chạy được, ghi chú điểm cắm.

### Giai đoạn 1 — Dữ liệu (Bài 11–12)
- Chọn/port một bộ VQA tiếng Việt (hoặc sinh bằng `lerobot-annotate` + prompt tiếng Việt).
- Tách **train (lớn, có thể tự sinh)** và **eval/benchmark (nhỏ, thủ công, người Việt soát)**.
- **Sản phẩm**: dataset format LeRobot + recipe `vqa_vi.yaml` verify được.

### Giai đoạn 2 — Baseline (Bài 13–14)
- Chọn backbone đa ngôn ngữ (Qwen-VL / Gemma-vision...), gói thành policy plugin `my_vqa`.
- Finetune LoRA trên train set. Đo trên eval set (accuracy + LLM-judge + ví dụ định tính).
- **Sản phẩm**: baseline số đầu tiên + bảng so sánh backbone.

### Giai đoạn 3 — Cải thiện tiếng Việt (Bài 15)
- Nếu nút thắt là ngôn ngữ: continued pretraining nhẹ (adapter/projection) trên cặp
  ảnh-caption tiếng Việt; hoặc mở rộng tokenizer.
- Ablation: full vs LoRA; backbone A vs B; có/không continued-pretrain.
- **Sản phẩm**: bảng ablation, phân tích lỗi (loại câu hỏi nào yếu: spatial? count?).

### Giai đoạn 4 — Đóng gói & công bố
- Viết model card + dataset card (LeRobot có `datasets/card_template.md`).
- Cân nhắc đóng góp in-tree (Bài 13, Path B) nếu hữu ích cho cộng đồng.

## 4. Hạ tầng train quy mô lớn
- **PEFT/LoRA** (Bài 14) cho đa số thí nghiệm.
- **FSDP** cho continued pretraining backbone lớn — repo có FSDP checkpoint saving
  (commit `73782447`); xem config liên quan trong `configs/` và docs.
- **Streaming dataset** (`datasets/streaming_dataset.py`, `examples/training/train_with_streaming.py`)
  khi corpus quá lớn để tải hết.

## 5. Đánh giá trung thực (rất quan trọng cho research)
- Không đánh giá trên dữ liệu tự sinh bằng chính loại VLM bạn dùng để train (rò rỉ/thiên lệch).
- Báo cáo cả **định lượng** (accuracy theo loại câu hỏi) và **định tính** (ví dụ thất bại).
- Ghi rõ phần nào tự sinh, phần nào người soát; seed; số liệu được tính trên tập nào.
- Nêu giới hạn: kích thước dữ liệu, thiên lệch domain (robot vs ảnh đời thường), v.v.

## 6. Sai lầm thường gặp cần tránh
- Pretrain khi finetune đã đủ → tốn tài nguyên vô ích.
- Bỏ qua đo tokenizer → model "học chậm" mà không hiểu vì sao.
- Trộn train/eval (đặc biệt khi cùng nguồn tự sinh) → số đẹp giả.
- Sửa lõi LeRobot khi đáng lẽ chỉ cần plugin + recipe → khó maintain, khó rebase.

## 7. Tự kiểm tra
1. Ba mức pretraining khác nhau ở chi phí và cái được nâng cấp thế nào?
2. Hai thí nghiệm chẩn đoán nào nên làm trước khi quyết pretrain?
3. Vì sao tách train/eval và minh bạch nguồn dữ liệu lại quyết định độ tin cậy của báo cáo?

---

## Tổng kết giáo trình
Bạn đã đi từ: kiến trúc & setup (01) → nền tảng dataset/config/processor/policy (02–05) →
train/eval (06–07) → VLA/VLM & VQA (08–11) → mở rộng dataset/model/finetune/pretrain (12–15).

Bước tiếp theo thực tế: quay lại **Giai đoạn 0** ở mục 3, chạy thật một vòng SmolVLA + một
recipe VQA nhỏ tiếng Việt, rồi tiến dần theo roadmap.

⬅️ Quay lại [00 — Index](./00-index.md)
