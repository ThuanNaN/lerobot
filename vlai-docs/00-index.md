# Giáo trình làm chủ LeRobot cho research VQA/VLM tiếng Việt

> Bộ tài liệu này được viết để bạn **làm chủ repo LeRobot** từ cơ bản đến nâng cao,
> với mục tiêu cuối cùng là dùng nó làm *nền tảng (base)* cho một dự án nghiên cứu
> **VQA / VLM cho tiếng Việt**: thêm dataset, thêm model, finetune và pretraining.

## Tại sao LeRobot phù hợp với hướng VQA/VLM?

LeRobot không chỉ là thư viện robot. Lõi của nó là một **pipeline huấn luyện multimodal**
(ảnh + ngôn ngữ + hành động) rất sạch và mở rộng được:

- Hàng loạt policy **VLA (Vision-Language-Action)** dựa trên VLM có sẵn:
  `smolvla`, `pi0`, `pi05`, `pi0_fast`, `eo1`, `groot`, `molmoact2`, `pi_gemma`,
  `xvla`, `wall_x`, `vla_jepa`, `multi_task_dit` (xem `src/lerobot/policies/`).
- Dataset **đã hỗ trợ sẵn style ngôn ngữ `vqa`** (xem `src/lerobot/datasets/language.py`,
  `src/lerobot/configs/recipe.py`) — nghĩa là format dữ liệu, recipe chat-turn, và
  pipeline render messages cho VQA đã tồn tại.
- Có sẵn **annotation pipeline** sinh câu hỏi-trả lời VQA tự động
  (`src/lerobot/annotations/steerable_pipeline/modules/general_vqa.py`).
- Hệ thống **config (draccus) + processor + registry** cho phép cắm dataset/model/processor
  mới mà không cần fork.

Nói cách khác: bạn có thể tận dụng hạ tầng training/eval/dataloader/multimodal của LeRobot,
rồi tập trung công sức vào phần *VLM tiếng Việt* của riêng bạn.

## Cách dùng bộ tài liệu

Mỗi bài là một file `.md` độc lập nhưng nên đọc theo thứ tự. Mỗi bài có cấu trúc:

- **Mục tiêu** — sau bài này bạn làm được gì.
- **Nội dung** — giải thích, có trỏ tới file/dòng code thật trong repo (dạng `path:line`).
- **Thực hành** — lệnh chạy được để bạn tự tay làm.
- **Tự kiểm tra** — câu hỏi để chắc chắn bạn đã hiểu.

> Quy ước: mọi lệnh Python nên chạy qua `uv run` (theo `CLAUDE.md`).

## Lộ trình

### Module 0 — Định hướng & môi trường
- [01 — Tổng quan kiến trúc & cài đặt](./01-tong-quan-va-setup.md)

### Module 1 — Nền tảng (bắt buộc nắm chắc)
- [02 — LeRobotDataset: format, episode, video, features](./02-dataset.md)
- [03 — Configs & draccus CLI (ChoiceRegistry)](./03-configs-cli.md)
- [04 — Processor pipeline (ProcessorStep, normalize, tokenizer)](./04-processor-pipeline.md)
- [05 — Policies: PreTrainedPolicy, factory, forward/select_action](./05-policies-co-ban.md)

### Module 2 — Training & Evaluation (trung cấp)
- [06 — Training loop end-to-end (lerobot-train)](./06-training.md)
- [07 — Eval, inference & rollout](./07-eval-inference.md)

### Module 3 — VLM/VLA (cốt lõi cho research)
- [08 — Tổng quan VLA: VLM + action expert](./08-vla-tong-quan.md)
- [09 — SmolVLA deep dive (code-level)](./09-smolvla-deep-dive.md)
- [10 — Language columns, recipes & VQA](./10-language-recipes-vqa.md)
- [11 — Annotation pipeline & sinh VQA tự động](./11-annotation-pipeline.md)

### Module 4 — Nâng cao: mở rộng cho dự án của bạn
- [12 — Thêm dataset VQA tiếng Việt](./12-them-dataset-tieng-viet.md)
- [13 — Thêm policy/model VLM mới](./13-them-policy-model-moi.md)
- [14 — Finetune & PEFT/LoRA](./14-finetune-peft-lora.md)
- [15 — Pretraining & roadmap dự án](./15-pretraining-va-roadmap.md)

## Tài liệu gốc trong repo nên đọc song song

- `AGENT_GUIDE.md` — hướng dẫn người dùng (setup, record, train, eval).
- `docs/source/*.mdx` — tài liệu chính thức của HuggingFace, đặc biệt:
  `language_and_recipes.mdx`, `annotation_pipeline.mdx`, `bring_your_own_policies.mdx`,
  `implement_your_own_processor.mdx`, `inference.mdx`, `lerobot-dataset-v3.mdx`.
- `examples/` — script chạy được theo từng use case.
