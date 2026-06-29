# Bài 14 — Finetune & PEFT/LoRA

## Mục tiêu
- Phân biệt full finetune, partial finetune và **PEFT/LoRA**.
- Dùng `--peft.*` để finetune VLM tiết kiệm GPU.
- Áp dụng cho finetune VQA tiếng Việt từ một backbone đa ngôn ngữ.

> Code: `configs/default.py` (`PeftConfig`, `:117`), field `peft` trong `TrainPipelineConfig`.
> SmolVLA đã có sẵn target LoRA mặc định (Bài 09, `_get_default_peft_targets`).

## 1. Ba mức finetune

| Mức | Train gì | Khi nào | Chi phí |
|---|---|---|---|
| Full finetune | Toàn bộ tham số | Dữ liệu lớn, GPU mạnh | Cao nhất |
| Partial | Một phần (vd chỉ action expert / chỉ head) | Giữ backbone, học đầu ra | Trung bình |
| **PEFT/LoRA** | Adapter rank thấp chèn vào layer | **GPU hạn chế, lặp nhanh — chọn cái này trước** | Thấp |

SmolVLA mặc định `train_expert_only=True` (partial) và `freeze_vision_encoder=True`. Cho VQA
bạn thường cần mở thêm phần **ngôn ngữ** của backbone (qua LoRA) để model học sinh tiếng Việt.

## 2. PEFT config (`configs/default.py:117`)

Các field chính (bật bằng `--peft.*` trên CLI):

| Field | Mặc định | Ý nghĩa |
|---|---|---|
| `method_type` | `"LORA"` | Loại adapter PEFT |
| `r` | `16` | Rank — cao hơn = nhiều tham số train hơn, gần full finetune hơn |
| `lora_alpha` | `None` (PEFT mặc định 8) | Hệ số scale `alpha/r`; thường đặt `= r` hoặc `2*r` |
| `target_modules` | `None` | Module nào gắn adapter (string/list/regex hoặc `"all-linear"`); nhiều policy có default |
| `full_training_modules` | `None` | Module train full + lưu kèm adapter (vd projection mới tạo) |
| `init_type` | `None` | Cách khởi tạo adapter |

## 3. Lệnh finetune LoRA

```bash
uv run lerobot-train \
  --policy.path=lerobot/smolvla_base \
  --dataset.repo_id=<user>/vivqa_lerobot \
  --batch_size=16 \
  --steps=20000 \
  --peft.method_type=LORA \
  --peft.r=16 \
  --peft.lora_alpha=32 \
  --output_dir=outputs/train/vivqa_lora
```

Với SmolVLA, nếu không set `--peft.target_modules`, nó dùng default
(`_get_default_peft_targets` — q/v proj của `lm_expert` + vài projection chung). Cho VQA bạn
có thể muốn target rộng hơn (bao gồm attention của phần LLM ngôn ngữ) — override
`--peft.target_modules` bằng regex phù hợp với backbone bạn chọn.

## 4. Mẹo finetune cho tiếng Việt
- **Bắt đầu nhỏ**: `r=8–16`, `batch_size` vừa GPU, `steps` vài nghìn để kiểm tra pipeline trước.
- **Mở đúng layer**: nếu mục tiêu là sinh text tiếng Việt, LoRA cần chạm tới attention/MLP của
  phần LLM, không chỉ action head.
- **`full_training_modules=[]`** khi finetune một policy *đã* được train, để không vô tình
  train lại các module mới (đọc chú thích trong `default.py`).
- **Tăng `tokenizer_max_length`** (Bài 09/13) nếu câu hỏi/đáp tiếng Việt bị cắt.
- **Theo dõi eval-loss** (`--eval_steps`) trên tập eval tiếng Việt giữ riêng (Bài 11/12).

## 5. Lưu & nạp adapter
LoRA adapter được lưu trong checkpoint (cạnh `train_config.json`). Khi eval/infer, nạp qua
`--policy.path=<checkpoint>` như thường; PEFT sẽ ráp adapter vào backbone.

## 6. Thực hành
1. Finetune LoRA SmolVLA trên dataset VQA tiếng Việt nhỏ, `steps=500`, xem nó chạy.
2. Đổi `--peft.r` 8→32 và quan sát số tham số train + tốc độ.
3. Đọc `_get_default_peft_targets` của SmolVLA, thử override `target_modules`.

## 7. Tự kiểm tra
1. LoRA tiết kiệm gì so với full finetune? `r` và `lora_alpha` ảnh hưởng thế nào?
2. Vì sao cho VQA cần LoRA chạm tới phần LLM ngôn ngữ chứ không chỉ action head?
3. Khi nào đặt `full_training_modules=[]`?

➡️ Tiếp theo: [15 — Pretraining & roadmap dự án](./15-pretraining-va-roadmap.md)
