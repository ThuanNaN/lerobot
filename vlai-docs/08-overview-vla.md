# Bài 08 — Tổng quan VLA: VLM + action expert

## Mục tiêu
- Hiểu kiến trúc chung của các policy **VLA (Vision-Language-Action)** trong LeRobot.
- Định vị các thành phần dùng lại được cho **VQA/VLM tiếng Việt**.
- Chọn được một policy làm điểm khởi đầu cho dự án.

## 1. VLA là gì và liên quan VQA thế nào?

VLA = **Vision-Language-Action**. Công thức chung:

```
[ảnh nhiều camera]  ─┐
[ngôn ngữ/lệnh]     ─┼──>  VLM backbone (vision encoder + LLM)  ──>  Action expert/head  ──> action chunk
[state robot]       ─┘                         │
                                               └────────────> (có thể) đầu ra ngôn ngữ
```

VQA = **Visual Question Answering**: cho ảnh + câu hỏi → sinh câu trả lời (text).

Điểm mấu chốt: **một VLA và một mô hình VQA dùng chung backbone VLM (vision encoder + LLM)**.
Khác biệt chỉ ở "đầu ra": VLA xuất *action*, VQA xuất *text*. Vì LeRobot đã có sẵn:
- pipeline đưa **ảnh + ngôn ngữ** vào VLM,
- format dataset có **cột ngôn ngữ + style `vqa`**,
- recipe dựng **chat turns hỏi-đáp**,

→ bạn có thể tái dùng backbone + dataloader + processor, và chỉ cần gắn/đổi **head sinh text**
(hoặc dùng thẳng đầu ra ngôn ngữ của LLM) để làm VQA. Đây là chiến lược trung tâm của dự án.

## 2. Các họ VLA trong repo (`src/lerobot/policies/`)

| Policy | Backbone/ý tưởng | Vì sao đáng chú ý cho bạn |
|---|---|---|
| **`smolvla`** | SmolVLM + action expert (HF, nhẹ, mở) | **Khởi đầu tốt nhất**: nhỏ, code dễ đọc, FOSS. Bài 09 mổ xẻ |
| `pi0`, `pi05`, `pi0_fast` | π0 flow-matching VLA | Mạnh, chuẩn công nghiệp; nặng hơn |
| `pi_gemma` | VLA dựa Gemma | Gemma có hỗ trợ đa ngôn ngữ tốt → cân nhắc cho tiếng Việt |
| `eo1`, `groot`, `molmoact2`, `xvla`, `wall_x`, `vla_jepa` | Các kiến trúc VLA khác | Tham khảo ý tưởng head/biểu diễn |
| `multi_task_dit` | Diffusion Transformer đa nhiệm | Không VLM nhưng đa nhiệm có điều kiện |

Mỗi policy có cùng bộ file chuẩn: `configuration_*.py` (config), `modeling_*.py` (model),
`processor_*.py` (pipeline riêng), `README.md` (paper + lệnh). Đây là **khuôn mẫu** bạn sẽ
nhân bản khi tạo model mới (Bài 13).

## 3. Hai thành phần dùng lại được cho VQA

1. **Vision encoder + LLM (VLM backbone)** — nhận ảnh + token ngôn ngữ. Trong SmolVLA là
   `SmolVLMWithExpertModel` (`policies/smolvla/smolvlm_with_expert.py`).
2. **Cơ chế đưa ngôn ngữ vào** — token hóa qua tokenizer của backbone và đặt vào batch dưới
   `OBS_LANGUAGE_TOKENS` / `OBS_LANGUAGE_ATTENTION_MASK` (xem `utils/constants.py`).

Cái bạn sẽ **thay/thêm**: đường ra. Thay vì (chỉ) action expert, thêm một LM head để sinh
câu trả lời, hoặc dùng generation của chính LLM backbone.

## 4. Chọn backbone hỗ trợ tiếng Việt

Tiếng Việt phụ thuộc nhiều vào tokenizer + dữ liệu pretrain của LLM backbone. Tiêu chí chọn:
- Tokenizer xử lý tốt dấu tiếng Việt (không vỡ thành quá nhiều byte).
- LLM backbone đã thấy tiếng Việt khi pretrain (đa ngôn ngữ).
- Gợi ý cân nhắc: backbone họ Gemma/Qwen/đa ngôn ngữ; hoặc tiếp tục pretrain/finetune
  backbone trên corpus tiếng Việt (Bài 15).

> Việc đánh giá tokenizer tiếng Việt là một thí nghiệm nhỏ nên làm sớm (Bài 13/15).

## 5. Thực hành
```bash
# So sánh cấu trúc file giữa các VLA để thấy khuôn mẫu chung
ls src/lerobot/policies/smolvla src/lerobot/policies/pi0 src/lerobot/policies/pi_gemma 2>/dev/null

# Đọc README có sẵn của từng policy (paper + lệnh train)
sed -n '1,40p' src/lerobot/policies/smolvla/README.md
```

## 6. Tự kiểm tra
1. VLA và VQA dùng chung phần nào của model? Khác nhau ở đâu?
2. Trong SmolVLA, lớp nào đóng vai trò backbone VLM?
3. Vì sao lựa chọn backbone lại quyết định chất lượng tiếng Việt?

➡️ Tiếp theo: [09 — SmolVLA deep dive](./09-smolvla-deep-dive.md)
