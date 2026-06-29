# Bài 07 — Eval, inference & rollout

## Mục tiêu
- Phân biệt 3 kiểu "đánh giá": eval-loss, rollout trong env, inference thật.
- Chạy được `lerobot-eval` / `lerobot-rollout`.
- Hiểu cách gọi policy ngoài framework (quan trọng khi bạn build demo VQA).

## 1. Ba mức đánh giá

| Mức | Cần gì | Đo gì | Script/flag |
|---|---|---|---|
| Eval-loss | Held-out episodes | Loss trên dữ liệu chưa thấy | `--eval_steps`, `--eval_split` (trong train) |
| Rollout (sim) | `EnvConfig` (Gymnasium) | Success rate / reward khi policy điều khiển | `lerobot-eval`, `--env_eval_freq` |
| Inference thật | Robot/observation thật | Hành vi thực tế | `lerobot-record` (policy mode), async inference |

Với dự án **VQA tiếng Việt**, "đánh giá" của bạn sẽ khác robotics một chút: bạn quan tâm
**chất lượng câu trả lời** (text) hơn là success-rate điều khiển. Bạn sẽ tự định nghĩa metric
(accuracy/BLEU/CIDEr/LLM-judge) — nhưng vẫn tận dụng được hạ tầng load model + processor ở đây.

## 2. lerobot-eval (rollout trong sim)

`lerobot-eval` → `scripts/lerobot_eval.py`, nhận `EvalConfig` (`configs/eval.py`) + policy +
env. Đại ý:

```bash
uv run lerobot-eval \
  --policy.path=outputs/train/act_demo/checkpoints/last/pretrained_model \
  --env.type=libero \
  --eval.n_episodes=10 \
  --eval.batch_size=5
```

Env có sẵn (`src/lerobot/envs/`): `libero`, `metaworld`, `robocasa`, `robotwin`, `vlabench`,
`robomme`. Nhiều env cần extra riêng (Bài 03 cách xem extras).

> Mới (commit `6f0ba4be`): có thể **ghi lại rollout thành LeRobot dataset** — hữu ích để tái
> sử dụng dữ liệu eval làm dữ liệu train/annotate.

## 3. lerobot-rollout & async inference

- `scripts/lerobot_rollout.py` — chạy policy theo từng bước trong env, thu episode.
- `src/lerobot/async_inference/` + `docs/source/async.mdx`, `inference.mdx` — kiến trúc
  client/server tách việc *suy luận model* khỏi *vòng điều khiển robot* (giảm độ trễ).
  Liên quan `rtc` policy (real-time chunking).

## 4. Gọi policy ngoài framework (cho demo VQA)

Đây là pattern bạn sẽ dùng nhiều khi prototyping VLM tiếng Việt:

```python
# Pseudo-pattern; tên processor có thể khác — xem processor/factory.py
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

policy = SmolVLAPolicy.from_pretrained("lerobot/smolvla_base")
policy.eval()

# Tự dựng batch: ảnh + state + ngôn ngữ (đã qua preprocessor),
# rồi:
# action = policy.select_action(batch)
```

`from_pretrained` đến từ `HubMixin` (Bài 05). Với VQA, bạn sẽ thay/đắp thêm một "head ngôn ngữ"
để model **sinh text trả lời** thay vì chỉ action (Bài 13).

## 5. Thực hành
1. Eval checkpoint ACT bạn train ở Bài 06 (cần cài extra env tương ứng, ví dụ libero).
2. Đọc `docs/source/inference.mdx` để nắm pattern load + preprocess + select_action.
3. Thử `from_pretrained` một VLA nhỏ và in `policy.config` để thấy cấu hình.

## 6. Tự kiểm tra
1. 3 mức đánh giá khác nhau ở đầu vào và metric thế nào?
2. Async inference giải quyết vấn đề gì?
3. Với VQA, metric "success rate" của robotics có còn phù hợp không? Bạn thay bằng gì?

➡️ Tiếp theo: [08 — Tổng quan VLA](./08-vla-tong-quan.md)
