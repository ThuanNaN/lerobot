# Bài 06 — Training loop end-to-end (`lerobot-train`)

## Mục tiêu
- Chạy được một phiên train hoàn chỉnh và hiểu từng mảnh ghép.
- Đọc hiểu `scripts/lerobot_train.py` ở mức "luồng điều khiển".
- Biết checkpoint, resume, logging, optimizer/scheduler hoạt động ra sao.

## 1. Điểm vào

`lerobot-train` → `src/lerobot/scripts/lerobot_train.py`. Nó nhận `TrainPipelineConfig`
(Bài 03) qua draccus và thực thi vòng train. Luồng tổng quát:

```
parse TrainPipelineConfig
   ├─ make_dataset()              # LeRobotDataset + delta_timestamps
   ├─ make_policy()               # model (factory.py)
   ├─ make_pre/post processors    # pipeline tiền/hậu xử lý
   ├─ make_optimizer + scheduler  # từ preset của policy hoặc config
   └─ for step in range(steps):
         batch = next(dataloader)
         batch = preprocessor(batch)
         loss, _ = policy.forward(batch)
         loss.backward(); optimizer.step(); scheduler.step()
         if step % log_freq == 0:   log (wandb)
         if step % save_freq == 0:   save_checkpoint
         if step % env_eval_freq:    eval trong env (nếu có)
```

## 2. Các tham số bạn sẽ chỉnh nhiều

Từ `TrainPipelineConfig` (`configs/train.py`):

- Quy mô: `--batch_size`, `--steps`, `--num_workers`, `--prefetch_factor`.
- Checkpoint: `--save_freq`, `--output_dir`, `--resume=true`, `--save_checkpoint`.
- Eval: `--eval_steps` (loss trên held-out), `--env_eval_freq` (rollout trong sim),
  `--eval_split`, `--max_eval_samples`.
- Tối ưu: nếu `--use_policy_training_preset=true` (mặc định) thì optimizer/scheduler lấy từ
  preset của policy; tắt đi để tự khai báo `--optimizer.*`, `--scheduler.*`.
- Logging: `--wandb.enable=true --wandb.project=...`.
- PEFT/LoRA: `--peft.*` (Bài 14).

## 3. Train policy nhẹ (làm quen nhanh)

Train ACT trên một dataset SO-100 công khai, ít step để thấy vòng đời:

```bash
uv run lerobot-train \
  --policy.type=act \
  --dataset.repo_id=lerobot/svla_so100_stacking \
  --batch_size=8 \
  --steps=200 \
  --save_freq=200 \
  --output_dir=outputs/train/act_demo \
  --wandb.enable=false
```

Kết quả vào `outputs/train/act_demo/` gồm checkpoint + `train_config.json`. Resume:

```bash
uv run lerobot-train --config_path=outputs/train/act_demo/checkpoints/last/pretrained_model --resume=true
```

## 4. Finetune một VLA pretrained (xem trước Bài 14)

```bash
# Finetune SmolVLA base (cần: uv sync --extra smolvla)
uv run lerobot-train \
  --policy.path=lerobot/smolvla_base \
  --dataset.repo_id=<USER>/<your_dataset> \
  --batch_size=64 \
  --steps=20000
```

`--policy.path` = nạp **weights + config** từ Hub/local; `--policy.type` = khởi tạo mới từ đầu.

## 5. Tài nguyên & tốc độ
- VLA models nặng → cần GPU. Giảm `--batch_size`, bật gradient checkpointing / `--peft` (LoRA),
  hoặc dùng `act`/`diffusion` khi prototyping.
- `--num_workers`/`--prefetch_factor` ảnh hưởng tốc độ nạp video — tăng nếu CPU/đĩa nhàn rỗi.
- FSDP cho model lớn: repo mới hỗ trợ FSDP checkpoint saving (commit `73782447`); xem
  `docs/source` và config liên quan nếu train đa GPU.

## 6. E2E test như tài liệu sống

`Makefile` có các target E2E chạy train→eval thật trên dataset nhỏ. Đọc chúng để thấy "lệnh
chuẩn":

```bash
grep -nA3 "test-end-to-end\|act\|smolvla" Makefile | head -60
```

## 7. Thực hành
1. Chạy lệnh ACT ở mục 3, mở `outputs/.../train_config.json` xem config đã được lưu.
2. Bật `--wandb.enable=true` và quan sát loss curve.
3. Thử `--eval_steps=100 --eval_split=...` để có eval-loss trên held-out.

## 8. Tự kiểm tra
1. Khác nhau giữa `--policy.path` và `--policy.type`?
2. Optimizer mặc định đến từ đâu khi `use_policy_training_preset=true`?
3. Resume hoạt động dựa trên file/thư mục nào?

➡️ Tiếp theo: [07 — Eval, inference & rollout](./07-eval-inference.md)
