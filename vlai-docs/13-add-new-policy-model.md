# Bài 13 — Thêm policy/model VLM mới

## Mục tiêu
- Biết 3 mảnh ghép bắt buộc của một policy mới và cách đăng ký.
- Biết 2 con đường: plugin (out-of-tree) vs in-tree.
- Áp dụng cho mục tiêu: một **VLM tiếng Việt** sinh câu trả lời VQA.

> Tài liệu gốc (đọc kỹ): `docs/source/bring_your_own_policies.mdx`. Mẫu code:
> mọi thư mục trong `src/lerobot/policies/` (đặc biệt `smolvla/`).

## 1. Ba mảnh ghép của một policy

Tên policy (`my_vqa`) là "load-bearing": phải khớp ở cả 3 nơi.

1. **Configuration class** — kế thừa `PreTrainedConfig`, đăng ký bằng decorator:

```python
# configuration_my_vqa.py
from dataclasses import dataclass
from lerobot.configs import PreTrainedConfig
from lerobot.optim import AdamWConfig

@PreTrainedConfig.register_subclass("my_vqa")
@dataclass
class MyVqaConfig(PreTrainedConfig):
    vlm_model_name: str = "Qwen/Qwen2.5-VL-3B-Instruct"   # backbone đa ngôn ngữ
    tokenizer_max_length: int = 128                        # câu hỏi tiếng Việt dài hơn
    max_new_tokens: int = 64                               # độ dài câu trả lời

    optimizer_lr: float = 1e-4

    def validate_features(self) -> None:
        if not self.image_features:
            raise ValueError("MyVqa cần ít nhất một image feature.")

    def get_optimizer_preset(self) -> AdamWConfig:
        return AdamWConfig(lr=self.optimizer_lr)

    def get_scheduler_preset(self):
        return None

    @property
    def observation_delta_indices(self): return None      # VQA ảnh tĩnh: 1 frame
    @property
    def action_delta_indices(self): return [0]            # nếu không có action thật
    @property
    def reward_delta_indices(self): return None
```

2. **Policy class** — kế thừa `PreTrainedPolicy`, gán `config_class`, `name`, cài
   `forward()` (loss) và `select_action()` (Bài 05). Với VQA bạn sẽ thêm phương thức **generate**
   để sinh text:

```python
# modeling_my_vqa.py
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.utils.constants import OBS_LANGUAGE_TOKENS, OBS_LANGUAGE_ATTENTION_MASK

class MyVqaPolicy(PreTrainedPolicy):
    config_class = MyVqaConfig
    name = "my_vqa"

    def __init__(self, config, ...):
        super().__init__(config)
        # nạp backbone VLM (Qwen-VL...) + LM head sinh text
        # tái dùng cách đưa ảnh + OBS_LANGUAGE_TOKENS vào VLM như SmolVLA (Bài 09)

    def forward(self, batch):
        # teacher-forcing: tính cross-entropy trên câu trả lời (target_message_indices)
        loss = ...
        return loss, {"ce_loss": loss.item()}

    def generate_answer(self, batch):
        # sinh câu trả lời VQA bằng VLM.generate(...)
        ...

    def select_action(self, batch, **kw):
        # nếu không điều khiển robot, có thể trả action rỗng/no-op,
        # hoặc bọc generate_answer tùy thiết kế eval của bạn
        ...
    def reset(self): ...
```

> Khác biệt cốt lõi so với VLA: **output là text, loss là cross-entropy trên token câu trả lời**
> (không phải flow-matching trên action). Backbone + đường đưa ảnh/ngôn ngữ vào thì tái dùng.

3. **Processor factory** — hàm `make_<name>_pre_post_processors(...)` dựng pre/post pipeline
   (Bài 04): tokenizer của backbone + RenderMessagesStep (Bài 10) + normalize ảnh + device.
   Xem `policies/smolvla/processor_smolvla.py` làm mẫu.

## 2. Hai con đường

| | Plugin (out-of-tree) | In-tree |
|---|---|---|
| Cách | Package riêng `lerobot_policy_my_vqa` | Thêm vào `src/lerobot/policies/` |
| Cần PR? | Không | Có |
| Khi nào | **Thử nghiệm/nghiên cứu (chọn cái này trước)** | Khi đã ổn định, muốn đóng góp |

➡️ Cho research, **đi plugin** để lặp nhanh. `bring_your_own_policies.mdx` có hướng dẫn Path A
(plugin) và Path B (in-tree) chi tiết, gồm cách `lerobot-train`/`lerobot-eval` tự nhận policy
qua registry.

## 3. Nếu thêm vào in-tree

Ngoài 3 file trên, bạn sẽ thêm nhánh trong `policies/factory.py`:
- `get_policy_class("my_vqa")` → trả class.
- `make_policy_config("my_vqa")` → trả config.
Nhớ **lazy import** (Bài 05) để không bắt buộc cài backbone nặng cho mọi người.

## 4. Chiến lược backbone cho tiếng Việt
- Bắt đầu bằng một VLM **đa ngôn ngữ có sẵn** (Qwen-VL, Gemma-vision, ...) thay vì SmolVLM
  thuần Anh — đỡ phần khó nhất.
- Thí nghiệm nhỏ đáng làm trước: đo **tỉ lệ token/ký tự tiếng Việt** của tokenizer backbone
  (fertility). Tokenizer tốt → ít token rác → học nhanh hơn.
- Nếu backbone yếu tiếng Việt: cân nhắc tiếp tục pretrain backbone trên corpus tiếng Việt
  (Bài 15) trước khi finetune VQA.

## 5. Thực hành
```bash
# Đọc đầy đủ hai con đường + scaffolding
sed -n '1,200p' docs/source/bring_your_own_policies.mdx

# Học khuôn mẫu từ SmolVLA
ls src/lerobot/policies/smolvla
sed -n '1,60p' src/lerobot/policies/smolvla/processor_smolvla.py
```

## 6. Tự kiểm tra
1. Ba mảnh ghép bắt buộc của một policy là gì? Tên policy phải khớp ở mấy nơi?
2. Với VQA, `forward()` tính loss kiểu gì, khác VLA ra sao?
3. Vì sao nên đi đường plugin trước khi in-tree?

➡️ Tiếp theo: [14 — Finetune & PEFT/LoRA](./14-finetune-peft-lora.md)
