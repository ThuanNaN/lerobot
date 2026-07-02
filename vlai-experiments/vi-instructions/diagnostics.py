from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

# Representative LIBERO-style imperative instructions, used only for this cheap
# sanity check — NOT the reviewed Stage 1 translations (those come from Task 2/3).
EN_VI_PROBE_PAIRS: list[tuple[str, str]] = [
    ("pick up the black bowl and place it in the tray", "nhấc cái bát đen lên và đặt nó vào khay"),
    ("open the top drawer and put the bowl inside", "mở ngăn kéo trên cùng và đặt cái bát vào bên trong"),
    ("turn on the stove", "bật bếp lên"),
    ("put the red mug on the plate", "đặt cốc màu đỏ lên đĩa"),
    ("close the microwave door", "đóng cửa lò vi sóng lại"),
    ("push the plate to the front of the stove", "đẩy cái đĩa ra phía trước bếp"),
    ("pick up the alphabet soup and put it in the basket", "nhấc hộp súp chữ cái lên và đặt vào giỏ"),
    ("put the wine bottle on the rack", "đặt chai rượu vang lên giá"),
]


def tokens_per_word(tokenizer, sentence: str) -> float:
    n_words = len(sentence.split())
    if n_words == 0:
        raise ValueError("sentence must contain at least one word")
    n_tokens = len(tokenizer(sentence)["input_ids"])
    return n_tokens / n_words


def mean_fertility(tokenizer, sentences: list[str]) -> float:
    return sum(tokens_per_word(tokenizer, s) for s in sentences) / len(sentences)


def pooled_embedding(text_model, tokenizer, sentence: str) -> torch.Tensor:
    encoded = tokenizer(sentence, return_tensors="pt")
    with torch.no_grad():
        out = text_model(input_ids=encoded["input_ids"], attention_mask=encoded["attention_mask"])
    mask = encoded["attention_mask"].unsqueeze(-1).to(out.last_hidden_state.dtype)
    summed = (out.last_hidden_state * mask).sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1)
    return (summed / counts).squeeze(0)


def paraphrase_gap(text_model, tokenizer, pairs: list[tuple[str, str]]) -> dict:
    embeddings_en = [pooled_embedding(text_model, tokenizer, en) for en, _ in pairs]
    embeddings_vi = [pooled_embedding(text_model, tokenizer, vi) for _, vi in pairs]

    matched_sims = [
        torch.nn.functional.cosine_similarity(embeddings_en[i], embeddings_vi[i], dim=0).item()
        for i in range(len(pairs))
    ]
    mismatched_sims = [
        torch.nn.functional.cosine_similarity(embeddings_en[i], embeddings_vi[j], dim=0).item()
        for i in range(len(pairs))
        for j in range(len(pairs))
        if i != j
    ]
    matched_mean = sum(matched_sims) / len(matched_sims)
    mismatched_mean = sum(mismatched_sims) / len(mismatched_sims)
    return {
        "matched_mean": matched_mean,
        "mismatched_mean": mismatched_mean,
        "gap": matched_mean - mismatched_mean,
    }


def main(model_id: str, output_path: Path) -> None:
    from transformers import AutoModelForImageTextToText, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForImageTextToText.from_pretrained(model_id, torch_dtype="bfloat16")
    text_model = model.model.text_model

    en_sentences = [en for en, _ in EN_VI_PROBE_PAIRS]
    vi_sentences = [vi for _, vi in EN_VI_PROBE_PAIRS]

    report = {
        "model_id": model_id,
        "fertility_en": mean_fertility(tokenizer, en_sentences),
        "fertility_vi": mean_fertility(tokenizer, vi_sentences),
        "paraphrase_gap": paraphrase_gap(text_model, tokenizer, EN_VI_PROBE_PAIRS),
    }
    report["fertility_ratio_vi_over_en"] = report["fertility_vi"] / report["fertility_en"]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", default="HuggingFaceTB/SmolVLM2-500M-Video-Instruct")
    parser.add_argument(
        "--output", type=Path, default=Path("vlai-experiments/vi-instructions/data/stage0_diagnostics.json")
    )
    args = parser.parse_args()
    main(args.model_id, args.output)
