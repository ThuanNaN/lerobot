import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest
import torch

from diagnostics import mean_fertility, paraphrase_gap, pooled_embedding, tokens_per_word


class _StubTokenizer:
    def __call__(self, sentence: str, return_tensors: str | None = None):
        token_ids = [ord(c) % 50 + 1 for c in sentence.replace(" ", "")]
        if return_tensors == "pt":
            input_ids = torch.tensor([token_ids])
            attention_mask = torch.ones_like(input_ids)
            return {"input_ids": input_ids, "attention_mask": attention_mask}
        return {"input_ids": token_ids}


class _StubOutput:
    def __init__(self, last_hidden_state: torch.Tensor):
        self.last_hidden_state = last_hidden_state


class _StubTextModel:
    hidden_size = 4

    def __call__(self, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        torch.manual_seed(int(input_ids.sum().item()))
        hidden = torch.randn(1, input_ids.shape[1], self.hidden_size)
        return _StubOutput(hidden)


def test_tokens_per_word_counts_ratio():
    tokenizer = _StubTokenizer()
    ratio = tokens_per_word(tokenizer, "hello world")
    assert ratio == len(tokenizer("hello world")["input_ids"]) / 2


def test_tokens_per_word_rejects_empty_sentence():
    with pytest.raises(ValueError):
        tokens_per_word(_StubTokenizer(), "")


def test_mean_fertility_averages_across_sentences():
    tokenizer = _StubTokenizer()
    sentences = ["hello world", "pick up the bowl"]
    expected = sum(tokens_per_word(tokenizer, s) for s in sentences) / len(sentences)
    assert mean_fertility(tokenizer, sentences) == expected


def test_pooled_embedding_has_hidden_size_shape():
    vec = pooled_embedding(_StubTextModel(), _StubTokenizer(), "hello")
    assert vec.shape == (_StubTextModel.hidden_size,)


def test_paraphrase_gap_returns_matched_and_mismatched_means():
    pairs = [("hello", "xin chao"), ("world", "the gioi")]
    result = paraphrase_gap(_StubTextModel(), _StubTokenizer(), pairs)
    assert set(result.keys()) == {"matched_mean", "mismatched_mean", "gap"}
    assert -1.0 <= result["matched_mean"] <= 1.0
    assert -1.0 <= result["mismatched_mean"] <= 1.0
