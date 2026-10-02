"""Deterministic scorers: log-likelihood choice ranking, greedy generation, token F1."""

from __future__ import annotations

import re
import unicodedata
from collections import Counter
from typing import Any

import torch

from polyglotlm.eval.tasks import MCItem, QAItem
from polyglotlm.prompts import format_prompt


def _context_ids(tokenizer: Any, prompt: str) -> list[int]:
    """Prompt tokens, or a lone EOS when the prompt is empty so every choice token is scored."""
    ids = tokenizer.encode(prompt, add_special_tokens=False) if prompt else []
    return ids or [tokenizer.eos_token_id]


@torch.no_grad()
def continuation_logprob(model: Any, tokenizer: Any, prompt: str, continuation: str) -> float:
    """Sum of log-probabilities of `continuation` tokens given `prompt`."""
    context = _context_ids(tokenizer, prompt)
    target = tokenizer.encode(continuation, add_special_tokens=False)
    ids = torch.tensor([context + target], device=model.device)
    logits = model(input_ids=ids).logits[0, len(context) - 1 : -1].float()
    logprobs = torch.log_softmax(logits, dim=-1)
    return logprobs.gather(1, torch.tensor(target, device=model.device)[:, None]).sum().item()


def score_mc(model: Any, tokenizer: Any, item: MCItem) -> bool:
    scores = [continuation_logprob(model, tokenizer, item.prompt, c) for c in item.choices]
    return max(range(len(scores)), key=scores.__getitem__) == item.answer


def greedy_answer(model: Any, tokenizer: Any, prompt: str, max_new_tokens: int = 64) -> str:
    """Greedy reply to an instruction, formatted with the same template as SFT."""
    return greedy_continuation(model, tokenizer, format_prompt(tokenizer, prompt), max_new_tokens)


@torch.no_grad()
def greedy_continuation(model: Any, tokenizer: Any, text: str, max_new_tokens: int = 64) -> str:
    """Deterministic raw continuation of `text` (no template)."""
    ids = tokenizer.encode(text, add_special_tokens=False) or [tokenizer.eos_token_id]
    input_ids = torch.tensor([ids], device=model.device)
    output = model.generate(
        input_ids=input_ids,
        attention_mask=torch.ones_like(input_ids),
        max_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    return tokenizer.decode(output[0, len(ids) :], skip_special_tokens=True).strip()


_PUNCT = re.compile(r"[\W_]+", re.UNICODE)


def normalise_answer(text: str) -> list[str]:
    """NFC, casefold, punctuation to spaces, whitespace tokens (works for every script)."""
    folded = unicodedata.normalize("NFC", text).casefold()
    return _PUNCT.sub(" ", folded).split()


def token_f1(prediction: str, reference: str) -> float:
    pred, ref = normalise_answer(prediction), normalise_answer(reference)
    common = sum((Counter(pred) & Counter(ref)).values())
    if not pred or not ref or not common:
        return float(pred == ref)
    precision, recall = common / len(pred), common / len(ref)
    return 2 * precision * recall / (precision + recall)


def score_qa(model: Any, tokenizer: Any, item: QAItem) -> tuple[float, float]:
    """(best F1, exact match) of the greedy answer against any reference."""
    prediction = greedy_answer(model, tokenizer, item.prompt)
    f1 = max(token_f1(prediction, ref) for ref in item.answers)
    exact = max(float(normalise_answer(prediction) == normalise_answer(r)) for r in item.answers)
    return f1, exact
