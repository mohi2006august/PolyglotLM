"""Initialise the rows of new tokens (FR-02).

* FVT (Fast Vocabulary Transfer): tokenize the new token's string with the base tokenizer into
  s1..sm and set `E[t] = mean(E[s1..sm])`, for the input embeddings and the LM head alike. The
  model then treats a new token roughly like the sub-token sequence it replaces from step 0.
* Dictionary alignment: when a bilingual dictionary has an English gloss for the token, blend
  the FVT vector with the mean embedding of the gloss, pulling the token towards the English
  word it translates. An untied LM head keeps plain FVT; a tied one shares the blended row.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from polyglotlm.logs import get_logger
from polyglotlm.modeling import embeddings_tied
from polyglotlm.text import nfc

log = get_logger(__name__)


@dataclass(frozen=True)
class InitReport:
    new_tokens: int
    dictionary_hits: int
    tied: bool


def load_dictionary(path: Path) -> dict[str, str]:
    """Read a `target<TAB>english` file into an NFC-normalised lookup."""
    entries = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if "\t" in line:
            target, english = line.split("\t", 1)
            entries[nfc(target).strip()] = english.strip()
    return entries


def _sub_token_ids(base_tokenizer: Any, text: str) -> list[int]:
    ids = base_tokenizer.encode(text, add_special_tokens=False)
    if not ids:
        raise ValueError(f"base tokenizer produced no tokens for new token {text!r}")
    return ids


@torch.no_grad()
def mean_rows(weight: torch.Tensor, ids: list[int]) -> torch.Tensor:
    return weight[ids].float().mean(dim=0).to(weight.dtype)


@torch.no_grad()
def init_new_embeddings(
    model: Any,
    base_tokenizer: Any,
    extended_tokenizer: Any,
    dictionary: dict[str, str] | None = None,
    dictionary_weight: float = 0.5,
) -> InitReport:
    """Fill rows `len(base)..len(extended)` of the embeddings and LM head in place."""
    old_size = len(base_tokenizer)
    embed = model.get_input_embeddings().weight
    tied = embeddings_tied(model)
    head = None if tied else model.get_output_embeddings()
    hits = 0
    for token_id in range(old_size, len(extended_tokenizer)):
        text = extended_tokenizer.convert_ids_to_tokens(token_id)
        sub_ids = _sub_token_ids(base_tokenizer, text)
        vector = mean_rows(embed, sub_ids)
        if head is not None:
            head.weight[token_id] = mean_rows(head.weight, sub_ids)
            if head.bias is not None:
                head.bias[token_id] = head.bias[sub_ids].mean()
        if dictionary and (gloss := dictionary.get(text.strip())):
            prefix = " " if text.startswith(" ") else ""
            gloss_vector = mean_rows(embed, _sub_token_ids(base_tokenizer, prefix + gloss))
            vector = (1 - dictionary_weight) * vector + dictionary_weight * gloss_vector
            hits += 1
        embed[token_id] = vector
    report = InitReport(len(extended_tokenizer) - old_size, hits, tied)
    log.info("embeddings_initialised", **report.__dict__)
    return report
