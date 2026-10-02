"""Prompt formatting shared by SFT and generative evaluation, so both see the same template."""

from __future__ import annotations

from typing import Any

PLAIN_TEMPLATE = "### Instruction:\n{instruction}\n\n### Response:\n"


def format_prompt(tokenizer: Any, instruction: str) -> str:
    """The model's chat template when it has one, otherwise a plain instruction template."""
    if getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": instruction}],
            tokenize=False,
            add_generation_prompt=True,
        )
    return PLAIN_TEMPLATE.format(instruction=instruction)
