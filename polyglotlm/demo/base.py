"""A tiny Llama-shaped "base model" for CPU smoke runs and tests.

The tokenizer is a byte-level BPE trained mostly on English with a sliver of target-language
text, which mimics how open LLM tokenizers over-segment low-resource scripts. The model is
briefly pre-trained on English so that "does English survive adaptation?" has a real answer.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from polyglotlm.logs import get_logger
from polyglotlm.modeling import save_model, set_seed

log = get_logger(__name__)

BOS = "<|begin_of_text|>"
EOS = "<|end_of_text|>"


def build_base_tokenizer(texts: list[str], vocab_size: int = 512) -> Any:
    """Byte-level BPE (never produces unknown tokens), wrapped as a fast HF tokenizer."""
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
    from transformers import PreTrainedTokenizerFast

    tok = Tokenizer(models.BPE())
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=[BOS, EOS],
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        show_progress=False,
    )
    tok.train_from_iterator(texts, trainer)
    return PreTrainedTokenizerFast(
        tokenizer_object=tok,
        bos_token=BOS,
        eos_token=EOS,
        pad_token=EOS,
        clean_up_tokenization_spaces=False,
    )


def build_base_model(vocab_size: int, seed: int = 0, tie_embeddings: bool = False) -> Any:
    from transformers import LlamaConfig, LlamaForCausalLM

    set_seed(seed)
    config = LlamaConfig(
        vocab_size=vocab_size,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=4,
        max_position_embeddings=256,
        bos_token_id=0,
        eos_token_id=1,
        pad_token_id=1,
        tie_word_embeddings=tie_embeddings,
    )
    return LlamaForCausalLM(config)


def pretrain(
    model: Any,
    tokenizer: Any,
    docs: list[str],
    steps: int,
    block_size: int = 64,
    batch_size: int = 16,
    lr: float = 3e-3,
    seed: int = 0,
) -> float:
    """Full-parameter next-token training on packed `docs`. Returns the final loss."""
    ids: list[int] = []
    for doc in docs:
        ids.extend([*tokenizer.encode(doc, add_special_tokens=False), tokenizer.eos_token_id])
    n_blocks = len(ids) // block_size
    blocks = torch.tensor(ids[: n_blocks * block_size]).view(n_blocks, block_size)
    generator = torch.Generator().manual_seed(seed)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    model.train()
    loss = torch.tensor(float("nan"))
    for _ in range(steps):
        batch = blocks[torch.randint(n_blocks, (batch_size,), generator=generator)]
        loss = model(input_ids=batch, labels=batch).loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
    model.eval()
    return loss.item()


def create_base(
    out_dir: Path,
    english_docs: list[str],
    target_docs: list[str],
    pretrain_steps: int = 150,
    seed: int = 0,
    tie_embeddings: bool = False,
) -> Path:
    """Train the tokenizer, build and pre-train the model, save both to `out_dir`."""
    tokenizer = build_base_tokenizer(english_docs + target_docs)
    model = build_base_model(len(tokenizer), seed, tie_embeddings)
    loss = pretrain(model, tokenizer, english_docs, pretrain_steps, seed=seed)
    save_model(model, tokenizer, out_dir)
    log.info("demo_base_created", path=str(out_dir), vocab=len(tokenizer), english_loss=loss)
    return out_dir
