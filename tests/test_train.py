import json

import numpy as np
import pytest
import torch

from polyglotlm.config import StageConfig
from polyglotlm.demo.base import build_base_model
from polyglotlm.modeling import load_model
from polyglotlm.tokenizer import load_tokenizer
from polyglotlm.train.blocks import BlockSet, tokenize_corpus
from polyglotlm.train.cpt import MixedBatches, run_cpt
from polyglotlm.train.guard import OverfitGuard
from polyglotlm.train.loop import (
    MetricsWriter,
    Trainer,
    apply_lora,
    lora_lr_scale,
    trainable_groups,
)
from polyglotlm.train.mixing import MixtureSampler, mixture, temperature_weights
from tests.conftest import make_recipe

# --- mixing -----------------------------------------------------------------------------------


def test_temperature_weights_interpolate_between_proportional_and_uniform():
    sizes = {"hi": 900, "ta": 100}
    assert temperature_weights(sizes, 1.0) == pytest.approx({"hi": 0.9, "ta": 0.1})
    flat = temperature_weights(sizes, 1e-6)
    assert flat["hi"] == pytest.approx(0.5, abs=1e-3)
    upweighted = temperature_weights(sizes, 0.3)
    assert 0.1 < upweighted["ta"] < 0.5  # the low-resource language gets more than its share


def test_mixture_reserves_the_english_replay_share(tmp_path):
    recipe = make_recipe(tmp_path)
    probs = mixture(recipe, {"hi": 4000, "ta": 1000, "ur": 500, "bn": 2000, "en": 9999})
    assert probs["en"] == pytest.approx(0.15)
    assert sum(probs.values()) == pytest.approx(1.0)
    assert probs["hi"] > probs["bn"] > probs["ta"] > probs["ur"]


def test_mixture_without_english_data_renormalises_targets(tmp_path):
    probs = mixture(make_recipe(tmp_path), {"hi": 10, "ta": 10, "ur": 10, "bn": 10})
    assert "en" not in probs
    assert probs == pytest.approx({"hi": 0.25, "ta": 0.25, "ur": 0.25, "bn": 0.25})


def test_explicit_mix_ratios_override_temperature(tmp_path):
    recipe = make_recipe(
        tmp_path,
        languages=[
            {"code": "hi", "script": "Deva", "cpt_mix_ratio": 0.6},
            {"code": "ta", "script": "Taml", "cpt_mix_ratio": 0.25},
        ],
    )
    probs = mixture(recipe, {"hi": 1, "ta": 1_000_000, "en": 5})
    assert probs == pytest.approx({"hi": 0.6, "ta": 0.25, "en": 0.15})


def test_sampler_is_seeded_and_follows_the_mixture():
    probs = {"hi": 0.7, "en": 0.3}
    draws = MixtureSampler(probs, seed=3).sample(5000)
    assert draws == MixtureSampler(probs, seed=3).sample(5000)
    assert draws.count("hi") / len(draws) == pytest.approx(0.7, abs=0.03)
    with pytest.raises(ValueError):
        MixtureSampler({"hi": 0.5}, seed=0)


def test_lora_lr_schedule():
    assert [lora_lr_scale(s, 3, 2) for s in range(6)] == [0.0, 0.0, 0.0, 0.5, 1.0, 1.0]
    assert lora_lr_scale(0, 0, 0) == 1.0


# --- packed blocks ----------------------------------------------------------------------------


def test_tokenize_corpus_packs_with_eos_and_reuses_the_cache(built, tmp_path):
    tokenizer = load_tokenizer(built.base_model)
    corpus = tmp_path / "c.txt"
    corpus.write_text("hello world\n\nmore text\n", "utf-8")
    tokens = tokenize_corpus(tokenizer, corpus, tmp_path / "cache" / "c.bin")
    expected = [
        *tokenizer.encode("hello world", add_special_tokens=False),
        tokenizer.eos_token_id,
        *tokenizer.encode("more text", add_special_tokens=False),
        tokenizer.eos_token_id,
    ]
    assert list(tokens) == expected
    again = tokenize_corpus(tokenizer, corpus, tmp_path / "cache" / "c.bin")
    assert np.array_equal(tokens, again)


def test_block_set_shapes():
    blocks = BlockSet(np.arange(10, dtype=np.int32), block_size=4)
    assert len(blocks) == 2
    assert blocks.batch([1, 0]).tolist() == [[4, 5, 6, 7], [0, 1, 2, 3]]
    short = BlockSet(np.arange(3, dtype=np.int32), block_size=8)
    assert (len(short), short.block_size) == (1, 3)


def test_mixed_batches_are_reproducible():
    sets = {
        "hi": BlockSet(np.arange(100, dtype=np.int32), 5),
        "en": BlockSet(np.arange(100, 200, dtype=np.int32), 5),
    }
    first = MixedBatches(sets, {"hi": 0.5, "en": 0.5}, batch_size=6, seed=1)(0)
    second = MixedBatches(sets, {"hi": 0.5, "en": 0.5}, batch_size=6, seed=1)(0)
    assert torch.equal(first["input_ids"], second["input_ids"])
    assert first["input_ids"].shape == (6, 5)


# --- trainer ----------------------------------------------------------------------------------


def _tiny_trainer(tmp_path, embed_warmup_steps, max_steps, evaluate):
    model = apply_lora(build_base_model(vocab_size=50), rank=2, alpha=None, dropout=0.0)
    groups = trainable_groups(model, train_embeddings=True)
    stage = StageConfig(
        max_steps=max_steps,
        batch_size=2,
        block_size=8,
        lr=1e-2,
        embed_lr=1e-2,
        lr_warmup_steps=1,
        eval_every=2,
        patience=10,
    )
    writer = MetricsWriter(tmp_path / "m.jsonl", "test")
    trainer = Trainer(
        model,
        groups,
        stage,
        evaluate=evaluate,
        guard=OverfitGuard(["hi"], patience=10),
        writer=writer,
        embed_warmup_steps=embed_warmup_steps,
        device=torch.device("cpu"),
    )
    return groups, trainer, writer


def _batch(step):
    ids = torch.randint(0, 50, (2, 8), generator=torch.Generator().manual_seed(step))
    return {"input_ids": ids, "labels": ids}


def test_embedding_warmup_trains_only_embeddings(tmp_path):
    scores = iter([3.0, 2.0, 1.0])  # keep improving so the final weights are the ones kept
    groups, trainer, writer = _tiny_trainer(tmp_path, 4, 4, lambda: {"hi": next(scores)})
    lora_before = [p.detach().clone() for p in groups.lora]
    embed_before = groups.embed[0].detach().clone()
    trainer.fit(_batch)
    writer.close()
    assert all(torch.equal(a, b) for a, b in zip(lora_before, groups.lora, strict=True))
    assert not torch.equal(embed_before, groups.embed[0])
    with open(tmp_path / "m.jsonl", encoding="utf-8") as fh:
        phases = {json.loads(line).get("phase") for line in fh}
    assert phases - {None} == {"embed_warmup"}


def test_trainer_restores_the_best_checkpoint(tmp_path):
    scores = iter([5.0, 1.0, 9.0, 9.0, 9.0])  # best at the second evaluation (step 2)
    groups, trainer, writer = _tiny_trainer(tmp_path, 0, 8, lambda: {"hi": next(scores)})
    snapshots = {}
    original_eval = trainer._eval_step

    def spy(step):
        snapshots[step] = [p.detach().clone() for p in groups.all()]
        return original_eval(step)

    trainer._eval_step = spy
    result = trainer.fit(_batch)
    writer.close()
    assert result.best_step == 2 and result.heldout_ppl == {"hi": 1.0}
    assert all(torch.equal(a, b) for a, b in zip(snapshots[2], groups.all(), strict=True))


# --- end to end -------------------------------------------------------------------------------


def test_run_cpt_adapts_target_languages_and_evaluates_every_language(extended):
    out_dir = run_cpt(extended)
    with extended.paths.metrics("cpt").open(encoding="utf-8") as fh:
        records = [json.loads(line) for line in fh]
    evals = [r for r in records if r["kind"] == "eval"]
    assert all(set(r["heldout_ppl"]) == set(extended.all_codes) for r in evals)
    first = evals[0]["heldout_ppl"]
    summary = json.loads((out_dir / "cpt.json").read_text("utf-8"))
    for lang in extended.codes:
        assert summary["heldout_ppl"][lang] < first[lang]
    assert summary["mixture"]["en"] == pytest.approx(extended.english_replay)
    assert {r["phase"] for r in records if r["kind"] == "train"} == {"embed_warmup", "full"}

    merged = load_model(out_dir)
    tokenizer = load_tokenizer(out_dir)
    assert merged.get_input_embeddings().weight.shape[0] == len(tokenizer)
    assert not any("lora" in name for name, _ in merged.named_parameters())
    assert (out_dir / "adapter" / "adapter_config.json").exists()
