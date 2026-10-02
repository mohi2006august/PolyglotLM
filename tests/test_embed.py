import json

import pytest
import torch

from polyglotlm.demo.corpus import parallel_sentences
from polyglotlm.embed.distill import (
    ParallelPair,
    pairs_from_dir,
    pairs_from_sft,
    retrieval_scores,
    run_embed_distill,
    split_pairs,
)
from polyglotlm.embed.encoder import LMEncoder, mean_pool
from polyglotlm.embed.search import EmbeddingIndex


class _LookupEncoder:
    """Embeds by lookup: translations of one sentence share a vector."""

    dim = 3

    def __init__(self, table):
        self.table = table

    def encode(self, texts):
        return torch.nn.functional.normalize(
            torch.tensor([self.table[t] for t in texts], dtype=torch.float), dim=-1
        )


def test_mean_pool_ignores_padding():
    hidden = torch.tensor([[[1.0, 1.0], [3.0, 3.0], [100.0, 100.0]]])
    mask = torch.tensor([[1, 1, 0]])
    assert mean_pool(hidden, mask).tolist() == [[2.0, 2.0]]


def test_retrieval_scores_rank_the_true_translation():
    table = {
        "a": [1, 0, 0],
        "b": [0, 1, 0],
        "c": [0, 0, 1],
        "A": [1, 0.1, 0],
        "B": [0, 1, 0.1],
        "C": [1, 0, 0.05],
    }
    pairs = [ParallelPair("hi", e, t) for e, t in (("a", "A"), ("b", "B"), ("c", "C"))]
    scores = retrieval_scores(_LookupEncoder(table), pairs)["hi"]
    assert scores["acc_at_1"] == pytest.approx(200 / 3)  # "C" lands nearest to "a"
    assert scores["mrr"] == pytest.approx((1 + 1 + 0.5) / 3)


def test_split_keeps_each_english_sentence_on_one_side():
    pairs = [
        ParallelPair(lang, row["en"], row[lang])
        for row in parallel_sentences()
        for lang in ("hi", "ta")
    ]
    train, heldout = split_pairs(pairs, 0.2, seed=0)
    assert train and heldout
    assert not {p.english for p in train} & {p.english for p in heldout}


def test_pairs_come_from_sft_translations_and_tsv_files(tmp_path):
    sft = tmp_path / "train.jsonl"
    rows = [
        {
            "lang": "hi",
            "origin": "translated",
            "instruction": "प्र",
            "response": "उ",
            "source": {"instruction": "q", "response": "a"},
        },
        {"lang": "hi", "origin": "native", "instruction": "x", "response": "y"},
    ]
    sft.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), "utf-8")
    assert pairs_from_sft(sft) == [ParallelPair("hi", "q", "प्र"), ParallelPair("hi", "a", "उ")]
    (tmp_path / "ta.tsv").write_text("hello\tவணக்கம்\nno tab line\n", "utf-8")
    assert pairs_from_dir(str(tmp_path), ["ta", "bn"]) == [ParallelPair("ta", "hello", "வணக்கம்")]


@pytest.fixture(scope="module")
def distilled(trained, tmp_path_factory):
    parallel = tmp_path_factory.mktemp("parallel")
    rows = parallel_sentences()
    for lang in trained.codes:
        (parallel / f"{lang}.tsv").write_text(
            "".join(f"{row['en']}\t{row[lang]}\n" for row in rows), "utf-8"
        )
    recipe = trained.model_copy(
        update={
            "embed": trained.embed.model_copy(
                update={
                    "parallel_dir": str(parallel),
                    "max_steps": 60,
                    "batch_size": 16,
                    "lr": 2e-3,
                    "max_length": 48,
                    "heldout_fraction": 0.15,
                    "eval_every": 20,
                }
            )
        }
    )
    return recipe, run_embed_distill(recipe)


def test_distillation_improves_cross_lingual_retrieval(distilled):
    _, out_dir = distilled
    summary = json.loads((out_dir / "embed.json").read_text("utf-8"))
    before = sum(v["mrr"] for v in summary["before"].values()) / len(summary["before"])
    after = sum(v["mrr"] for v in summary["after"].values()) / len(summary["after"])
    assert after > before
    assert summary["heldout_pairs"] > 0 and summary["train_pairs"] > summary["heldout_pairs"]


def test_saved_encoder_reloads_and_searches_across_languages(distilled):
    _, out_dir = distilled
    encoder = LMEncoder.load(out_dir)
    vectors = encoder.encode(["आज बाज़ार में बहुत भीड़ है।", "hello"])
    assert vectors.shape == (2, encoder.dim)
    assert torch.allclose(vectors.norm(dim=-1), torch.ones(2), atol=1e-5)
    rows = parallel_sentences()[:30]
    index = EmbeddingIndex(encoder, [row["en"] for row in rows])
    hits = index.search(rows[0]["hi"], k=3)
    assert len(hits) == 3 and hits[0][0] >= hits[-1][0]
