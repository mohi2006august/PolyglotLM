import json
import re
import xml.etree.ElementTree as ET

import pytest
import torch

from polyglotlm.demo.base import build_base_model
from polyglotlm.demo.corpus import demo_eval_items
from polyglotlm.eval.charts import log_ticks, nice_ticks
from polyglotlm.eval.english_regression import compare
from polyglotlm.eval.harness import checkpoints, english_monitor, evaluate_checkpoint
from polyglotlm.eval.report import collect, render_html, render_model_card, write_report
from polyglotlm.eval.tasks import MCItem, load_items
from polyglotlm.eval.tasks.scoring import (
    REPLACEMENT_CHAR,
    drop_partial_char,
    normalise_answer,
    token_f1,
)
from polyglotlm.tokenizer import load_tokenizer
from polyglotlm.tokenizer.fertility import fertility_report, save_report


@pytest.fixture(scope="module")
def evaluated(trained, tmp_path_factory):
    """Demo task files plus eval results for the base model and the CPT model."""
    tasks = tmp_path_factory.mktemp("tasks")
    for lang in trained.all_codes:
        rows = demo_eval_items(lang, 8, seed=0)
        (tasks / f"{lang}.jsonl").write_text(
            "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), "utf-8"
        )
    recipe = trained.model_copy(update={"eval_tasks_dir": str(tasks)})
    for name, tokenizer in (
        ("base", recipe.base_model),
        ("extended", recipe.paths.extended_tokenizer),
    ):
        report = fertility_report(load_tokenizer(tokenizer), recipe, recipe.all_codes)
        save_report(report, recipe.paths.fertility(name))
    for name, path in checkpoints(recipe).items():
        if name in ("base", "cpt"):
            evaluate_checkpoint(recipe, name, path, max_docs=20)
    return recipe


# --- tasks and metrics ------------------------------------------------------------------------


def test_items_validate_and_load(tmp_path):
    path = tmp_path / "hi.jsonl"
    path.write_text(
        '{"type": "mc", "id": "a", "choices": ["x", "y"], "answer": 1}\n'
        '{"type": "qa", "id": "b", "prompt": "q", "answers": ["r"]}\n',
        "utf-8",
    )
    items = load_items(path)
    assert [type(i).__name__ for i in items] == ["MCItem", "QAItem"]
    with pytest.raises(ValueError, match="out of range"):
        MCItem(id="c", choices=["x", "y"], answer=2)


def test_demo_items_have_one_grammatical_choice():
    for item in demo_eval_items("ta", 5, seed=1):
        choices = item["choices"]
        assert len(set(choices)) == 4
        assert sorted(choices[item["answer"]].split()) == sorted(choices[0].split())


def test_token_f1_works_across_scripts():
    assert token_f1("चावल का दाम", "चावल का दाम।") == 1.0
    assert token_f1("The Market!", "the market") == 1.0
    assert 0 < token_f1("அரிசி விலை உயர்ந்தது", "அரிசி விலை") < 1
    assert token_f1("", "x") == 0.0
    assert normalise_answer("Hello, WORLD") == ["hello", "world"]


def test_truncated_generation_drops_only_the_trailing_partial_character():
    assert drop_partial_char("ہے" + REPLACEMENT_CHAR) == "ہے"
    inner = "a" + REPLACEMENT_CHAR + "b"
    assert drop_partial_char(inner) == inner


def test_ticks():
    assert nice_ticks(0, 9.3) == [0, 2, 4, 6, 8, 10]
    assert log_ticks(3, 600) == [5, 10, 20, 50, 100, 200, 500]


# --- comparison -------------------------------------------------------------------------------


def _result(name, en_score, en_bpc, hi_score):
    return {
        "name": name,
        "languages": {
            "en": {"score": en_score, "bits_per_char": en_bpc},
            "hi": {"score": hi_score, "bits_per_char": 3.0},
        },
    }


def test_compare_flags_english_regression_in_points():
    ok = compare(_result("base", 60, 1.0, 25), _result("sft", 58, 1.0, 70), 3.0)
    assert ok["english"] == {"metric": "points", "drop": 2, "regressed": False}
    assert ok["languages"]["hi"]["score_delta"] == 45
    bad = compare(_result("base", 60, 1.0, 25), _result("sft", 55, 1.0, 70), 3.0)
    assert bad["english"]["regressed"]


def test_compare_falls_back_to_bits_per_char_without_tasks():
    verdict = compare(_result("base", None, 1.0, None), _result("cpt", None, 1.05, None), 3.0)
    assert verdict["english"]["metric"] == "bpc_change_pct"
    assert verdict["english"]["regressed"]  # +5% bits per char > 3% budget


# --- harness ----------------------------------------------------------------------------------


def test_evaluation_covers_every_language_and_is_deterministic(evaluated):
    first = json.loads(evaluated.paths.eval("cpt").read_text("utf-8"))
    assert set(first["languages"]) == set(evaluated.all_codes)
    for scores in first["languages"].values():
        assert scores["bits_per_char"] > 0 and scores["n_mc"] == 8
        assert 0 <= scores["score"] <= 100
        assert scores["sample"]["prompt"]
    again = evaluate_checkpoint(evaluated, "cpt", str(evaluated.paths.model("cpt")), max_docs=20)
    assert again["languages"] == first["languages"]


def test_bits_per_char_is_comparable_across_tokenizers(evaluated):
    base = json.loads(evaluated.paths.eval("base").read_text("utf-8"))
    cpt = json.loads(evaluated.paths.eval("cpt").read_text("utf-8"))
    hi_base, hi_cpt = base["languages"]["hi"], cpt["languages"]["hi"]
    assert hi_cpt["tokens_per_char"] < hi_base["tokens_per_char"]  # extended tokenizer
    assert hi_cpt["bits_per_char"] < hi_base["bits_per_char"]  # and the model learned Hindi


def test_english_monitor_scores_and_restores_training_mode(evaluated, tmp_path):
    tokenizer = load_tokenizer(evaluated.base_model)
    model = build_base_model(len(tokenizer))
    model.train()
    task = tmp_path / "en.jsonl"
    task.write_text(
        "".join(json.dumps(r) + "\n" for r in demo_eval_items("en", 6, seed=0)), "utf-8"
    )
    score = english_monitor(task, model, tokenizer)
    with torch.no_grad():
        value = score()
    assert 0 <= value <= 100 and model.training


# --- report -----------------------------------------------------------------------------------


def test_report_card_renders_every_section_accessibly(evaluated):
    path = write_report(evaluated)
    html = path.read_text("utf-8")
    for heading in ("Token tax", "Benchmarks", "Training monitor", "Known limitations"):
        assert heading in html
    assert 'class="chip' in html and 'role="status"' in html
    assert html.count("<details>") >= 2  # every chart has a data table
    charts = re.findall(r'<svg [^>]*class="chart".*?</svg>', html, flags=re.S)
    assert len(charts) >= 2
    for svg in charts:
        root = ET.fromstring(svg)  # well-formed
        assert root.get("role") == "img" and root.get("aria-describedby")
    for icon in re.findall(r'<svg viewBox="0 0 16 16".*?</svg>', html, flags=re.S):
        assert 'aria-hidden="true"' in icon
    assert "prefers-color-scheme: dark" in html and 'data-theme="dark"' in html
    card = (path.parent / "MODEL_CARD.md").read_text("utf-8")
    assert "| Language | Base | Extended | Saving |" in card
    assert "English:" in card and "## Known limitations" in card


def test_report_handles_a_workdir_with_nothing_evaluated(tmp_path):
    from tests.conftest import make_recipe

    data = collect(make_recipe(tmp_path))
    html = render_html(data)
    assert "poly fertility" in html and "not measured yet" in html
    assert "_Not evaluated yet._" in render_model_card(data)
