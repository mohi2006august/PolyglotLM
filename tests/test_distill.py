import json
from types import SimpleNamespace

import pytest

from polyglotlm.config import DistillConfig
from polyglotlm.data.download import LicenceError
from polyglotlm.demo.corpus import all_sentences, demo_seed_pairs
from polyglotlm.distill.back_translate import length_problem, script_problem
from polyglotlm.distill.chrf import chrf
from polyglotlm.distill.review import apply_reviews
from polyglotlm.distill.teacher import ClaudeTeacher, DemoTeacher, Pair, TeacherError
from polyglotlm.distill.teacher_generate import distill_sft, load_examples, plan_tasks
from tests.conftest import make_recipe


def _recipe_with_seeds(tmp_path, n=40, **distill):
    seeds = tmp_path / "seeds.jsonl"
    seeds.write_text(
        "".join(json.dumps(p, ensure_ascii=False) + "\n" for p in demo_seed_pairs(30, seed=0)),
        "utf-8",
    )
    config = {
        "teacher": "demo",
        "seed_path": str(seeds),
        "seed_licence": "CC0-1.0",
        "n": n,
        "workers": 2,
        **distill,
    }
    return make_recipe(tmp_path, distill=config)


# --- metrics and checks -----------------------------------------------------------------------


def test_chrf_bounds_and_partial_credit():
    assert chrf("the price of rice", "the price of rice") == pytest.approx(100.0)
    assert chrf("xyz", "abc") == 0.0
    partial = chrf("the price of milk went up", "the price of rice has gone up")
    assert 30 < partial < 90
    assert chrf("a b c", "abc") == pytest.approx(100.0)  # whitespace is ignored


def test_script_and_length_checks():
    hindi = Pair(instruction="आज बाज़ार में भीड़ है।", response="किसान कल शहर जाएँगे।")
    assert script_problem(hindi, "Deva", 0.6) is None
    assert script_problem(Pair(instruction="hello", response="world"), "Deva", 0.6) == (
        "wrong_script"
    )
    assert script_problem(Pair(instruction="", response="कुछ"), "Deva", 0.6) == "empty_field"
    code = Pair(instruction="यह कोड चलाइए", response="```python\nprint('hello world')\n```")
    assert script_problem(code, "Deva", 0.6) is None  # code blocks are not judged on script
    source = Pair(instruction="a" * 40, response="b" * 40)
    assert length_problem(Pair(instruction="x", response="y"), source) == "length_ratio"
    assert length_problem(Pair(instruction="x" * 30, response="y" * 30), source) is None


def test_plan_splits_budget_between_translation_and_native():
    seeds = [Pair(instruction=f"i{i}", response="r") for i in range(3)]
    cfg = DistillConfig(n=40, native_fraction=0.5, pairs_per_call=3)
    tasks = plan_tasks(["hi", "ta"], seeds, cfg, seed=0)
    for lang in ("hi", "ta"):
        translate = [t for t in tasks if t.lang == lang and t.kind == "translate"]
        native = [t for t in tasks if t.lang == lang and t.kind == "native"]
        assert len(translate) == 10
        assert sum(t.n for t in native) == 10 and max(t.n for t in native) <= 3
    assert plan_tasks(["hi"], seeds, cfg, seed=0) == plan_tasks(["hi"], seeds, cfg, seed=0)


# --- teachers ---------------------------------------------------------------------------------


def test_demo_teacher_round_trips_through_back_translation():
    teacher = DemoTeacher()
    source = Pair(instruction=all_sentences("en")[0], response=all_sentences("en")[5])
    hindi = teacher.translate(source, "hi")
    assert hindi.instruction == all_sentences("hi")[0]
    assert teacher.back_translate(hindi, "hi") == source
    with pytest.raises(TeacherError):
        teacher.translate(Pair(instruction="unknown", response="x"), "hi")


class _FakeMessages:
    def __init__(self, response):
        self.response = response
        self.calls = []

    def parse(self, **kwargs):
        self.calls.append(kwargs)
        return self.response


def _claude_with(monkeypatch, response):
    monkeypatch.setenv("TEACHER_API_KEY", "test-key-not-real")
    teacher = ClaudeTeacher("claude-opus-5-5", effort="high")
    messages = _FakeMessages(response)
    teacher._client = SimpleNamespace(beta=SimpleNamespace(messages=messages))
    return teacher, messages


def test_claude_teacher_requests_structured_output_with_fallbacks(monkeypatch):
    translated = Pair(instruction="नमस्ते", response="धन्यवाद")
    ok = SimpleNamespace(stop_reason="end_turn", stop_details=None, parsed_output=translated)
    teacher, messages = _claude_with(monkeypatch, ok)
    assert teacher.translate(Pair(instruction="hello", response="thanks"), "hi") == translated
    call = messages.calls[0]
    assert call["model"] == "claude-opus-5-5"
    assert call["output_format"] is Pair
    assert call["output_config"] == {"effort": "high"}
    assert call["fallbacks"] == "default"
    assert call["betas"] == ["server-side-fallback-2026-07-01"]
    assert "Hindi" in call["system"] and "hello" in call["messages"][0]["content"]
    assert "temperature" not in call


@pytest.mark.parametrize("stop_reason", ["refusal", "max_tokens"])
def test_claude_teacher_turns_refusals_and_truncation_into_errors(monkeypatch, stop_reason):
    bad = SimpleNamespace(
        stop_reason=stop_reason, stop_details=SimpleNamespace(category="bio"), parsed_output=None
    )
    teacher, _ = _claude_with(monkeypatch, bad)
    with pytest.raises(TeacherError):
        teacher.back_translate(Pair(instruction="नमस्ते", response="धन्यवाद"), "hi")


# --- pipeline ---------------------------------------------------------------------------------


def test_distill_with_demo_teacher_writes_qa_passed_examples(tmp_path):
    recipe = _recipe_with_seeds(tmp_path, n=40, review_fraction=0.25)
    report = distill_sft(recipe, DemoTeacher())
    examples = load_examples(recipe.paths.sft_data)
    by_origin = {
        o: [e for e in examples if e.origin == o] for o in ("seed", "translated", "native")
    }
    unique_seeds = {p["instruction"] for p in demo_seed_pairs(30, seed=0)}
    assert len(by_origin["seed"]) == len(unique_seeds)
    assert {e.lang for e in by_origin["seed"]} == {"en"}
    assert by_origin["translated"] and by_origin["native"]
    assert all(e.chrf == pytest.approx(100.0) for e in by_origin["translated"])
    assert all(e.back_translation is not None for e in by_origin["native"])
    assert report.accepted["en"] == len(unique_seeds)
    assert sum(report.accepted[lang] for lang in recipe.codes) == len(examples) - len(unique_seeds)
    assert len({e.id for e in examples}) == len(examples)
    queued = sum(len(load_examples(recipe.paths.review_queue(lang))) for lang in recipe.codes)
    assert queued == report.queued_for_review > 0
    assert sum(e.review == "pending" for e in examples) == queued


class _UnreliableTeacher(DemoTeacher):
    """Fails in each way the QA must catch, keyed on which demo frame the seed comes from."""

    def translate(self, pair, lang):
        if pair.instruction.startswith("Eating"):
            return Pair(instruction="untranslated english", response="still english")
        if pair.instruction.startswith("The price"):
            raise TeacherError("simulated refusal")
        return super().translate(pair, lang)

    def back_translate(self, pair, lang):
        back = super().back_translate(pair, lang)
        if back.instruction.startswith("I enjoy"):
            return Pair(instruction="zzzz qqqq", response="xxxx")
        return back


def test_distill_rejects_bad_teacher_outputs_with_reasons(tmp_path):
    recipe = _recipe_with_seeds(tmp_path, n=120, native_fraction=0.0)
    report = distill_sft(recipe, _UnreliableTeacher())
    rejected = load_examples(recipe.paths.sft_rejected)
    reasons = {e.reason for e in rejected}
    assert {"wrong_script", "back_translation", "duplicate"} <= reasons
    assert report.teacher_errors > 0
    kept = load_examples(recipe.paths.sft_data)
    assert not any(e.instruction == "untranslated english" for e in kept)
    assert not any(e.source and e.source.instruction.startswith("I enjoy") for e in kept)


def test_distill_refuses_seeds_without_an_open_licence(tmp_path):
    recipe = _recipe_with_seeds(tmp_path, seed_licence="UNSET")
    with pytest.raises(LicenceError):
        distill_sft(recipe, DemoTeacher())


def test_native_review_decisions_are_applied(tmp_path):
    recipe = _recipe_with_seeds(tmp_path, n=40, review_fraction=1.0)
    distill_sft(recipe, DemoTeacher())
    queue_path = recipe.paths.review_queue("hi")
    queue = load_examples(queue_path)
    assert len(queue) >= 2
    rejected_id, corrected_id = queue[0].id, queue[1].id
    queue[0] = queue[0].model_copy(update={"review": "rejected"})
    queue[1] = queue[1].model_copy(update={"review": "approved", "response": "सही उत्तर।"})
    queue_path.write_text("".join(e.model_dump_json() + "\n" for e in queue), "utf-8")

    counts = apply_reviews(recipe)
    assert counts["rejected"] == 1 and counts["approved"] == 1
    kept = {e.id: e for e in load_examples(recipe.paths.sft_data)}
    assert rejected_id not in kept
    assert kept[corrected_id].response == "सही उत्तर।"
    dropped = load_examples(recipe.paths.sft_rejected)
    assert any(e.id == rejected_id and e.reason == "native_review" for e in dropped)
