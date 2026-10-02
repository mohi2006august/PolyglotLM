import json

import torch

from polyglotlm.distill.teacher import DemoTeacher
from polyglotlm.distill.teacher_generate import SFTExample, distill_sft
from polyglotlm.modeling import load_model
from polyglotlm.prompts import PLAIN_TEMPLATE, format_prompt
from polyglotlm.tokenizer import load_tokenizer
from polyglotlm.train.sft import IGNORE_INDEX, collate, encode_example, run_sft


def _example(instruction="Say hi", response="hi there"):
    return SFTExample(id="x", lang="en", origin="seed", instruction=instruction, response=response)


def test_plain_template_is_used_without_a_chat_template(built):
    tokenizer = load_tokenizer(built.base_model)
    assert format_prompt(tokenizer, "Q") == PLAIN_TEMPLATE.format(instruction="Q")


def test_encode_example_masks_the_prompt_and_ends_with_eos(built):
    tokenizer = load_tokenizer(built.base_model)
    ids, labels = encode_example(tokenizer, _example(), max_length=256)
    prompt_len = len(tokenizer.encode(format_prompt(tokenizer, "Say hi"), add_special_tokens=False))
    assert labels[:prompt_len] == [IGNORE_INDEX] * prompt_len
    assert labels[prompt_len:] == ids[prompt_len:]
    assert ids[-1] == tokenizer.eos_token_id
    short_ids, short_labels = encode_example(tokenizer, _example(), max_length=5)
    assert len(short_ids) == len(short_labels) == 5


def test_collate_pads_and_masks():
    batch = collate([([5, 6, 7], [IGNORE_INDEX, 6, 7]), ([8], [8])], pad_id=0)
    assert batch["input_ids"].tolist() == [[5, 6, 7], [8, 0, 0]]
    assert batch["labels"].tolist() == [[IGNORE_INDEX, 6, 7], [8, IGNORE_INDEX, IGNORE_INDEX]]
    assert batch["attention_mask"].tolist() == [[1, 1, 1], [1, 0, 0]]


def test_run_sft_trains_on_distilled_data(trained, tmp_path):
    from polyglotlm.demo.corpus import demo_seed_pairs

    seeds = tmp_path / "seeds.jsonl"
    seeds.write_text("".join(json.dumps(p) + "\n" for p in demo_seed_pairs(20, 0)), "utf-8")
    recipe = trained.model_copy(
        update={
            "distill": trained.distill.model_copy(
                update={"seed_path": str(seeds), "seed_licence": "CC0-1.0", "n": 40, "workers": 2}
            )
        }
    )
    distill_sft(recipe, DemoTeacher())
    out_dir = run_sft(recipe)

    with recipe.paths.metrics("sft").open(encoding="utf-8") as fh:
        records = [json.loads(line) for line in fh]
    evals = [r for r in records if r["kind"] == "eval"]
    assert evals and all(set(r["heldout_ppl"]) == set(recipe.all_codes) for r in evals)
    assert {r["phase"] for r in records if r["kind"] == "train"} == {"full"}
    summary = json.loads((out_dir / "sft.json").read_text("utf-8"))
    assert set(summary["mixture"]) == set(recipe.all_codes)

    model = load_model(out_dir)
    tokenizer = load_tokenizer(out_dir)
    prompt = tokenizer.encode(format_prompt(tokenizer, "x"), return_tensors="pt")
    with torch.no_grad():
        assert model(input_ids=prompt).logits.shape[-1] == len(tokenizer)
