import json

import pytest
import torch

from polyglotlm.demo.base import build_base_model
from polyglotlm.demo.corpus import all_sentences
from polyglotlm.modeling import embeddings_tied, load_model
from polyglotlm.surgery import run_surgery
from polyglotlm.surgery.init_embeddings import init_new_embeddings
from polyglotlm.surgery.resize import resize_embeddings, vocab_rows
from polyglotlm.tokenizer import load_tokenizer
from tests.conftest import make_recipe


@pytest.fixture(scope="module")
def tokenizers(extended):
    return load_tokenizer(extended.base_model), load_tokenizer(extended.paths.extended_tokenizer)


@pytest.mark.parametrize("tied", [False, True])
def test_resize_grows_embeddings_and_head_and_keeps_old_rows(tied):
    model = build_base_model(vocab_size=300, tie_embeddings=tied)
    old_embed = model.get_input_embeddings().weight.detach().clone()
    old_head = model.get_output_embeddings().weight.detach().clone()
    assert resize_embeddings(model, 320) == (300, 320)
    assert vocab_rows(model) == (320, 320)
    assert embeddings_tied(model) is tied
    assert model.config.vocab_size == 320
    assert torch.equal(model.get_input_embeddings().weight[:300], old_embed)
    assert torch.equal(model.get_output_embeddings().weight[:300], old_head)


def test_resize_refuses_to_shrink():
    with pytest.raises(ValueError, match="shrink"):
        resize_embeddings(build_base_model(vocab_size=300), 200)


def test_fvt_sets_new_rows_to_the_mean_of_their_sub_tokens(extended, tokenizers):
    base_tok, ext_tok = tokenizers
    model = load_model(extended.base_model)
    resize_embeddings(model, len(ext_tok))
    report = init_new_embeddings(model, base_tok, ext_tok)
    assert report.new_tokens == len(ext_tok) - len(base_tok)
    embed = model.get_input_embeddings().weight
    head = model.get_output_embeddings().weight
    for token_id in (len(base_tok), len(ext_tok) - 1):
        sub_ids = base_tok.encode(ext_tok.convert_ids_to_tokens(token_id), add_special_tokens=False)
        assert torch.allclose(embed[token_id], embed[sub_ids].mean(0), atol=1e-6)
        assert torch.allclose(head[token_id], head[sub_ids].mean(0), atol=1e-6)


def test_dictionary_init_pulls_glossed_tokens_towards_english(extended, tokenizers):
    base_tok, ext_tok = tokenizers
    token_id = len(base_tok)
    word = ext_tok.convert_ids_to_tokens(token_id).strip()
    fvt_model, dict_model = load_model(extended.base_model), load_model(extended.base_model)
    for model in (fvt_model, dict_model):
        resize_embeddings(model, len(ext_tok))
    init_new_embeddings(fvt_model, base_tok, ext_tok)
    report = init_new_embeddings(dict_model, base_tok, ext_tok, dictionary={word: "market"})
    assert report.dictionary_hits == 1
    fvt_row = fvt_model.get_input_embeddings().weight[token_id]
    dict_row = dict_model.get_input_embeddings().weight[token_id]
    assert not torch.allclose(fvt_row, dict_row)
    # The LM head of an untied model keeps plain FVT.
    assert torch.equal(
        fvt_model.get_output_embeddings().weight[token_id],
        dict_model.get_output_embeddings().weight[token_id],
    )


def test_surgery_keeps_english_outputs_unchanged(extended, tokenizers):
    base_tok, _ = tokenizers
    before = load_model(extended.base_model)
    out_dir = run_surgery(extended)
    after = load_model(out_dir)
    summary = json.loads((out_dir / "surgery.json").read_text("utf-8"))
    assert summary["old_vocab"] == len(base_tok) and summary["new_vocab"] > len(base_tok)
    assert (
        load_tokenizer(out_dir).get_vocab()
        == load_tokenizer(extended.paths.extended_tokenizer).get_vocab()
    )
    prompt = torch.tensor([base_tok.encode(all_sentences("en")[3])])
    with torch.no_grad():
        old_logits = before(input_ids=prompt).logits
        new_logits = after(input_ids=prompt).logits
    assert torch.allclose(new_logits[..., : len(base_tok)], old_logits, atol=1e-5)


def test_new_token_logits_equal_the_mean_of_sub_token_logits(extended, tokenizers):
    base_tok, ext_tok = tokenizers
    model = load_model(run_surgery(extended))
    prompt = torch.tensor([base_tok.encode(all_sentences("en")[0])])
    with torch.no_grad():
        logits = model(input_ids=prompt).logits[0, -1]
    token_id = len(ext_tok) - 1
    sub_ids = base_tok.encode(ext_tok.convert_ids_to_tokens(token_id), add_special_tokens=False)
    assert logits[token_id] == pytest.approx(logits[sub_ids].mean().item(), abs=1e-4)


def test_surgery_rejects_a_model_whose_rows_do_not_match_its_tokenizer(extended, tmp_path):
    from polyglotlm.modeling import save_model

    mismatched = tmp_path / "mismatched"
    tokenizer = load_tokenizer(extended.base_model)
    save_model(build_base_model(vocab_size=len(tokenizer) + 5), tokenizer, mismatched)
    recipe = make_recipe(
        tmp_path,
        base_model=str(mismatched),
        workdir=extended.workdir,
    )
    with pytest.raises(ValueError, match="would not line up"):
        run_surgery(recipe)
