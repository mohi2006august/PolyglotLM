import json
import shutil
import sys

import pytest

from polyglotlm.quant.bench import parse_llama_bench, words_per_second
from polyglotlm.quant.gguf import LlamaCpp, plan, preflight, quantise
from polyglotlm.tokenizer.fertility import Fertility

# A stand-in for convert_hf_to_gguf.py / llama-quantize: writes its output file argument.
FAKE_TOOL = """
import sys
args = sys.argv[1:]
out = args[args.index("--outfile") + 1] if "--outfile" in args else args[1]
open(out, "wb").write(b"GGUF" + b"\\0" * 64)
"""


@pytest.fixture
def fake_tools(tmp_path):
    script = tmp_path / "fake_tool.py"
    script.write_text(FAKE_TOOL, "utf-8")
    cmd = [sys.executable, str(script)]
    return LlamaCpp(convert=cmd, quantize=cmd, bench=cmd)


@pytest.fixture
def model_copy(trained, tmp_path):
    target = tmp_path / "model"
    shutil.copytree(trained.paths.model("cpt"), target, ignore=shutil.ignore_patterns("adapter"))
    return target


def test_preflight_accepts_a_merged_model_with_its_tokenizer(model_copy):
    assert preflight(model_copy) == []


def test_preflight_rejects_missing_tokenizer_and_unresized_vocab(model_copy):
    config = json.loads((model_copy / "config.json").read_text("utf-8"))
    config["vocab_size"] -= 7
    (model_copy / "config.json").write_text(json.dumps(config), "utf-8")
    assert any("not resized" in p for p in preflight(model_copy))
    (model_copy / "tokenizer.json").unlink()
    assert any("tokenizer.json missing" in p for p in preflight(model_copy))


def test_plan_builds_llama_cpp_commands(tmp_path, fake_tools):
    steps = plan(tmp_path / "m", tmp_path / "out", "Q4_K_M", fake_tools, "indic-sft")
    assert steps.convert_cmd[-5:] == [
        str(tmp_path / "m"),
        "--outfile",
        str(steps.f16),
        "--outtype",
        "f16",
    ]
    assert steps.quantize_cmd[-3:] == [str(steps.f16), str(steps.quantized), "Q4_K_M"]
    assert steps.quantized.name == "indic-sft-Q4_K_M.gguf"
    with pytest.raises(ValueError, match="unknown quant type"):
        plan(tmp_path, tmp_path, "Q3_XYZ", fake_tools, "x")


def test_quantise_runs_convert_then_quantize(model_copy, tmp_path, fake_tools):
    out = quantise(model_copy, tmp_path / "gguf", "Q4_K_M", fake_tools, "demo")
    assert out.name == "demo-Q4_K_M.gguf" and out.read_bytes().startswith(b"GGUF")
    assert (tmp_path / "gguf" / "demo-F16.gguf").exists()


def test_quantise_refuses_a_model_that_fails_preflight(model_copy, tmp_path, fake_tools):
    (model_copy / "tokenizer.json").unlink()
    with pytest.raises(ValueError, match="not ready"):
        quantise(model_copy, tmp_path / "gguf", "Q4_K_M", fake_tools, "demo")


def test_discover_finds_tools_in_a_cmake_build(tmp_path, monkeypatch):
    (tmp_path / "convert_hf_to_gguf.py").write_text("", "utf-8")
    bin_dir = tmp_path / "build" / "bin"
    bin_dir.mkdir(parents=True)
    suffix = ".exe" if sys.platform == "win32" else ""
    for name in ("llama-quantize", "llama-bench"):
        (bin_dir / f"{name}{suffix}").write_text("", "utf-8")
    monkeypatch.setenv("LLAMA_CPP_DIR", str(tmp_path))
    tools = LlamaCpp.discover()
    assert tools.quantize == [str(bin_dir / f"llama-quantize{suffix}")]
    assert tools.convert[-1] == str(tmp_path / "convert_hf_to_gguf.py")
    with pytest.raises(FileNotFoundError, match="LLAMA_CPP_DIR"):
        LlamaCpp.discover(tmp_path / "nowhere")


def test_parse_llama_bench_json():
    records = [
        {"model_filename": "m.gguf", "n_prompt": 128, "n_gen": 0, "avg_ts": 41.5},
        {"model_filename": "m.gguf", "n_prompt": 0, "n_gen": 64, "avg_ts": 9.25},
    ]
    speed = parse_llama_bench(records)
    assert (speed.prompt_tps, speed.gen_tps) == (41.5, 9.25)
    with pytest.raises(ValueError, match="avg_ts"):
        parse_llama_bench([{"n_gen": 64}])


def test_words_per_second_exposes_the_token_tax():
    speed = parse_llama_bench([{"n_prompt": 0, "n_gen": 64, "avg_ts": 8.0}])
    fertility = {
        "hi": Fertility("hi", docs=10, words=100, tokens=190),
        "en": Fertility("en", docs=10, words=100, tokens=130),
    }
    result = words_per_second(speed, fertility)
    assert result["hi"]["gen_words_per_s"] == pytest.approx(8.0 / 1.9, abs=0.01)
    assert result["en"]["gen_words_per_s"] > result["hi"]["gen_words_per_s"]
    assert result["hi"]["meets_target"] is True
