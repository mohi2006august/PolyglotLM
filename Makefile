RECIPE ?= recipes/indic-3b.yaml
LANGS ?=
QUANT ?= Q4_K_M
POLY := uv run poly

.PHONY: install data tokenizer cpt distill sft eval quant embed demo test lint

install:
	uv sync

# FR-04: download, clean, language-ID, dedupe, split held-out, write snapshot manifest.
data:
	$(POLY) data build --recipe $(RECIPE) $(if $(LANGS),--langs $(LANGS))

# FR-01: token tax before, extend the tokenizer, token tax after.
tokenizer:
	$(POLY) fertility --recipe $(RECIPE) --tokenizer base
	$(POLY) tokenizer extend --recipe $(RECIPE)
	$(POLY) fertility --recipe $(RECIPE) --tokenizer extended

# FR-02 + FR-03: model surgery (if not done yet), then continued pre-training.
cpt:
	$(POLY) train cpt --recipe $(RECIPE)

# FR-05: teacher-distilled instruction data. Needs TEACHER_API_KEY.
distill:
	$(POLY) distill sft --recipe $(RECIPE)

sft:
	$(POLY) train sft --recipe $(RECIPE)

# Every checkpoint is evaluated on all languages, English included.
eval:
	$(POLY) eval --all --recipe $(RECIPE)
	$(POLY) report --recipe $(RECIPE)

# FR-07: needs a llama.cpp checkout (LLAMA_CPP_DIR).
quant:
	$(POLY) quant --recipe $(RECIPE) --q $(QUANT)

# FR-08
embed:
	$(POLY) embed distill --recipe $(RECIPE)

# End-to-end CPU smoke run on synthetic corpora.
demo:
	$(POLY) demo run

test:
	uv run pytest

lint:
	uv run ruff check .
	uv run ruff format --check .
