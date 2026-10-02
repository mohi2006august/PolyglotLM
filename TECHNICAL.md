# PolyglotLM: Technical Specification

> Adapt an open LLM to low-resource languages (e.g. Hindi, Tamil, Urdu, Bengali) by extending the tokenizer, continued pre-training, distilling from a stronger teacher and shipping a quantised model for phones.
>
> **Concepts:** `Tokenization`, `Fine Tuning`, `LoRA`, `Knowledge Distillation`, `Overfitting`, `Quantization`, `Embeddings`, `LLMs`

## 1. Tech Stack

| Layer | Choice | Why |
|---|---|---|
| Training | PyTorch, HF Transformers, PEFT, FSDP/DeepSpeed | Scale. |
| Tokenizers | SentencePiece, HF tokenizers | Standard. |
| Data | datatrove / custom, fastText LangID, MinHash | Cleaning at scale. |
| Eval | lm-eval-harness + Indic benchmarks | Coverage. |
| Edge | llama.cpp (Android NDK) | Mobile inference. |

## 2. Repository Layout

```text
polyglotlm/
├── data/          # download.py, clean.py, langid.py, dedupe.py
├── tokenizer/     # train_new.py, merge_vocab.py, fertility.py
├── surgery/       # resize.py, init_embeddings.py
├── train/         # cpt.py, sft.py, configs/
├── distill/       # teacher_generate.py, back_translate.py
├── eval/          # tasks/, english_regression.py, report.py
├── quant/         # to_gguf.sh
└── mobile/android/
```

## 3. Data Models

```python
class LanguageConfig(BaseModel):
    code: str                        # "ta", "hi", "ur", "bn"
    script: str                      # "Taml", "Deva", "Arab", "Beng"
    new_tokens: int                  # e.g. 8000
    cpt_mix_ratio: float             # share of CPT tokens
    heldout_ppl_path: str

class TrainingRecipe(BaseModel):
    base_model: str                  # e.g. "Llama-3.2-3B"
    languages: list[LanguageConfig]
    english_replay: float = 0.15
    lora_rank: int = 64
    train_embeddings: bool = True
    embed_warmup_steps: int = 1000
```

## 4. API Surface

| Method | Endpoint / Tool | Purpose |
|---|---|---|
| cli | poly fertility --tokenizer base --langs hi,ta,ur,bn | Baseline token-tax report. |
| cli | poly tokenizer extend --lang ta --new-tokens 8000 | Extend tokenizer. |
| cli | poly train cpt --recipe recipes/indic-3b.yaml | Continued pre-training. |
| cli | poly distill sft --teacher T --n 100000 | Build instruction data. |
| cli | poly eval --all && poly quant --q Q4_K_M | Evaluate and quantise. |

## 5. Core Algorithms

### Vocabulary extension

Train a target-language SentencePiece model and keep tokens not already in the base vocab. Rank candidates by `freq × (base_tokens(t) − 1)` (tokens saved) and append the top-N at the end so existing IDs are unchanged.

### Embedding init (FVT)

For a new token t, tokenise its string with the old tokenizer into s₁..s_m and set `E[t] = mean(E[s₁..s_m])`. Do the same for LM-head rows. The model 'understands' new tokens from step 0.

### Replay + curriculum

Batch mixture: target languages weighted by `n_i^0.3` (temperature sampling to upweight low-resource) + 15% English replay to limit forgetting.

### Overfitting guard

Early stop per stage on held-out native text perplexity, not on translated data. Alert if English eval drops > 3 pts.

## 6. Configuration

| Variable | Purpose |
|---|---|
| HF_TOKEN | Models/datasets |
| WANDB_API_KEY | Tracking |
| TEACHER_API_KEY | Teacher model |
| DATA_BUCKET | Corpora |

## 7. Testing Strategy

- Tokenizer round-trip on all scripts (NFC/NFD).
- Old token IDs unchanged after extension.
- Surgery test: base model outputs unchanged for English prompts right after resize.
- Eval harness determinism.

## 8. Performance Budgets

- CPT throughput ≥ 40% MFU on H100.
- Mobile ≥ 8 tok/s (Q4, 3B).

## 9. Local Setup

```bash
uv sync
make data LANGS=hi,ta
make tokenizer
make cpt RECIPE=recipes/indic-3b.yaml
make eval
```
