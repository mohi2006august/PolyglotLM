# PolyglotLM: Brain

> Second brain for this project: concepts, mental models, research, ideas and decisions. Keep it messy and alive.

## 1. Why This Project

It combines almost every model-level concept (tokenization, embeddings, fine-tuning, LoRA, distillation, overfitting, quantisation) into one socially valuable goal: making AI good and affordable in languages billions of people speak.

## 2. Concept Primers

### Tokenization

Converting raw text into the integer token IDs a model consumes. Subword algorithms (BPE, WordPiece, Unigram) balance vocabulary size against sequence length.

- **BPE**: Byte-Pair Encoding: iteratively merges the most frequent adjacent pair; byte-level BPE never hits unknown tokens.
- **Unigram LM**: Starts with a large vocabulary and prunes tokens that least reduce corpus likelihood (SentencePiece).
- **Fertility**: Average tokens per word; high fertility means higher cost and less effective context for that language.
- **Special tokens**: BOS/EOS/PAD and chat-template markers that structure model input.

> **Mental model:** Tokens are the unit of cost, latency and context. The same sentence can cost 3x more in Hindi than in English, and that 'token tax' is a product and fairness issue.

### Fine Tuning

Continuing training of a pre-trained model on task- or domain-specific data to change its behaviour, style or knowledge.

- **SFT**: Supervised fine-tuning on prompt → ideal response pairs.
- **DPO / preference tuning**: Optimising directly on chosen-vs-rejected pairs without a separate reward model.
- **Catastrophic forgetting**: Losing general capabilities while specialising.
- **PEFT**: Parameter-efficient methods (LoRA, adapters, prefix tuning) that train a small fraction of weights.

> **Mental model:** Fine-tune for form (format, tone, task behaviour). Retrieve for facts. Data quality beats data quantity almost every time.

### LoRA

Low-Rank Adaptation: freeze the base weights W and learn a low-rank update ΔW = B·A (rank r ≪ d), training well under 1% of parameters with small, swappable adapters.

- **Rank r / alpha**: Capacity of the update and its scaling (alpha/r).
- **Target modules**: Which projections get adapters (q,k,v,o, MLP).
- **QLoRA**: LoRA on top of a 4-bit quantised base model; fine-tune big models on one GPU.
- **Multi-LoRA serving**: One base model in memory serving many adapters per request (S-LoRA, Punica).

> **Mental model:** The base model is the OS and adapters are apps. They are cheap to train, store and switch, so give each tenant, task or style its own adapter.

### Knowledge Distillation

Training a small 'student' model to mimic a large 'teacher' model's outputs (soft labels, logits, rationales), keeping most of the quality at a fraction of the cost.

- **Soft targets**: Teacher probability distributions softened with temperature T carry 'dark knowledge'.
- **KD loss**: KL divergence between teacher and student distributions, mixed with hard-label cross-entropy.
- **Sequence-level KD**: Student trained on teacher-generated outputs (synthetic data distillation).
- **Capacity gap**: Very large teacher/student gaps can hurt; teacher assistants bridge them.

> **Mental model:** You are compressing a function, not a dataset. The teacher's uncertainty is signal, so preserve it.

### Overfitting

When a model fits noise or specifics of the training set so closely that it performs worse on unseen data; train loss keeps dropping while validation loss rises.

- **Generalisation gap**: Difference between training and validation/test performance.
- **Regularisation**: Weight decay, dropout, data augmentation, early stopping, smaller capacity.
- **Memorisation**: Verbatim reproduction of training examples; measured with canaries or extraction probes.
- **Data leakage**: Test information leaking into training, which hides overfitting behind inflated scores.

> **Mental model:** A low training loss is a claim. The held-out set is the evidence. Always keep a test set you never tune against.

### Quantization

Representing weights (and sometimes activations/KV cache) with fewer bits (int8, int4, fp8) to cut memory and increase speed, with some accuracy loss.

- **PTQ vs QAT**: Post-training quantisation vs quantisation-aware training.
- **GPTQ / AWQ**: Weight-only int4 methods using second-order info or activation-aware scaling.
- **GGUF**: llama.cpp file format with k-quants (Q4_K_M, Q5_K_M, …) for CPU/edge inference.
- **Outliers**: A few large-magnitude activation channels that break naive quantisation.

> **Mental model:** Memory bandwidth, not FLOPs, bounds LLM decoding. Halving bytes per weight roughly doubles tokens/sec until you hit compute limits.

### Embeddings

Dense vector representations where geometric closeness reflects semantic similarity; the backbone of search, clustering, recommendation and RAG.

- **Contrastive training**: Pulling positive pairs together and pushing negatives apart (e.g. InfoNCE).
- **Matryoshka embeddings**: Vectors whose prefixes are themselves usable lower-dimensional embeddings.
- **Anisotropy**: Embeddings crowding into a narrow cone, which hurts cosine similarity spread.
- **Embedding drift**: Distribution shift in vectors over time or after a model upgrade.

> **Mental model:** An embedding model is an opinion about what 'similar' means. Benchmarks like MTEB are a starting point, so evaluate on your own queries.

### LLMs

Large Language Models: transformer networks trained on large text corpora to predict the next token, then aligned (SFT and RLHF/RLAIF) to follow instructions and use tools.

- **Prompting**: System prompts, few-shot examples, structured outputs, tool use.
- **Sampling**: Temperature, top-p and max tokens shape determinism and length.
- **Evals**: Task-specific test sets plus LLM-as-judge with calibrated rubrics.
- **Routing**: Sending each request to the cheapest model that meets its quality bar.

> **Mental model:** Treat the LLM as a probabilistic component. Wrap it with evals, guardrails, retries and observability, the same way you would a flaky network dependency.

## 3. What I Want to Learn

- [ ] Vocabulary surgery and embedding initialisation.
- [ ] Continued pre-training at scale.
- [ ] Distilled multilingual instruction data.
- [ ] Forgetting vs adaptation trade-offs.
- [ ] Mobile deployment.

## 4. Reading & Research List

- Petrov et al., *Language Model Tokenizers Introduce Unfairness* (2023)
- Gee et al., *Fast Vocabulary Transfer* (2022)
- Cui et al., *Chinese LLaMA/Alpaca: vocabulary extension* (2023)
- AI4Bharat IndicTrans2 / IndicCorp papers
- Gupta et al., *Continual Pre-Training of LLMs: How to (re)warm your model* (2023)

## 5. Experiments to Run

| # | Hypothesis | How to test |
|---|---|---|
| 1 | FVT init converges 2x faster than random init | Loss curves. |
| 2 | 8k vs 20k new tokens per language | Fertility vs quality. |
| 3 | 15% English replay keeps regression < 3 pts | Sweep 0/5/15/30%. |

## 6. Ideas Backlog

- Speech front-end (ASR/TTS).
- Code-mixed text (Hinglish).
- Per-language LoRA adapters hot-swapped in one app.

## 7. Decision Log

| Date | Decision | Reason |
|---|---|---|
| YYYY-MM-DD | Append-only vocab | Keeps compatibility with base model tooling. |
| YYYY-MM-DD | Held-out native text for early stopping | Translated val data hid overfitting. |
| 2026-10-02 | New tokens are non-special *added tokens*, restricted to target-script characters | Appending BPE merges would only fire after every base merge and need intermediate tokens; added tokens are matched leftmost-longest before BPE, keep base IDs untouched and can never change Latin-script tokenization. llama.cpp keeps them as USER_DEFINED tokens. Targets byte-level BPE bases (Llama-3.x). |
| 2026-10-02 | Script-based language ID by default, fastText `lid.176` optional | Works offline with no extra wheel (fastText does not build cleanly on every platform). Recipes with two languages in one script are rejected unless `langid_model` is set. |
| 2026-10-02 | `cpt_mix_ratio` optional; default mixture is n^0.3 temperature sampling + English replay | TECHNICAL.md specifies both; explicit ratios win when every language sets one, otherwise corpus sizes drive the mix. |
| 2026-10-02 | Compare checkpoints with different tokenizers by bits per character | Token perplexity falls when tokens get longer even if the model got no better; bits/char measures the same text under both tokenizers. |
| 2026-10-02 | English regression measured in benchmark points, falling back to relative bits/char change | The 3-point budget in the PRD is in points; without English task files the report still says something, and says which metric it used. |
| 2026-10-02 | Teacher = `claude-opus-5-5` with structured output and server-side refusal fallbacks, behind a `Teacher` protocol | Typed outputs remove JSON parsing failures; the protocol keeps the provider swappable. Check the teacher provider's terms before training on its outputs. |
| 2026-10-02 | Back-translation chrF gates translated SFT data; native generations go to review instead | Native items have no reference to compare against; a sampled native-speaker queue is the check that scales. |
| 2026-10-02 | Embedding teacher defaults to the base LLM's mean-pooled English states | No new dependency, same tokenizer family as the student; a sentence-transformers teacher is one config line away. |
| 2026-10-02 | Single-device trainer; FSDP/multi-GPU not wired yet | Everything else is testable on CPU; distributed training needs real hardware to verify. |

## 8. Scratchpad

- _Dump raw thoughts, links and half-formed ideas here._
