# PolyglotLM: Design Document

> Adapt an open LLM to low-resource languages (e.g. Hindi, Tamil, Urdu, Bengali) by extending the tokenizer, continued pre-training, distilling from a stronger teacher and shipping a quantised model for phones.
>
> **Concepts:** `Tokenization`, `Fine Tuning`, `LoRA`, `Knowledge Distillation`, `Overfitting`, `Quantization`, `Embeddings`, `LLMs`

## 1. Design Principles

- Native speakers are part of the evaluation loop.
- Report the token tax before and after, not just accuracy.

## 2. Key User Flows

### Adapt a new language

1. Add LanguageConfig.
2. Run fertility report.
3. Extend tokenizer.
4. Train CPT + SFT.
5. Review per-language report card.
6. Publish model.

## 3. Screens & Surfaces

| Screen | What it shows / does |
|---|---|
| Fertility dashboard | Tokens/word per language before vs after. |
| Training monitor | Per-language loss + English regression. |
| Report card | Benchmarks, sample generations, known limitations. |
| Mobile demo | Offline chat app. |

## 4. Wireframe (Primary Screen)

```text
Token tax (tokens per word)     base    extended   saving
Hindi   (hi)                    4.8     1.9        −60%
Tamil   (ta)                    7.1     2.4        −66%
Urdu    (ur)                    4.2     1.8        −57%
Bengali (bn)                    5.9     2.2        −63%
English (en)                    1.3     1.3          0%
Benchmarks: ta-QA 41→63 · hi-QA 48→66 · MMLU(en) 58.2→56.9 (−1.3 ✔)
```

## 5. Visual Language

| Token | Value |
|---|---|
| Primary accent | `#EA580C` (saffron) |
| Neutral surface | `#0F1117` dark / `#FAFAF9` light |
| Success / Warning / Danger | `#22C55E` / `#F59E0B` / `#EF4444` |
| Type | Inter for UI, JetBrains Mono for code, tokens and numbers |
| Spacing scale | 4 / 8 / 12 / 16 / 24 / 32 px |
| Radius | 8px cards, 6px inputs, full for pills |

## 6. States

| State | Treatment |
|---|---|
| English regression alert | Red chip on report card. |

## 7. Accessibility

- WCAG 2.2 AA contrast in both themes; never encode meaning with colour alone (pair with icon or label).
- Full keyboard navigation with visible focus rings; `/` focuses search, `?` opens shortcuts.
- Charts ship with a data-table fallback and `aria-describedby` summaries.
- Respect `prefers-reduced-motion`; streaming text is announced politely via `aria-live`.
