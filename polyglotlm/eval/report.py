"""Report card: token tax, benchmarks, training curves, samples and known limitations.

`poly report` gathers what earlier stages wrote (fertility reports, eval results, metrics,
SFT data) and renders `report/index.html` (self-contained, light + dark, every chart paired
with a data table) and `report/MODEL_CARD.md`.
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
from html import escape
from pathlib import Path
from typing import Any

from polyglotlm.config import TrainingRecipe, language_name
from polyglotlm.eval.charts import Series, fmt, grouped_bars, line_chart
from polyglotlm.eval.english_regression import compare
from polyglotlm.logs import get_logger
from polyglotlm.tokenizer.fertility import Fertility, load_report, saving

log = get_logger(__name__)

MAX_LINE_SERIES = 8  # categorical slots; more languages fall back to the table only

STATIC_LIMITATIONS = (
    "Benchmarks only cover the task files provided; they are not a safety evaluation.",
    "Teacher-generated data can carry the teacher's errors and cultural blind spots; "
    "native-speaker review covers a sample, not every example.",
    "Script-based language ID cannot separate languages that share a script.",
)


@dataclass
class ReportData:
    recipe: TrainingRecipe
    fertility_base: dict[str, Fertility] | None = None
    fertility_extended: dict[str, Fertility] | None = None
    evals: dict[str, dict[str, Any]] = field(default_factory=dict)
    comparison: dict[str, Any] | None = None
    curves: dict[str, list[dict[str, Any]]] = field(default_factory=dict)
    summaries: dict[str, dict[str, Any]] = field(default_factory=dict)
    sft_counts: dict[str, Counter[str]] = field(default_factory=dict)
    snapshot: dict[str, Any] | None = None

    @property
    def adapted(self) -> str | None:
        return next((name for name in ("sft", "cpt") if name in self.evals), None)

    @property
    def alerts(self) -> list[str]:
        return [f"{stage}: {a}" for stage, s in self.summaries.items() for a in s.get("alerts", [])]


def _json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def _eval_records(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as fh:
        return [r for r in map(json.loads, fh) if r.get("kind") == "eval"]


def _sft_counts(path: Path) -> dict[str, Counter[str]]:
    counts: dict[str, Counter[str]] = {"origin": Counter(), "review": Counter()}
    if path.exists():
        with path.open(encoding="utf-8") as fh:
            for row in map(json.loads, fh):
                counts["origin"][f"{row['lang']}:{row['origin']}"] += 1
                counts["review"][row.get("review", "none")] += 1
    return counts


def collect(recipe: TrainingRecipe) -> ReportData:
    paths = recipe.paths
    data = ReportData(recipe=recipe)
    if paths.fertility("base").exists():
        data.fertility_base = load_report(paths.fertility("base"))
    if paths.fertility("extended").exists():
        data.fertility_extended = load_report(paths.fertility("extended"))
    for name in ("base", "cpt", "sft"):
        if (result := _json(paths.eval(name))) is not None:
            data.evals[name] = result
    if "base" in data.evals and data.adapted:
        data.comparison = compare(
            data.evals["base"], data.evals[data.adapted], recipe.max_english_regression
        )
    for stage in ("cpt", "sft"):
        if records := _eval_records(paths.metrics(stage)):
            data.curves[stage] = records
        if (summary := _json(paths.model(stage) / f"{stage}.json")) is not None:
            data.summaries[stage] = summary
    data.sft_counts = _sft_counts(paths.sft_data)
    data.snapshot = _json(paths.snapshot)
    return data


def limitations(data: ReportData) -> list[str]:
    items = list(STATIC_LIMITATIONS)
    verdict = (data.comparison or {}).get("english", {})
    if verdict.get("regressed"):
        items.insert(0, f"English regressed by {verdict['drop']:.1f} ({verdict['metric']}).")
    if pending := data.sft_counts.get("review", Counter())["pending"]:
        items.append(f"{pending} SFT examples are still awaiting native-speaker review.")
    if not data.recipe.eval_tasks_dir:
        items.append("No benchmark task files were configured; scores are perplexity only.")
    items.extend(f"Training alert - {alert}" for alert in data.alerts[:10])
    return items


# --- HTML -------------------------------------------------------------------------------------

CSS = """
:root {
  color-scheme: light;
  --surface: #FAFAF9; --surface-2: #F1F0EE; --grid: #E4E3DF;
  --text-primary: #1C1B19; --text-secondary: #57554F;
  --accent: #EA580C; --success: #22C55E; --warning: #F59E0B; --danger: #EF4444;
  --compare-base: #2a78d6; --compare-ext: #EA580C;
  --series-1: #2a78d6; --series-2: #eb6834; --series-3: #1baf7a; --series-4: #eda100;
  --series-5: #e87ba4; --series-6: #008300; --series-7: #4a3aa7; --series-8: #e34948;
  --font-sans: Inter, "Segoe UI", system-ui, sans-serif;
  --font-mono: "JetBrains Mono", ui-monospace, "Cascadia Mono", Consolas, monospace;
}
@media (prefers-color-scheme: dark) {
  :root:not([data-theme="light"]) { DARK }
}
:root[data-theme="dark"] { DARK }
* { box-sizing: border-box; }
body { margin: 0; background: var(--surface); color: var(--text-primary);
  font: 15px/1.5 var(--font-sans); }
main { max-width: 760px; margin: 0 auto; padding: 32px 16px 64px; }
h1 { font-size: 24px; margin: 0 0 4px; }
h2 { font-size: 18px; margin: 40px 0 8px; }
p.meta, .note { color: var(--text-secondary); margin: 0 0 16px; }
code { font-family: var(--font-mono); font-size: 0.9em; overflow-wrap: anywhere; }
.chip { display: inline-flex; align-items: center; gap: 8px; padding: 4px 12px;
  border-radius: 999px; border: 1px solid var(--chip); font-weight: 600;
  background: color-mix(in srgb, var(--chip) 14%, transparent); }
.chip svg { width: 16px; height: 16px; fill: var(--chip); flex: none; }
.chip.ok { --chip: var(--success); } .chip.bad { --chip: var(--danger); }
.chip.unknown { --chip: var(--warning); }
figure { margin: 16px 0; padding: 16px; background: var(--surface-2); border-radius: 8px; }
figcaption { font-weight: 600; margin-bottom: 4px; }
.legend { display: flex; flex-wrap: wrap; gap: 4px 16px; margin: 4px 0 8px;
  color: var(--text-secondary); font-size: 13px; }
.legend span::before { content: ""; display: inline-block; width: 10px; height: 10px;
  border-radius: 2px; margin-right: 6px; background: var(--swatch); vertical-align: -1px; }
svg.chart { width: 100%; height: auto; display: block; overflow: visible; }
.chart .grid, .chart .axis { stroke: var(--grid); stroke-width: 1; }
.chart .tick, .chart .value { fill: var(--text-secondary); }
.chart .label { fill: var(--text-primary); font-size: 13px; }
.chart .line { fill: none; stroke-width: 2; stroke-linejoin: round; stroke-linecap: round; }
.chart .dot { stroke: var(--surface-2); stroke-width: 2; }
.chart .hit { fill: transparent; }
.chart .pt:hover .dot, .chart path:hover { opacity: 0.8; }
details { margin-top: 8px; } summary { cursor: pointer; color: var(--text-secondary); }
summary:focus-visible, a:focus-visible { outline: 2px solid var(--accent); outline-offset: 2px; }
.table-wrap { overflow-x: auto; }
table { border-collapse: collapse; width: 100%; font-size: 14px; }
th, td { text-align: left; padding: 6px 8px; border-bottom: 1px solid var(--grid); }
td.num, th.num { text-align: right; font-family: var(--font-mono);
  font-variant-numeric: tabular-nums; }
.sample { font-family: var(--font-mono); font-size: 13px; white-space: pre-wrap; }
ul.limits li { margin-bottom: 4px; }
@media (prefers-reduced-motion: reduce) { * { transition: none !important; } }
"""

DARK = """color-scheme: dark;
    --surface: #0F1117; --surface-2: #171A21; --grid: #2A2E37;
    --text-primary: #F3F2EF; --text-secondary: #A9A7A0;
    --compare-base: #3987e5; --compare-ext: #d95926;
    --series-1: #3987e5; --series-2: #d95926; --series-3: #199e70; --series-4: #c98500;
    --series-5: #d55181; --series-6: #008300; --series-7: #9085e9; --series-8: #e66767;"""

ICONS = {
    "ok": '<svg viewBox="0 0 16 16" aria-hidden="true"><path d="M6.2 11.6 2.6 8l1.1-1.1 2.5 '
    '2.5 6.1-6.1 1.1 1.1z"/></svg>',
    "bad": '<svg viewBox="0 0 16 16" aria-hidden="true"><path d="M8 1 15 14H1zm-.8 5v4h1.6V6zm0 '
    '5v1.6h1.6V11z"/></svg>',
    "unknown": '<svg viewBox="0 0 16 16" aria-hidden="true"><path d="M8 1a7 7 0 1 0 0 14A7 7 0 0 '
    '0 8 1zm-.8 3h1.6v5H7.2zm0 6h1.6v1.6H7.2z"/></svg>',
}


def _num(value: float | None, digits: int = 2, signed: bool = False) -> str:
    if value is None:
        return "–"
    return f"{value:+.{digits}f}" if signed else f"{value:.{digits}f}"


def _lang(code: str) -> str:
    return f"{language_name(code)} ({code})"


def _table(headers: list[str], rows: list[list[str]], numeric_from: int = 1) -> str:
    head = "".join(
        f'<th scope="col"{" class=num" if i >= numeric_from else ""}>{escape(h)}</th>'
        for i, h in enumerate(headers)
    )
    body = "".join(
        "<tr>"
        + "".join(
            f"<{'th scope=row' if i == 0 else 'td'}{' class=num' if i >= numeric_from else ''}>"
            f"{escape(cell)}</{'th' if i == 0 else 'td'}>"
            for i, cell in enumerate(row)
        )
        + "</tr>"
        for row in rows
    )
    table = f"<table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>"
    return f'<div class="table-wrap">{table}</div>'


def _legend(items: list[tuple[str, str]]) -> str:
    spans = "".join(
        f'<span style="--swatch:{color}">{escape(label)}</span>' for label, color in items
    )
    return f'<div class="legend">{spans}</div>'


def english_chip(data: ReportData) -> str:
    verdict = (data.comparison or {}).get("english")
    if not verdict or verdict.get("metric") is None:
        state, text = "unknown", "English regression not measured yet"
    elif verdict["regressed"]:
        state, text = (
            "bad",
            f"English regression alert: −{verdict['drop']:.1f} ({verdict['metric']})",
        )
    else:
        state, text = (
            "ok",
            (
                f"English within budget: {-verdict['drop']:+.1f} ({verdict['metric']}, "
                f"limit {data.recipe.max_english_regression:g})"
            ),
        )
    return f'<p class="chip {state}" role="status">{ICONS[state]}{escape(text)}</p>'


def token_tax_section(data: ReportData) -> str:
    base, ext = data.fertility_base, data.fertility_extended
    if not base:
        return '<h2>Token tax</h2><p class="note">Run <code>poly fertility</code> first.</p>'
    langs = list(base)
    rows, end_labels, table_rows = [], [], []
    for lang in langs:
        values = [base[lang].tokens_per_word]
        after = ext.get(lang) if ext else None
        if after:
            values.append(after.tokens_per_word)
        rows.append((_lang(lang), values))
        change = saving(base[lang], after) if after else None
        end_labels.append(f"{change:+.0%}" if change is not None else "")
        table_rows.append(
            [
                _lang(lang),
                _num(base[lang].tokens_per_word),
                _num(after.tokens_per_word if after else None),
                f"{change:+.0%}" if change is not None else "–",
            ]
        )
    series = [("base", "var(--compare-base)")] + (
        [("extended", "var(--compare-ext)")] if ext else []
    )
    chart = grouped_bars(rows, series, end_labels, "tt-title", "tt-desc", "tokens/word")
    worst = max(langs, key=lambda c: base[c].tokens_per_word)
    desc = (
        f"Tokens per word on held-out text for {len(langs)} languages. The base tokenizer is "
        f"most expensive for {language_name(worst)} at "
        f"{fmt(base[worst].tokens_per_word)} tokens per word"
        + (
            f"; after extension it needs {fmt(ext[worst].tokens_per_word)}."
            if ext and worst in ext
            else "."
        )
    )
    return (
        "<h2>Token tax</h2>"
        '<figure><figcaption id="tt-title">'
        "Tokens per word, base vs extended tokenizer</figcaption>"
        f"{_legend([(name, color) for name, color in series])}{chart}"
        f'<p id="tt-desc" class="note">{escape(desc)}</p>'
        "<details><summary>Data table</summary>"
        f"{_table(['Language', 'Base', 'Extended', 'Saving'], table_rows)}</details></figure>"
    )


def benchmarks_section(data: ReportData) -> str:
    if not data.comparison:
        return '<h2>Benchmarks</h2><p class="note">Run <code>poly eval --all</code> first.</p>'
    rows = [
        [
            _lang(lang),
            _num(c["score_before"], 1),
            _num(c["score_after"], 1),
            _num(c["score_delta"], 1, signed=True),
            _num(c["bpc_before"], 3),
            _num(c["bpc_after"], 3),
        ]
        for lang, c in data.comparison["languages"].items()
    ]
    return (
        f"<h2>Benchmarks (base → {escape(data.comparison['adapted'])})</h2>"
        '<p class="note">Score is multiple-choice accuracy or QA F1 in points. Bits per character '
        "is held-out cross-entropy that stays comparable across tokenizers (lower is better).</p>"
        + _table(
            ["Language", "Score base", "Score adapted", "Δ pts", "BPC base", "BPC adapted"], rows
        )
    )


def curves_section(data: ReportData) -> str:
    if not data.curves:
        return ""
    langs = data.recipe.all_codes
    colors = {lang: f"var(--series-{i + 1})" for i, lang in enumerate(langs[:MAX_LINE_SERIES])}
    parts = [
        "<h2>Training monitor</h2>",
        '<p class="note">Held-out perplexity per language at every evaluation (log axis when '
        "values span 10x or more). Every checkpoint is evaluated on all languages, English "
        "included.</p>",
    ]
    for stage, records in data.curves.items():
        series = [
            Series(
                lang,
                _lang(lang),
                [(r["step"], r["heldout_ppl"][lang]) for r in records],
                colors[lang],
            )
            for lang in langs
            if lang in colors and all(lang in r["heldout_ppl"] for r in records)
        ]
        title, desc = f"{stage}-title", f"{stage}-desc"
        best = data.summaries.get(stage, {}).get("best_step")
        summary = (
            f"{len(records)} evaluations over {records[-1]['step']} steps; "
            f"best checkpoint at step {best}."
            if best is not None
            else f"{len(records)} evaluations."
        )
        table = _table(
            ["Step", *langs],
            [
                [str(r["step"]), *(_num(r["heldout_ppl"].get(lang), 2) for lang in langs)]
                for r in records
            ],
        )
        chart = line_chart(series, title, desc) if len(langs) <= MAX_LINE_SERIES else ""
        parts.append(
            f'<figure><figcaption id="{title}">'
            f"{escape(stage.upper())}: held-out perplexity</figcaption>"
            f"{_legend([(s.label, s.color) for s in series])}{chart}"
            f'<p id="{desc}" class="note">{escape(summary)}</p>'
            f"<details><summary>Data table</summary>{table}</details></figure>"
        )
    return "".join(parts)


def samples_section(data: ReportData) -> str:
    name = data.adapted
    if not name:
        return ""
    rows = [
        f"<tr><th scope=row>{escape(_lang(lang))}</th>"
        f'<td class="sample" dir="auto">{escape(res["sample"]["prompt"])} → '
        f"{escape(res['sample']['output'])}</td></tr>"
        for lang, res in data.evals[name]["languages"].items()
        if "sample" in res
    ]
    return (
        f"<h2>Sample generations ({escape(name)})</h2>"
        '<p class="note">Greedy continuations of the first held-out document. Review them with '
        "native speakers before trusting any score.</p>"
        f'<div class="table-wrap"><table><tbody>{"".join(rows)}</tbody></table></div>'
    )


def render_html(data: ReportData) -> str:
    recipe = data.recipe
    generated = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    langs = ", ".join(_lang(code) for code in recipe.codes)
    limits = "".join(f"<li>{escape(item)}</li>" for item in limitations(data))
    body = (
        f"<h1>PolyglotLM report card</h1>"
        f'<p class="meta">Base <code>{escape(recipe.base_model)}</code> · {escape(langs)} · '
        f"generated {generated}</p>"
        f"{english_chip(data)}{token_tax_section(data)}{benchmarks_section(data)}"
        f"{curves_section(data)}{samples_section(data)}"
        f'<h2>Known limitations</h2><ul class="limits">{limits}</ul>'
    )
    css = CSS.replace("DARK", DARK)
    return (
        '<!doctype html><html lang="en"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width, initial-scale=1">'
        f"<title>PolyglotLM report card</title><style>{css}</style></head>"
        f"<body><main>{body}</main></body></html>\n"
    )


# --- Model card -------------------------------------------------------------------------------


def render_model_card(data: ReportData) -> str:
    recipe = data.recipe
    lines = [
        "# PolyglotLM model card",
        "",
        f"- **Base model:** `{recipe.base_model}`",
        f"- **Target languages:** {', '.join(_lang(c) for c in recipe.codes)}",
        f"- **Method:** tokenizer extension ({recipe.embedding_init} init), continued pre-training "
        f"with LoRA r={recipe.lora_rank} + trainable embeddings, teacher-distilled SFT, "
        f"{recipe.english_replay:.0%} English replay.",
        f"- **Seed:** {recipe.seed}",
        "",
        "## Token tax (tokens per word, held-out text)",
        "",
    ]
    if data.fertility_base:
        lines += ["| Language | Base | Extended | Saving |", "|---|---:|---:|---:|"]
        for lang, row in data.fertility_base.items():
            after = (data.fertility_extended or {}).get(lang)
            change = f"{saving(row, after):+.0%}" if after else "–"
            lines.append(
                f"| {_lang(lang)} | {row.tokens_per_word:.2f} | "
                f"{_num(after.tokens_per_word if after else None)} | {change} |"
            )
    else:
        lines.append("_Not measured._")
    lines += ["", "## Evaluation", ""]
    if data.comparison:
        verdict = data.comparison["english"]
        status = "REGRESSED" if verdict["regressed"] else "within budget"
        lines.append(
            f"English: {status} (drop {_num(verdict['drop'], 1)} {verdict['metric']}, "
            f"limit {recipe.max_english_regression:g})."
        )
        lines += [
            "",
            "| Language | Score base | Score adapted | Δ | BPC base | BPC adapted |",
            "|---|---:|---:|---:|---:|---:|",
        ]
        for lang, c in data.comparison["languages"].items():
            lines.append(
                f"| {_lang(lang)} | {_num(c['score_before'], 1)} | {_num(c['score_after'], 1)} | "
                f"{_num(c['score_delta'], 1, signed=True)} | {_num(c['bpc_before'], 3)} | "
                f"{_num(c['bpc_after'], 3)} |"
            )
    else:
        lines.append("_Not evaluated yet._")
    lines += ["", "## Training data", ""]
    if data.snapshot:
        for lang, stats in data.snapshot["languages"].items():
            lines.append(
                f"- {_lang(lang)}: {stats['kept']} train / {stats['heldout']} held-out documents, "
                f"{stats['words']} words, dropped {sum(stats['dropped'].values())}"
            )
        licences = sorted({s["licence"] for s in data.snapshot["sources"]})
        lines.append(f"- Source licences: {', '.join(licences)}")
    if origin := data.sft_counts.get("origin"):
        lines.append(f"- SFT examples by language/origin: {dict(sorted(origin.items()))}")
    lines += ["", "## Known limitations", "", *(f"- {item}" for item in limitations(data)), ""]
    return "\n".join(lines)


def write_report(recipe: TrainingRecipe) -> Path:
    data = collect(recipe)
    out_dir = recipe.paths.report
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "index.html").write_text(render_html(data), encoding="utf-8")
    (out_dir / "MODEL_CARD.md").write_text(render_model_card(data), encoding="utf-8")
    if data.comparison:
        (out_dir / "comparison.json").write_text(
            json.dumps(data.comparison, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
    log.info("report_written", path=str(out_dir), english=(data.comparison or {}).get("english"))
    return out_dir / "index.html"


def english_regressed(recipe: TrainingRecipe) -> bool:
    data = collect(recipe)
    return bool((data.comparison or {}).get("english", {}).get("regressed"))
