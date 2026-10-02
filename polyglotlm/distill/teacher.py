"""Teacher models that translate and generate instruction data.

`ClaudeTeacher` calls the Claude API with typed structured outputs. Before training a model on
any teacher's outputs, check that provider's terms of service: some providers, Anthropic
included, restrict using outputs to develop competing models.

`DemoTeacher` is an offline stand-in that "translates" by looking sentences up in the synthetic
parallel demo corpus, so the whole pipeline can run in tests and on a laptop without a key.
"""

from __future__ import annotations

import os
import random
from typing import Any, Protocol, TypeVar

from pydantic import BaseModel, ConfigDict, Field

from polyglotlm.config import language_name
from polyglotlm.logs import get_logger

log = get_logger(__name__)


class Pair(BaseModel):
    model_config = ConfigDict(extra="forbid")

    instruction: str
    response: str


class PairList(BaseModel):
    model_config = ConfigDict(extra="forbid")

    pairs: list[Pair] = Field(min_length=1)


class TeacherError(RuntimeError):
    """The teacher could not produce a usable answer (refusal, truncation, unknown input)."""


class Teacher(Protocol):
    name: str

    def translate(self, pair: Pair, lang: str) -> Pair: ...

    def generate_native(self, lang: str, topic: str, n: int, seed: int) -> list[Pair]: ...

    def back_translate(self, pair: Pair, lang: str) -> Pair: ...


SYSTEM_PROMPT = """\
You create training data for an assistant that serves speakers of {language}. Write the way an \
educated native speaker writes everyday {language}: natural word order and idiom in the standard \
script, never word-for-word calques of English. Keep numbers, names, code and URLs unchanged. \
Do not add facts that the source does not contain; when a fact depends on region or date, say \
so rather than guessing."""

TRANSLATE_PROMPT = """\
Translate this instruction and its response into {language}. Preserve meaning, tone and \
formatting (lists, line breaks, code blocks). Adapt an example or unit only if the literal \
version would confuse a {language} reader.

<instruction>
{instruction}
</instruction>
<response>
{response}
</response>"""

NATIVE_PROMPT = """\
Write {n} different instruction/response pairs in {language} about: {topic}.
Instructions should be requests a real {language} speaker might make of an assistant: \
questions, how-to requests, explanations, short writing tasks. Ground them in everyday life \
where {language} is spoken. Responses must be accurate, helpful and self-contained."""

BACK_TRANSLATE_PROMPT = """\
Translate this {language} instruction and response into English, as literally as fluent \
English allows. Do not correct, improve or complete them.

<instruction>
{instruction}
</instruction>
<response>
{response}
</response>"""

BACK_TRANSLATE_SYSTEM = "You are a careful translator into English."

T = TypeVar("T", bound=BaseModel)


class ClaudeTeacher:
    """Claude via the Anthropic SDK (`pip install polyglotlm[teacher]`).

    Reads TEACHER_API_KEY; without it the SDK's default credential chain applies
    (ANTHROPIC_API_KEY, an `ant auth login` profile, ...). Server-side refusal fallbacks are on,
    so a declined request is retried on Anthropic's recommended fallback model.
    """

    FALLBACK_BETA = "server-side-fallback-2026-07-01"

    def __init__(self, model: str, effort: str = "high", max_tokens: int = 16000) -> None:
        try:
            import anthropic
        except ImportError as exc:
            raise RuntimeError(
                "the Claude teacher needs `pip install polyglotlm[teacher]`"
            ) from exc
        api_key = os.environ.get("TEACHER_API_KEY")
        self._client: Any = (
            anthropic.Anthropic(api_key=api_key) if api_key else anthropic.Anthropic()
        )
        self.name = model
        self.effort = effort
        self.max_tokens = max_tokens

    def _ask(self, system: str, prompt: str, schema: type[T]) -> T:
        response = self._client.beta.messages.parse(
            model=self.name,
            max_tokens=self.max_tokens,
            system=system,
            messages=[{"role": "user", "content": prompt}],
            output_format=schema,
            output_config={"effort": self.effort},
            betas=[self.FALLBACK_BETA],
            fallbacks="default",
        )
        if response.stop_reason == "refusal":
            category = getattr(response.stop_details, "category", None)
            raise TeacherError(f"teacher declined the request (category={category})")
        if response.stop_reason == "max_tokens":
            raise TeacherError("teacher output was truncated at max_tokens")
        if response.parsed_output is None:
            raise TeacherError("teacher returned no parseable output")
        return response.parsed_output

    def translate(self, pair: Pair, lang: str) -> Pair:
        language = language_name(lang)
        prompt = TRANSLATE_PROMPT.format(language=language, **pair.model_dump())
        return self._ask(SYSTEM_PROMPT.format(language=language), prompt, Pair)

    def generate_native(self, lang: str, topic: str, n: int, seed: int) -> list[Pair]:
        language = language_name(lang)
        prompt = NATIVE_PROMPT.format(language=language, topic=topic, n=n)
        return self._ask(SYSTEM_PROMPT.format(language=language), prompt, PairList).pairs[:n]

    def back_translate(self, pair: Pair, lang: str) -> Pair:
        prompt = BACK_TRANSLATE_PROMPT.format(language=language_name(lang), **pair.model_dump())
        return self._ask(BACK_TRANSLATE_SYSTEM, prompt, Pair)


class DemoTeacher:
    """Offline teacher over the parallel demo corpus. Only knows the demo sentences."""

    name = "demo"

    def __init__(self) -> None:
        from polyglotlm.demo.corpus import parallel_sentences

        self._rows = parallel_sentences()
        self._from_en = {row["en"]: row for row in self._rows}
        self._to_en = {text: row["en"] for row in self._rows for text in row.values()}

    def _lookup(self, text: str, table: dict[str, Any]) -> Any:
        if text not in table:
            raise TeacherError(f"demo teacher does not know {text[:40]!r}")
        return table[text]

    def translate(self, pair: Pair, lang: str) -> Pair:
        return Pair(
            instruction=self._lookup(pair.instruction, self._from_en)[lang],
            response=self._lookup(pair.response, self._from_en)[lang],
        )

    def generate_native(self, lang: str, topic: str, n: int, seed: int) -> list[Pair]:
        rng = random.Random(f"{seed}:{lang}:{topic}")
        picks = [rng.sample(self._rows, 2) for _ in range(n)]
        return [Pair(instruction=a[lang], response=b[lang]) for a, b in picks]

    def back_translate(self, pair: Pair, lang: str) -> Pair:
        return Pair(
            instruction=self._lookup(pair.instruction, self._to_en),
            response=self._lookup(pair.response, self._to_en),
        )


def build_teacher(name: str, effort: str = "high") -> Teacher:
    if name == "demo":
        return DemoTeacher()
    log.info("teacher", model=name, effort=effort)
    return ClaudeTeacher(name, effort)
