"""Optional LLM fallback for notes nothing else can categorise.

This is the last gate in the cascade, and it is deliberately *optional*. With no
API key the app runs end to end -- novel notes land in `unclear` and are surfaced
on the dashboard rather than silently absorbed into an existing category. That
keeps requirement 4 honest: loading next week's export needs no key, no network
and no developer.

Two properties matter more than accuracy here:

  Caching   Every answer is written to `labels/llm_cache.json`, keyed on the
            normalised note. The same note is never paid for twice, and
            `note_classifications.csv` is reproducible -- they said they would
            read a sample of it, so it must not change between runs.

  Honesty   The model may answer `other` and propose a name. A new kind of
            problem appearing at a site is exactly what a contract manager needs
            to see; forcing it into an existing bucket hides it.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

from . import config as cfg

CACHE_PATH = cfg.REPO_ROOT / "labels" / "llm_cache.json"

# Haiku is the cheapest current model and this is a short classification task.
# Costs about ten cents to label every distinct note in the corpus from scratch;
# in normal operation it is only called for notes nothing else could handle.
DEFAULT_MODEL = "claude-haiku-4-5"
BATCH_SIZE = 25

# A result carrying one of these is a failure, not a judgement. Never cached,
# so a transient outage is retried rather than frozen in as a real answer.
FAILURE_SOURCES = {"llm_error", "llm_unparsed", "no_llm"}

SYSTEM_PROMPT = """\
You categorise short notes typed by shift supervisors at South African \
facilities-management sites. Notes are terse and may mix English, isiZulu and \
Afrikaans, with frequent typos.

Choose exactly one category per note:

- nothing_reported: the shift was normal ("ok", "ntr", "akukho lutho")
- client_requested: the client asked for the extra hours and approved them
- disputed_client_request: claims client approval BUT names a real operational
  cause (e.g. "client signed but real reason is relief no show")
- relief_no_show: the incoming shift never arrived, so this person stayed on
- colleague_absent: a named colleague did not come in, so this person covered
- late_handover: the shift ran over because handover dragged (keys, paperwork)
- equipment_failure: something broke and the work took longer
- unclear: there is text but it does not explain the hours
- other: a real reason that none of the above covers

Use `other` when the note describes a genuine cause outside the list -- traffic, \
load shedding, weather, protests. Do not force it into a category that nearly \
fits. When you use `other`, suggest a short snake_case name for it.

Reply with one JSON object per line, nothing else:
{"id": <number>, "category": "<category>", "suggested": "<name or null>"}"""


@dataclass
class LLMResult:
    category: str
    suggested_name: str | None = None
    source: str = "llm"


class LLMLabeller(Protocol):
    """Anything that can label a batch of notes."""

    available: bool

    def label(self, notes: list[str]) -> list[LLMResult]:
        ...


class NullLabeller:
    """What runs when there is no API key. Everything comes back `unclear`."""

    available = False

    def label(self, notes: list[str]) -> list[LLMResult]:
        return [LLMResult("unclear", source="no_llm") for _ in notes]


@dataclass
class CachedLabeller:
    """Wraps a labeller with an on-disk cache.

    The cache is what makes the output reproducible and the cost bounded. It is
    committed to the repo, so a reviewer sees exactly what the model decided.
    """

    inner: LLMLabeller
    path: Path = CACHE_PATH
    cache: dict[str, dict] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.path.exists():
            try:
                self.cache = json.loads(self.path.read_text())
            except json.JSONDecodeError:
                self.cache = {}

    @property
    def available(self) -> bool:
        return self.inner.available

    def label(self, notes: list[str]) -> list[LLMResult]:
        unknown = [n for n in notes if n and n not in self.cache]
        fresh_results: dict[str, LLMResult] = {}

        if unknown and self.inner.available:
            for note, result in zip(unknown, self.inner.label(unknown)):
                fresh_results[note] = result
                # Only cache a real answer. Caching a timeout or a parse failure
                # would freeze it permanently and make a transient outage look
                # like a considered `unclear` -- which is exactly how a retired
                # model ID went unnoticed while every note came back `unclear`.
                if result.source not in FAILURE_SOURCES:
                    self.cache[note] = {
                        "category": result.category,
                        "suggested": result.suggested_name,
                    }
            self.save()

        out = []
        for note in notes:
            if note in fresh_results:
                out.append(fresh_results[note])       # keeps the real provenance
            elif note in self.cache:
                hit = self.cache[note]
                out.append(LLMResult(hit["category"], hit.get("suggested"), "llm_cache"))
            else:
                out.append(LLMResult("unclear", source="no_llm"))
        return out

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self.cache, indent=2, sort_keys=True))


class AnthropicLabeller:
    """Claude via the Anthropic SDK.

    Set ANTHROPIC_API_KEY to enable. Absent key or absent SDK means
    `available` is False and the cascade stops one gate earlier -- no crash.
    """

    def __init__(self, model: str = DEFAULT_MODEL, api_key: str | None = None) -> None:
        self.model = model
        self._client = None
        key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not key:
            return
        try:
            import anthropic
        except ImportError:
            return
        self._client = anthropic.Anthropic(api_key=key)

    @property
    def available(self) -> bool:
        return self._client is not None

    def label(self, notes: list[str]) -> list[LLMResult]:
        if not self.available:
            return [LLMResult("unclear", source="no_llm") for _ in notes]

        results: list[LLMResult] = []
        for start in range(0, len(notes), BATCH_SIZE):
            batch = notes[start:start + BATCH_SIZE]
            results.extend(self._label_batch(batch))
        return results

    def _label_batch(self, batch: list[str]) -> list[LLMResult]:
        listing = "\n".join(f"{i}. {note}" for i, note in enumerate(batch))
        try:
            response = self._client.messages.create(
                model=self.model,
                max_tokens=2048,
                system=SYSTEM_PROMPT,
                messages=[{"role": "user", "content": listing}],
            )
        except Exception:
            # A failed call must never break an upload. Fall back to `unclear`,
            # which the dashboard already surfaces for review.
            return [LLMResult("unclear", source="llm_error") for _ in batch]

        text = "".join(b.text for b in response.content if b.type == "text")
        return _parse_response(text, len(batch))


def _parse_response(text: str, expected: int) -> list[LLMResult]:
    """Read one JSON object per line, tolerating stray prose around it."""
    out = [LLMResult("unclear", source="llm_unparsed") for _ in range(expected)]
    for match in re.finditer(r"\{[^{}]*\}", text):
        try:
            row = json.loads(match.group())
        except json.JSONDecodeError:
            continue
        index = row.get("id")
        if not isinstance(index, int) or not 0 <= index < expected:
            continue
        category = str(row.get("category", "unclear")).strip()
        suggested = row.get("suggested")
        out[index] = LLMResult(
            category=category,
            suggested_name=str(suggested) if suggested and suggested != "null" else None,
        )
    return out


class GeminiLabeller:
    """Google AI Studio (Gemini), over the REST API.

    Raw HTTP on purpose: it needs no extra dependency, so a reviewer who clones
    the repo and runs `pip install -r requirements.txt` gets a working fallback
    the moment they set a key. Google AI Studio has a genuine free tier, which
    matters -- the brief says not to spend money on this.
    """

    ENDPOINT = (
        "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
    )
    ENV_KEYS = ("GEMINI_API_KEY", "Gemini_API_KEY", "GOOGLE_API_KEY")

    # Tried in order. Google retires model IDs for new keys (gemini-2.5-flash
    # already 404s with "no longer available to new users") and returns 503 when
    # a model is busy, so a single hardcoded ID is a liability in something that
    # has to keep working months from now.
    MODELS = ("gemini-3.6-flash", "gemini-3.1-flash-lite", "gemini-flash-latest")

    def __init__(self, model: str | None = None, api_key: str | None = None) -> None:
        self.models = (model,) if model else self.MODELS
        self.api_key = api_key or _first_env(self.ENV_KEYS)

    @property
    def available(self) -> bool:
        return bool(self.api_key)

    def label(self, notes: list[str]) -> list[LLMResult]:
        if not self.available:
            return [LLMResult("unclear", source="no_llm") for _ in notes]
        results: list[LLMResult] = []
        for start in range(0, len(notes), BATCH_SIZE):
            results.extend(self._label_batch(notes[start:start + BATCH_SIZE]))
        return results

    def _label_batch(self, batch: list[str]) -> list[LLMResult]:
        listing = "\n".join(f"{i}. {note}" for i, note in enumerate(batch))
        payload = {
            "systemInstruction": {"parts": [{"text": SYSTEM_PROMPT}]},
            "contents": [{"role": "user", "parts": [{"text": listing}]}],
            "generationConfig": {"temperature": 0, "maxOutputTokens": 8192},
        }

        for model in self.models:
            body = self._post(model, payload)
            if body is None:
                continue  # retired ID, rate limit, outage -- try the next one
            try:
                text = "".join(
                    part.get("text", "")
                    for part in body["candidates"][0]["content"]["parts"]
                )
            except (KeyError, IndexError):
                continue
            return _parse_response(text, len(batch))

        # Every model failed. Never break an upload over it -- `unclear` is
        # surfaced on the dashboard for a human to look at, and is not cached.
        return [LLMResult("unclear", source="llm_error") for _ in batch]

    def _post(self, model: str, payload: dict) -> dict | None:
        import urllib.request

        request = urllib.request.Request(
            self.ENDPOINT.format(model=model),
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json", "x-goog-api-key": self.api_key},
        )
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                return json.load(response)
        except Exception:
            return None


def _first_env(names: tuple[str, ...]) -> str | None:
    for name in names:
        value = os.environ.get(name)
        if value:
            return value.strip()
    return None


def load_dotenv(path: Path | None = None) -> None:
    """Read `.env` into the environment if present. Never overwrites a real var."""
    path = path or cfg.REPO_ROOT / ".env"
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip())


def build_labeller(enabled: bool = True) -> CachedLabeller:
    """The labeller the pipeline uses.

    Picks whichever provider has a key: Gemini first (it has a real free tier),
    then Anthropic. With neither, `NullLabeller` means the cascade stops one gate
    earlier and unplaced notes surface as `unclear` -- which is why requirement 4
    holds without any key at all.

    Always wrapped in the cache, so `note_classifications.csv` is reproducible.
    """
    inner: LLMLabeller = NullLabeller()
    if enabled:
        load_dotenv()
        for candidate in (GeminiLabeller(), AnthropicLabeller()):
            if candidate.available:
                inner = candidate
                break
    return CachedLabeller(inner=inner)
