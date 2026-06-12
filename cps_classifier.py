"""Single-call LLM classifier for CPS skill coding of chat messages.

Implements the design inspired by CAP4CPS (Zhu et al., 2026):
for each target message, extract a cognitive context (the speaker's own
recent messages) and a social context (teammates' recent messages),
assemble one prompt with the codebook and few-shot examples, and make a
single LLM call that returns a JSON label + rationale.

Provider-agnostic: pass any object implementing LLMClient.complete().

Usage (library):
    from cps_classifier import CPSClassifier, Message, load_codebook
    clf = CPSClassifier(client, load_codebook("codebook_andrews_todd.json"))
    results = clf.classify_dialogue(messages)

Usage (CLI):
    python cps_classifier.py input.csv output.csv --provider anthropic
    # input.csv columns: speaker,text
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol


# ---------------------------------------------------------------- data model

@dataclass
class Message:
    """One chat message: who said it and what they said."""

    speaker: str
    text: str

@dataclass
class Result:
    """Classification outcome for the message at `index`.

    `votes` maps label -> count across self-consistency samples
    (a single entry when n_samples == 1).
    """

    index: int
    label: str
    dimension: str | None
    rationale: str
    raw_response: str
    votes: dict[str, int] = field(default_factory=dict)


def load_codebook(path: str | Path) -> dict:
    """Load a codebook JSON file (see codebook_andrews_todd.json for the schema)."""
    with open(path, encoding="utf-8") as f:
        return json.load(f)


# --------------------------------------------------------- context extraction

def extract_contexts(
    messages: list[Message], t: int, w_cognitive: int = 2, w_social: int = 1
) -> tuple[list[Message], list[Message]]:
    """Return (cognitive, social) context windows for the message at index t.

    Cognitive = the target speaker's own most recent prior messages (w_cognitive).
    Social    = other participants' most recent prior messages (w_social).
    Mirrors the CE module of CAP4CPS; best windows in the paper were 2 and 1.
    """
    speaker = messages[t].speaker
    cognitive = [m for m in messages[:t] if m.speaker == speaker][-w_cognitive:]
    social = [m for m in messages[:t] if m.speaker != speaker][-w_social:]
    return cognitive, social


# ------------------------------------------------------------ prompt assembly

OUTPUT_SCHEMA = (
    '{"dimension": "<cognitive|social>", '
    '"label": "<one code from the codebook>", '
    '"rationale": "<one sentence>"}'
)


def build_system_prompt(codebook: dict) -> str:
    """Render the codebook into a system prompt: code definitions, few-shot
    examples, and the JSON output instructions."""
    lines = [
        "You are an expert coder of Collaborative Problem Solving (CPS) chat data.",
        "You classify a single target chat message into exactly one CPS subskill "
        f"from the {codebook['framework']}.",
        "",
        "## Codebook",
    ]
    for code, entry in codebook["codes"].items():
        lines.append(f"- {code} ({entry['name']}, {entry['dimension']}): {entry['definition']}")

    lines += ["", "## Examples"]
    for code, entry in codebook["codes"].items():
        for ex in entry.get("examples", []):
            lines.append(f'- "{ex}" -> {code}')

    lines += [
        "",
        "## Instructions",
        "1. Read the cognitive context (the speaker's own recent messages) and the "
        "social context (teammates' recent messages). The same message can mean "
        "different things in different contexts.",
        "2. First decide whether the target message primarily serves an individual "
        "problem-solving (cognitive) or interactional (social) function.",
        "3. Then pick exactly one code. Classify ONLY the target message - never the "
        "context messages.",
        f"4. Respond with only this JSON, nothing else: {OUTPUT_SCHEMA}",
    ]
    return "\n".join(lines)


def build_user_prompt(
    cognitive: list[Message], social: list[Message], target: Message
) -> str:
    """Assemble the per-message user prompt: both context blocks (marked
    reference-only) followed by the target message to classify."""
    def block(title: str, msgs: list[Message]) -> str:
        if not msgs:
            return f"## {title}\n(none)"
        body = "\n".join(f"{m.speaker}: {m.text}" for m in msgs)
        return f"## {title} (for reference only - do not classify)\n{body}"

    return "\n\n".join(
        [
            block("Cognitive context - target speaker's recent messages", cognitive),
            block("Social context - teammates' recent messages", social),
            f"## Target message (classify this one)\n{target.speaker}: {target.text}",
        ]
    )


# ------------------------------------------------------------------ providers

class LLMClient(Protocol):
    """Minimal provider interface: one completion from a system + user prompt."""

    def complete(self, system: str, user: str) -> str: ...


class AnthropicClient:
    """Anthropic Messages API backend; reads ANTHROPIC_API_KEY from the environment."""

    def __init__(self, model: str = "claude-sonnet-4-6", temperature: float = 0.0):
        import anthropic

        self._client = anthropic.Anthropic()  # uses ANTHROPIC_API_KEY
        self.model = model
        self.temperature = temperature

    def complete(self, system: str, user: str) -> str:
        resp = self._client.messages.create(
            model=self.model,
            max_tokens=300,
            temperature=self.temperature,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        return resp.content[0].text


class OpenAIClient:
    """OpenAI Chat Completions backend; reads OPENAI_API_KEY from the environment."""

    def __init__(self, model: str = "gpt-4o", temperature: float = 0.0):
        import openai

        self._client = openai.OpenAI()  # uses OPENAI_API_KEY
        self.model = model
        self.temperature = temperature

    def complete(self, system: str, user: str) -> str:
        resp = self._client.chat.completions.create(
            model=self.model,
            temperature=self.temperature,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
        return resp.choices[0].message.content


class MockClient:
    """Offline client for testing the pipeline. Trivial keyword heuristic."""

    RULES = [
        (r"\?", "SESU"),
        (r"\b(changed|set|got|adjusted)\b", "CEC"),
        (r"\b(volts?|ohms?|resistor|value)\b", "SSI"),
        (r"\b(try|plan|let's|should)\b", "CP"),
        (r"\b(agree|think so|yes|no|disagree)\b", "SN"),
    ]

    def complete(self, system: str, user: str) -> str:
        target = user.rsplit("## Target message", 1)[-1].lower()
        label, dim = "SMC", "social"
        for pat, code in self.RULES:
            if re.search(pat, target):
                label = code
                dim = "cognitive" if code.startswith("C") else "social"
                break
        return json.dumps(
            {"dimension": dim, "label": label, "rationale": "mock heuristic"}
        )


def make_client(provider: str, model: str | None = None) -> LLMClient:
    """Build an LLMClient by provider name: "anthropic", "openai", or "mock"."""
    if provider == "anthropic":
        return AnthropicClient(**({"model": model} if model else {}))
    if provider == "openai":
        return OpenAIClient(**({"model": model} if model else {}))
    if provider == "mock":
        return MockClient()
    raise ValueError(f"Unknown provider: {provider}")


# ----------------------------------------------------------------- classifier

def _parse_json(text: str) -> dict:
    """Extract and parse the first JSON object in `text` (tolerates prose around it)."""
    match = re.search(r"\{.*\}", text, re.S)
    if not match:
        raise ValueError(f"No JSON object in response: {text[:200]!r}")
    return json.loads(match.group(0))


class CPSClassifier:
    """Codes chat messages into CPS subskills, one LLM call per message.

    Each call sees the codebook (system prompt), the speaker's own recent
    messages (cognitive context), and teammates' recent messages (social
    context). Set n_samples > 1 for self-consistency: the message is
    classified that many times and the majority label wins.
    """

    def __init__(
        self,
        client: LLMClient,
        codebook: dict,
        w_cognitive: int = 2,
        w_social: int = 1,
        n_samples: int = 1,
        max_retries: int = 3,
        retry_wait: float = 2.0,
    ):
        self.client = client
        self.codebook = codebook
        self.w_cognitive = w_cognitive
        self.w_social = w_social
        self.n_samples = n_samples  # >1 enables self-consistency majority vote
        self.max_retries = max_retries
        self.retry_wait = retry_wait
        self.system_prompt = build_system_prompt(codebook)
        self.valid_labels = set(codebook["codes"])

    def _call_once(self, user_prompt: str) -> dict:
        """One LLM call returning the parsed, label-validated response;
        retries with linear backoff on parse or validation failure."""
        last_err: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                raw = self.client.complete(self.system_prompt, user_prompt)
                parsed = _parse_json(raw)
                label = str(parsed.get("label", "")).strip().upper()
                if label not in self.valid_labels:
                    raise ValueError(f"Label {label!r} not in codebook")
                parsed["label"] = label
                parsed["_raw"] = raw
                return parsed
            except Exception as err:  # noqa: BLE001 - retry on any failure
                last_err = err
                time.sleep(self.retry_wait * (attempt + 1))
        raise RuntimeError(f"Classification failed after retries: {last_err}")

    def classify(self, messages: list[Message], t: int) -> Result:
        """Classify the message at index t, using the prior dialogue for context."""
        cognitive, social = extract_contexts(
            messages, t, self.w_cognitive, self.w_social
        )
        user_prompt = build_user_prompt(cognitive, social, messages[t])

        samples = [self._call_once(user_prompt) for _ in range(self.n_samples)]
        votes = Counter(s["label"] for s in samples)
        label = votes.most_common(1)[0][0]
        winner = next(s for s in samples if s["label"] == label)

        return Result(
            index=t,
            label=label,
            dimension=self.codebook["codes"][label]["dimension"],
            rationale=str(winner.get("rationale", "")),
            raw_response=winner["_raw"],
            votes=dict(votes),
        )

    def classify_dialogue(
        self, messages: list[Message], verbose: bool = False
    ) -> list[Result]:
        """Classify every message in order; returns one Result per message."""
        results = []
        for t in range(len(messages)):
            res = self.classify(messages, t)
            results.append(res)
            if verbose:
                print(f"[{t}] {messages[t].speaker}: {messages[t].text!r} -> "
                      f"{res.label} ({res.rationale})", file=sys.stderr)
        return results


# ------------------------------------------------------------------------ CLI

def main() -> None:
    """CLI entry point: code a CSV of chat messages (columns: speaker,text)."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("input_csv", help="CSV with columns: speaker,text")
    ap.add_argument("output_csv")
    ap.add_argument("--provider", default="anthropic",
                    choices=["anthropic", "openai", "mock"])
    ap.add_argument("--model", default=None)
    ap.add_argument("--codebook", default=str(
        Path(__file__).parent / "codebook_andrews_todd.json"))
    ap.add_argument("--w-cognitive", type=int, default=2)
    ap.add_argument("--w-social", type=int, default=1)
    ap.add_argument("--n-samples", type=int, default=1,
                    help=">1 enables self-consistency majority voting")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    with open(args.input_csv, encoding="utf-8") as f:
        messages = [Message(r["speaker"], r["text"]) for r in csv.DictReader(f)]

    clf = CPSClassifier(
        client=make_client(args.provider, args.model),
        codebook=load_codebook(args.codebook),
        w_cognitive=args.w_cognitive,
        w_social=args.w_social,
        n_samples=args.n_samples,
    )
    results = clf.classify_dialogue(messages, verbose=args.verbose)

    with open(args.output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["index", "speaker", "text", "label", "dimension",
                         "rationale", "votes"])
        for msg, res in zip(messages, results):
            writer.writerow([res.index, msg.speaker, msg.text, res.label,
                             res.dimension, res.rationale, json.dumps(res.votes)])

    print(f"Wrote {len(results)} coded messages to {args.output_csv}")


if __name__ == "__main__":
    main()
