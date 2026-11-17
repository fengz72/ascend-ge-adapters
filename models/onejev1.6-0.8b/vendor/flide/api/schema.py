"""Minimal System One request types (the subset ``flide_onejev.prompt`` imports).

Upstream: ``flide/api/schema.py`` in flide 0.1.16.  The upstream module is a
pydantic model tree; this slice keeps only the three question kinds and their
two validation rules, as plain keyword-only dataclasses, so the vendored prompt
renderer stays importable without the serving stack.  Field names and order,
the option limits, the ``Entry`` union and the ``Question`` union are unchanged
from upstream.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Union

Entry = Union[str, dict[str, Any], list[Any], None]

MAX_CHOICE_OPTIONS = 256
MAX_SCORE_LEVELS = 256


@dataclass(kw_only=True)
class NoulCriteria:
    true: Entry = None
    false: Entry = None


@dataclass(kw_only=True)
class NoulQuestion:
    type: Literal["noul"]
    instructions: Entry = None
    criteria: NoulCriteria | None = None


@dataclass(kw_only=True)
class ChoiceQuestion:
    type: Literal["choice"]
    instructions: Entry = None
    criteria: dict[str, Entry] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.criteria:
            raise ValueError("choice question needs at least one option in criteria")
        if len(self.criteria) > MAX_CHOICE_OPTIONS:
            raise ValueError(f"choice question has {len(self.criteria)} options, maximum is {MAX_CHOICE_OPTIONS}")


@dataclass(kw_only=True)
class ScoreQuestion:
    type: Literal["score"]
    instructions: Entry = None
    criteria: list[Entry] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.criteria:
            raise ValueError("score question needs at least one level in criteria")
        if len(self.criteria) > MAX_SCORE_LEVELS:
            raise ValueError(f"score question has {len(self.criteria)} levels, maximum is {MAX_SCORE_LEVELS}")


Question = Union[NoulQuestion, ChoiceQuestion, ScoreQuestion]
