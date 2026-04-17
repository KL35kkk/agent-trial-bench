"""
DeterministicTestsGrader – code-based checks on the Outcome.

Runs deterministic assertions that do not require an LLM: exact match,
substring containment, regex patterns, numeric range, multiple-choice validation
and domain-aware format checks (``url``, ``email``, ``iso_date``, ``address``,
``tx_hash``, …).  Which format checks apply is driven entirely by
``task.context`` / grader config, so the grader itself has no Web3 (or any
other vertical) hardcoded.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

from ..domains import get_validator
from ..domains.general import normalize_number
from ..trajectory import GradeResult, Outcome, Trajectory
from .base import BaseGrader


class DeterministicTestsGrader(BaseGrader):
    """Runs deterministic (code-based) tests against the ``Outcome``.

    Configuration (``self.config``):
        regex_patterns: list[str] – patterns that must match the answer.
        numeric_range:  {"min": float, "max": float}.

    Task-context keys consulted (per task):
        check_format: list[str] | str – named validators to run on the answer
            (e.g. ``["url"]``, ``["address", "tx_hash"]``).  Names are resolved
            via :func:`agent_trial_bench.domains.get_validator`.
        options: list[str] – multiple-choice options.
    """

    name = "deterministic_tests"

    # ── helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _normalize(text: str) -> str:
        return " ".join(text.lower().split())

    @classmethod
    def _exact_match(cls, expected: str, actual: str) -> bool:
        return cls._normalize(expected) == cls._normalize(actual)

    @classmethod
    def _contains(cls, expected: str, actual: str) -> bool:
        return cls._normalize(expected) in cls._normalize(actual)

    @staticmethod
    def _coerce_format_list(value: Any) -> List[str]:
        if value is None:
            return []
        if isinstance(value, str):
            return [value]
        if isinstance(value, (list, tuple, set)):
            return [str(v) for v in value]
        return []

    # ── grading ───────────────────────────────────────────────────────────────

    async def grade(
        self,
        outcome: Outcome,
        trajectory: Optional[Trajectory] = None,
        task: Optional[Any] = None,
    ) -> GradeResult:
        assertions: List[Dict[str, Any]] = []
        expected = (task.expected_answer if task else None) or ""
        answer = outcome.answer
        context = (task.context if task else {}) or {}

        # 1. Exact match
        if expected:
            assertions.append(
                {"name": "exact_match", "passed": self._exact_match(expected, answer)}
            )

        # 2. Substring containment
        if expected:
            assertions.append(
                {"name": "contains_answer", "passed": self._contains(expected, answer)}
            )

        # 3. Regex patterns
        for pattern in self.config.get("regex_patterns", []):
            match = bool(re.search(pattern, answer))
            assertions.append({"name": f"regex:{pattern[:30]}", "passed": match})

        # 4. Numeric range check
        num_range = self.config.get("numeric_range")
        if num_range:
            parsed = normalize_number(answer)
            in_range = (
                parsed is not None
                and num_range.get("min", float("-inf")) <= parsed <= num_range.get("max", float("inf"))
            )
            assertions.append({"name": "numeric_range", "passed": in_range})

        # 5. Domain-agnostic format checks.
        # Dataset authors declare which validators apply via ``context["check_format"]``.
        requested_formats = self._coerce_format_list(context.get("check_format"))

        # Back-compat: older Web3 datasets used boolean flags.  Translate them.
        if context.get("check_address"):
            requested_formats.append("address")
        if context.get("check_tx_hash"):
            requested_formats.append("tx_hash")
        if context.get("check_url"):
            requested_formats.append("url")
        if context.get("check_email"):
            requested_formats.append("email")

        for fmt in requested_formats:
            validator = get_validator(fmt)
            if validator is None:
                assertions.append(
                    {"name": f"format:{fmt}", "passed": False, "detail": "unknown validator"}
                )
                continue
            # For ``address`` tolerate ENS as an equivalent shape by also
            # trying the ENS validator when present.
            passed = validator(answer)
            if not passed and fmt in ("address", "eth_address"):
                ens_validator = get_validator("ens")
                if ens_validator:
                    passed = ens_validator(answer)
            assertions.append({"name": f"format:{fmt}", "passed": passed})

        # 6. Multiple-choice option check
        mc_options = context.get("options") or task_options(task)
        if mc_options:
            answer_clean = answer.strip().upper()
            assertions.append(
                {
                    "name": "valid_mc_choice",
                    "passed": answer_clean in [str(o).upper() for o in mc_options],
                }
            )

        if not assertions:
            return GradeResult(
                grader_name=self.name,
                score=0.5,
                passed=True,
                reasoning="No deterministic assertions defined for this task.",
            )

        passed_count = sum(1 for a in assertions if a["passed"])
        score = passed_count / len(assertions)
        passed = score >= 0.5

        return GradeResult(
            grader_name=self.name,
            score=round(score, 4),
            passed=passed,
            reasoning=f"Passed {passed_count}/{len(assertions)} deterministic assertions.",
            assertions=assertions,
        )

    def get_assertions(self) -> List[str]:
        base = ["exact_match", "contains_answer"]
        for p in self.config.get("regex_patterns", []):
            base.append(f"regex:{p[:30]}")
        return base


# ── helpers ───────────────────────────────────────────────────────────────────


def task_options(task: Optional[Any]) -> List[str]:
    """Extract multiple-choice options from a task object if present."""
    if task is None:
        return []
    ctx = getattr(task, "context", {}) or {}
    return ctx.get("options", [])
