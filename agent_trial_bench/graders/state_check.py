"""
StateCheckGrader – validates the final environment state in the Outcome.

Checks source reliability (against the union of trusted domains contributed by
every registered ``DomainPlugin``), state ↔ answer consistency, and any
domain-declared format checks on state values.  Optional ``required_state_keys``
and ``state_format_checks`` can be supplied via grader config or
``task.context``.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Set

from ..domains import all_trusted_sources, get_validator
from ..trajectory import GradeResult, Outcome, Trajectory
from .base import BaseGrader


_DOMAIN_RE = re.compile(r"(?:https?://)?(?:www\.)?([^/\s]+)")


def _extract_domain(url: str) -> str:
    m = _DOMAIN_RE.match(url.strip())
    return m.group(1).lower() if m else url.lower()


def _is_trusted(url: str, trusted: Set[str]) -> bool:
    domain = _extract_domain(url)
    return any(domain == d or domain.endswith("." + d) for d in trusted)


class StateCheckGrader(BaseGrader):
    """Checks the final state captured in the ``Outcome``.

    Configuration (``self.config``):
        trusted_sources: Iterable[str] – additional allowlist entries, merged
            with the union of all registered domains' ``trusted_sources``.
        required_state_keys: list[str] – keys that must be present in state.
        state_format_checks: dict[str, str] – map ``state_key -> format_name``
            (e.g. ``{"contract_address": "address", "tx": "tx_hash"}``).  The
            format name is resolved against the domain registry.
        source_trust_threshold: float – min fraction of trusted sources
            (default 0.5).
        pass_threshold: float – min score to count as passed (default 0.6).
    """

    name = "state_check"

    async def grade(
        self,
        outcome: Outcome,
        trajectory: Optional[Trajectory] = None,
        task: Optional[Any] = None,
    ) -> GradeResult:
        assertions: List[Dict[str, Any]] = []
        ctx = (task.context if task else {}) or {}

        # ── 1. Source reliability ────────────────────────────────────────
        if outcome.sources:
            trusted = set(all_trusted_sources()) | set(
                self.config.get("trusted_sources") or ctx.get("trusted_sources") or []
            )
            trusted_count = sum(1 for s in outcome.sources if _is_trusted(s, trusted))
            threshold = float(self.config.get("source_trust_threshold", 0.5))
            reliable = (trusted_count / len(outcome.sources)) >= threshold
            assertions.append(
                {
                    "name": "reliable_sources",
                    "passed": reliable,
                    "detail": {
                        "trusted": trusted_count,
                        "total": len(outcome.sources),
                        "threshold": threshold,
                    },
                }
            )

        # ── 2. State ↔ answer consistency ───────────────────────────────
        if outcome.state and outcome.answer:
            consistent = self._check_state_consistency(outcome.state, outcome.answer)
            assertions.append({"name": "state_consistency", "passed": consistent})

        # ── 3. Declarative state format checks ──────────────────────────
        state_checks: Dict[str, str] = dict(self.config.get("state_format_checks") or {})
        state_checks.update(ctx.get("state_format_checks") or {})

        if isinstance(outcome.state, dict):
            for state_key, fmt in state_checks.items():
                value = outcome.state.get(state_key)
                if value is None:
                    # Only assert when the key is present – missingness is
                    # handled by ``required_state_keys`` below.
                    continue
                validator = get_validator(fmt)
                if validator is None:
                    assertions.append(
                        {"name": f"state_format:{state_key}", "passed": False,
                         "detail": f"unknown validator '{fmt}'"}
                    )
                    continue
                passed = isinstance(value, str) and validator(value)
                assertions.append(
                    {"name": f"state_format:{state_key}({fmt})", "passed": passed}
                )

        # ── 4. Required state keys ──────────────────────────────────────
        required_keys = (
            self.config.get("required_state_keys")
            or ctx.get("required_state_keys")
            or []
        )
        for key in required_keys:
            assertions.append(
                {
                    "name": f"state_has_{key}",
                    "passed": key in (outcome.state or {}),
                }
            )

        if not assertions:
            return GradeResult(
                grader_name=self.name,
                score=0.5,
                passed=True,
                reasoning="No state to check – returning neutral score.",
            )

        passed_count = sum(1 for a in assertions if a["passed"])
        score = passed_count / len(assertions)
        pass_threshold = float(self.config.get("pass_threshold", 0.6))

        return GradeResult(
            grader_name=self.name,
            score=round(score, 4),
            passed=score >= pass_threshold,
            reasoning=f"State check: {passed_count}/{len(assertions)} assertions passed.",
            assertions=assertions,
        )

    # ── helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _check_state_consistency(state: Dict[str, Any], answer: str) -> bool:
        """Light heuristic: flag blatant contradictions between state and answer."""
        answer_lower = answer.lower()
        for _key, value in state.items():
            if not isinstance(value, str):
                continue
            if value.lower() in answer_lower:
                return True
            if f"not {value.lower()}" in answer_lower or f"no {value.lower()}" in answer_lower:
                return False
        return True
