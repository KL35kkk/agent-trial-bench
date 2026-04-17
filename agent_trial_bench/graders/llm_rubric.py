"""
LLMRubricGrader – model-based scoring against a rubric.

Wraps the existing LLMEvaluator so it conforms to the BaseGrader
interface and can evaluate both Outcome and Trajectory.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, Optional

from ..trajectory import GradeResult, Outcome, Trajectory
from .base import BaseGrader

logger = logging.getLogger(__name__)

_DEFAULT_RUBRIC = (
    "Evaluate the answer on: accuracy (does it match the expected answer?), "
    "completeness (are all required details present?), "
    "professionalism (clear and well-formatted?), "
    "usefulness (actionable and relevant?)."
)


class LLMRubricGrader(BaseGrader):
    """Grades Outcome (+ optionally Trajectory) using an LLM judge."""

    name = "llm_rubric"

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(config)
        # Lazy-import to avoid hard dependency
        self._evaluator = None

    def _get_evaluator(self):
        if self._evaluator is None:
            from ..evaluation.llm_evaluator import LLMEvaluator

            self._evaluator = LLMEvaluator(self.config)
        return self._evaluator

    async def grade(
        self,
        outcome: Outcome,
        trajectory: Optional[Trajectory] = None,
        task: Optional[Any] = None,
    ) -> GradeResult:
        evaluator = self._get_evaluator()

        if not evaluator.client:
            return GradeResult(
                grader_name=self.name,
                score=0.5,
                passed=True,
                reasoning="LLM client not configured – returning neutral score.",
                details={"mode": "degraded"},
            )

        question = getattr(task, "question", "") if task else ""
        expected = getattr(task, "expected_answer", None) if task else None
        context: Dict[str, Any] = {}

        if task:
            context["rubric"] = self.config.get("rubric", _DEFAULT_RUBRIC)
            context["category"] = getattr(task, "category", "")
            context["task_type"] = getattr(task, "task_type", "")

        # Optionally include trajectory summary
        if trajectory and self.config.get("include_trajectory", True):
            context["trajectory_summary"] = _summarise_trajectory(trajectory)

        try:
            raw = await evaluator.evaluate(
                question=question,
                agent_response=outcome.answer,
                expected_answer=expected,
                context=context,
            )
        except Exception as exc:
            logger.warning("LLMRubricGrader.grade() failed: %s", exc)
            return GradeResult(
                grader_name=self.name,
                score=0.0,
                passed=False,
                reasoning=f"LLM evaluation error: {exc}",
                details={"error": str(exc)},
            )

        score = float(raw.get("score", 0.0))
        passed = score >= self.config.get("pass_threshold", 0.6)

        return GradeResult(
            grader_name=self.name,
            score=round(score, 4),
            passed=passed,
            reasoning=raw.get("reasoning", ""),
            details=raw.get("details", {}),
        )


# ── helpers ───────────────────────────────────────────────────────────────────


def _summarise_trajectory(trajectory: Trajectory) -> str:
    """Build a short textual summary of a trajectory for the LLM prompt."""
    lines = [f"Turns: {trajectory.n_turns}, Tool calls: {trajectory.n_toolcalls}"]
    for step in trajectory.steps[:20]:  # cap to avoid token explosion
        if step.tool_name:
            status = "OK" if step.tool_success else "FAIL"
            lines.append(f"  [{status}] {step.tool_name}({json.dumps(step.tool_input or {})})")
        elif step.reasoning:
            lines.append(f"  [think] {step.reasoning[:120]}")
    return "\n".join(lines)
