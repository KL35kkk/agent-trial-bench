"""
Base Grader abstract class.

All graders receive an Outcome and optionally a Trajectory, then produce
a GradeResult. Multiple graders run in parallel per trial.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from ..trajectory import GradeResult, Outcome, Trajectory


class BaseGrader(ABC):
    """Abstract base for all graders.

    Subclasses must implement ``grade()``.
    """

    name: str = "base"

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        self.config = config or {}

    @abstractmethod
    async def grade(
        self,
        outcome: Outcome,
        trajectory: Optional[Trajectory] = None,
        task: Optional[Any] = None,
    ) -> GradeResult:
        """Grade a trial.

        Args:
            outcome: The final state produced by the agent.
            trajectory: The full execution trace (may be None if the agent
                did not return structured trace data).
            task: The originating task object (carries expected_answer,
                category, context, etc.).

        Returns:
            GradeResult with score 0.0–1.0.
        """

    def get_assertions(self) -> List[str]:
        """Return the list of assertion names this grader checks."""
        return []

    def _make_degraded_result(self, reason: str = "trajectory not available") -> GradeResult:
        """Return a neutral degraded result when grading is not possible."""
        return GradeResult(
            grader_name=self.name,
            score=0.5,
            passed=True,
            reasoning=f"Degraded: {reason}",
            details={"mode": "degraded"},
        )
