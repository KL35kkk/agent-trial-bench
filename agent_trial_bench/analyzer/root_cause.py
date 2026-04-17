"""
Root Cause Analyzer – aggregates failure classifications into a
percentage breakdown and actionable report.

Output format::

    {
      "failure_distribution": {
        "tool_error": 0.60,
        "reasoning_error": 0.25,
        "planning_error": 0.15
      },
      "total_failures": 20,
      "top_failing_tasks": ["task_042", "task_078"],
      "recommendations": [...]
    }
"""

from __future__ import annotations

import json
import logging
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

from .failure_classifier import ClassifiedFailure, FailureCategory

logger = logging.getLogger(__name__)


@dataclass
class RootCauseReport:
    """Aggregated root-cause analysis."""

    total_trials: int
    total_failures: int
    pass_rate: float

    failure_distribution: Dict[str, float]  # category → fraction of failures
    failure_counts: Dict[str, int]          # category → raw count

    # Tasks that failed most often across multiple trials
    top_failing_tasks: List[str]

    # Per-task failure breakdown
    per_task: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    recommendations: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_trials": self.total_trials,
            "total_failures": self.total_failures,
            "pass_rate": self.pass_rate,
            "failure_distribution": self.failure_distribution,
            "failure_counts": self.failure_counts,
            "top_failing_tasks": self.top_failing_tasks,
            "per_task": self.per_task,
            "recommendations": self.recommendations,
        }

    def summary_text(self) -> str:
        lines = [
            "=== Root Cause Analysis ===",
            f"Total trials   : {self.total_trials}",
            f"Total failures : {self.total_failures}",
            f"Pass rate      : {self.pass_rate:.1%}",
            "",
            "Failure distribution:",
        ]
        for cat, frac in sorted(
            self.failure_distribution.items(), key=lambda x: x[1], reverse=True
        ):
            bar = "█" * int(frac * 20)
            lines.append(f"  {cat:<20} {bar} {frac:.0%} ({self.failure_counts.get(cat, 0)})")

        if self.top_failing_tasks:
            lines.append("")
            lines.append("Top failing tasks:")
            for tid in self.top_failing_tasks[:10]:
                lines.append(f"  {tid}")

        if self.recommendations:
            lines.append("")
            lines.append("Recommendations:")
            for rec in self.recommendations:
                lines.append(f"  • {rec}")

        return "\n".join(lines)


class RootCauseAnalyzer:
    """Aggregates ClassifiedFailures into a RootCauseReport."""

    _RECOMMENDATIONS: Dict[str, str] = {
        FailureCategory.TOOL_ERROR.value: (
            "Improve tool reliability: add retry logic, validate API keys, "
            "and implement graceful fallback when primary tools fail."
        ),
        FailureCategory.REASONING_ERROR.value: (
            "Improve reasoning quality: strengthen chain-of-thought prompting, "
            "add self-verification steps, and include few-shot examples for this task type."
        ),
        FailureCategory.PLANNING_ERROR.value: (
            "Improve planning: add explicit tool-selection reasoning, "
            "define clear stopping criteria, and penalise excessive tool calls."
        ),
        FailureCategory.LOOP_DETECTED.value: (
            "Fix infinite loops: add a global tool-call counter limit, "
            "detect repeated (tool, args) pairs, and break on stagnation."
        ),
        FailureCategory.NO_ANSWER.value: (
            "Enforce answer completeness: add a post-processing step that validates "
            "the output is non-empty before returning."
        ),
        FailureCategory.UNKNOWN.value: (
            "Enable trajectory logging to capture more signal for future classification."
        ),
    }

    def analyse(
        self,
        failures: Sequence[ClassifiedFailure],
        total_trials: int,
    ) -> RootCauseReport:
        """Produce a root-cause report from a batch of classified failures.

        Args:
            failures: All ClassifiedFailure objects from FailureClassifier.
            total_trials: Total number of trials run (including passed ones),
                used to compute the pass rate.
        """
        n_failures = len(failures)
        pass_rate = max(0.0, 1.0 - n_failures / max(total_trials, 1))

        # Category counts
        category_counter: Counter = Counter(f.category.value for f in failures)

        # Distribution (fraction of failures, not total trials)
        distribution: Dict[str, float] = {}
        if n_failures > 0:
            for cat, count in category_counter.items():
                distribution[cat] = round(count / n_failures, 4)

        # Per-task breakdown
        per_task: Dict[str, Dict[str, Any]] = {}
        task_fail_counts: Counter = Counter()
        for f in failures:
            task_fail_counts[f.task_id] += 1
            if f.task_id not in per_task:
                per_task[f.task_id] = {
                    "failure_count": 0,
                    "categories": [],
                    "evidence": [],
                }
            per_task[f.task_id]["failure_count"] += 1
            per_task[f.task_id]["categories"].append(f.category.value)
            per_task[f.task_id]["evidence"].extend(f.evidence[:2])

        # Top failing tasks (most frequently failing)
        top_failing = [tid for tid, _ in task_fail_counts.most_common(20)]

        # Recommendations for dominant failure categories
        top_cats = [cat for cat, _ in category_counter.most_common(3)]
        recommendations = [
            self._RECOMMENDATIONS[cat]
            for cat in top_cats
            if cat in self._RECOMMENDATIONS
        ]

        return RootCauseReport(
            total_trials=total_trials,
            total_failures=n_failures,
            pass_rate=round(pass_rate, 4),
            failure_distribution=distribution,
            failure_counts=dict(category_counter),
            top_failing_tasks=top_failing,
            per_task=per_task,
            recommendations=recommendations,
        )

    def analyse_from_trials(self, trials: Sequence[Any], total_trials: int) -> RootCauseReport:
        """Convenience wrapper that runs FailureClassifier internally.

        Args:
            trials: Sequence of Trial objects.
            total_trials: Total trial count for pass-rate denominator.
        """
        from .failure_classifier import FailureClassifier

        classifier = FailureClassifier()
        failures = classifier.classify_batch(trials)
        return self.analyse(failures, total_trials)
