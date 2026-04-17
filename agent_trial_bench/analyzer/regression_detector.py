"""
Regression Detector – compares two evaluation runs and identifies tasks
where quality has degraded.

A "regression" is defined as a task whose score decreased by more than
``regression_threshold`` relative to a baseline run.  A "new failure mode"
is a task that previously had no failure classification but now has one.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

logger = logging.getLogger(__name__)


@dataclass
class TaskSnapshot:
    """Score snapshot for a single task in one evaluation run."""

    task_id: str
    score: float
    grader_scores: Dict[str, float] = field(default_factory=dict)
    failure_category: Optional[str] = None  # From FailureClassifier
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_id": self.task_id,
            "score": self.score,
            "grader_scores": self.grader_scores,
            "failure_category": self.failure_category,
            "metadata": self.metadata,
        }


@dataclass
class RegressionReport:
    """Diff report between a baseline run and a candidate run."""

    # Tasks where score dropped below regression_threshold
    regressions: List[Dict[str, Any]] = field(default_factory=list)

    # Tasks where score improved
    improvements: List[Dict[str, Any]] = field(default_factory=list)

    # Tasks that now fail for the first time
    new_failures: List[str] = field(default_factory=list)

    # Tasks that previously failed but now pass
    recovered: List[str] = field(default_factory=list)

    # New failure categories that didn't exist in baseline
    new_failure_modes: List[str] = field(default_factory=list)

    # Summary statistics
    baseline_mean: float = 0.0
    candidate_mean: float = 0.0
    score_delta: float = 0.0
    regression_count: int = 0
    improvement_count: int = 0
    is_passing: bool = True  # Set by CIGate

    def to_dict(self) -> Dict[str, Any]:
        return {
            "regressions": self.regressions,
            "improvements": self.improvements,
            "new_failures": self.new_failures,
            "recovered": self.recovered,
            "new_failure_modes": self.new_failure_modes,
            "baseline_mean": self.baseline_mean,
            "candidate_mean": self.candidate_mean,
            "score_delta": self.score_delta,
            "regression_count": self.regression_count,
            "improvement_count": self.improvement_count,
            "is_passing": self.is_passing,
        }

    def summary_text(self) -> str:
        lines = [
            "=== Regression Report ===",
            f"Baseline mean score : {self.baseline_mean:.4f}",
            f"Candidate mean score: {self.candidate_mean:.4f}",
            f"Score delta         : {self.score_delta:+.4f}",
            f"Regressions         : {self.regression_count}",
            f"Improvements        : {self.improvement_count}",
            f"New failures        : {len(self.new_failures)}",
            f"Recovered           : {len(self.recovered)}",
            f"New failure modes   : {self.new_failure_modes or 'none'}",
        ]
        if self.regressions:
            lines.append("\nRegressed tasks:")
            for r in self.regressions[:10]:
                lines.append(
                    f"  {r['task_id']}: {r['baseline_score']:.3f} → {r['candidate_score']:.3f}"
                    f" ({r['delta']:+.3f})"
                )
        return "\n".join(lines)


class RegressionDetector:
    """Detects regressions between two evaluation runs.

    Args:
        regression_threshold: Minimum score decrease to count as a regression
            (default: 0.05 = 5 percentage points).
        improvement_threshold: Minimum score increase to count as an improvement
            (default: 0.03).
    """

    def __init__(
        self,
        regression_threshold: float = 0.05,
        improvement_threshold: float = 0.03,
    ) -> None:
        self.regression_threshold = regression_threshold
        self.improvement_threshold = improvement_threshold

    # ── public API ────────────────────────────────────────────────────────────

    def compare(
        self,
        baseline: Sequence[TaskSnapshot],
        candidate: Sequence[TaskSnapshot],
    ) -> RegressionReport:
        """Compare baseline and candidate snapshots.

        Args:
            baseline: Snapshots from the reference run (e.g. previous release).
            candidate: Snapshots from the run under test (e.g. new release).

        Returns:
            RegressionReport with full diff.
        """
        baseline_map = {s.task_id: s for s in baseline}
        candidate_map = {s.task_id: s for s in candidate}

        regressions: List[Dict[str, Any]] = []
        improvements: List[Dict[str, Any]] = []
        new_failures: List[str] = []
        recovered: List[str] = []

        baseline_failures = {s.task_id: s.failure_category for s in baseline if s.failure_category}
        candidate_failures = {s.task_id: s.failure_category for s in candidate if s.failure_category}

        # Per-task comparison
        for task_id, base_snap in baseline_map.items():
            cand_snap = candidate_map.get(task_id)
            if cand_snap is None:
                continue

            delta = cand_snap.score - base_snap.score

            if delta <= -self.regression_threshold:
                regressions.append(
                    {
                        "task_id": task_id,
                        "baseline_score": base_snap.score,
                        "candidate_score": cand_snap.score,
                        "delta": delta,
                        "baseline_failure": base_snap.failure_category,
                        "candidate_failure": cand_snap.failure_category,
                    }
                )

            elif delta >= self.improvement_threshold:
                improvements.append(
                    {
                        "task_id": task_id,
                        "baseline_score": base_snap.score,
                        "candidate_score": cand_snap.score,
                        "delta": delta,
                    }
                )

            # New failure (was passing, now failing)
            if task_id not in baseline_failures and task_id in candidate_failures:
                new_failures.append(task_id)

            # Recovery (was failing, now passing)
            if task_id in baseline_failures and task_id not in candidate_failures:
                recovered.append(task_id)

        # New failure modes
        baseline_mode_set = set(baseline_failures.values())
        candidate_mode_set = set(candidate_failures.values())
        new_modes = sorted(candidate_mode_set - baseline_mode_set - {None})

        # Means
        baseline_scores = [s.score for s in baseline]
        candidate_scores = [s.score for s in candidate]
        baseline_mean = sum(baseline_scores) / len(baseline_scores) if baseline_scores else 0.0
        candidate_mean = sum(candidate_scores) / len(candidate_scores) if candidate_scores else 0.0

        return RegressionReport(
            regressions=sorted(regressions, key=lambda r: r["delta"]),
            improvements=sorted(improvements, key=lambda r: r["delta"], reverse=True),
            new_failures=new_failures,
            recovered=recovered,
            new_failure_modes=new_modes,
            baseline_mean=round(baseline_mean, 4),
            candidate_mean=round(candidate_mean, 4),
            score_delta=round(candidate_mean - baseline_mean, 4),
            regression_count=len(regressions),
            improvement_count=len(improvements),
        )

    # ── snapshot I/O ─────────────────────────────────────────────────────────

    @staticmethod
    def snapshots_from_results(results: Sequence[Any]) -> List[TaskSnapshot]:
        """Convert a list of evaluation result dicts to TaskSnapshot objects.

        Accepts dicts with at minimum ``task_id`` and ``evaluation_score`` (v1
        format) or ``mean_score`` (v2 MultiTrialResult format).
        """
        snapshots: List[TaskSnapshot] = []
        for r in results:
            if isinstance(r, dict):
                task_id = r.get("task_id", "")
                score = float(r.get("mean_score") or r.get("evaluation_score") or 0.0)
                grader_scores = r.get("grader_scores", {})
                failure_category = r.get("failure_category")
            else:
                task_id = getattr(r, "task_id", "")
                score = float(getattr(r, "mean_score", None) or getattr(r, "evaluation_score", 0.0))
                grader_scores = getattr(r, "grader_scores", {})
                failure_category = getattr(r, "failure_category", None)

            snapshots.append(
                TaskSnapshot(
                    task_id=task_id,
                    score=score,
                    grader_scores=grader_scores,
                    failure_category=failure_category,
                )
            )
        return snapshots

    @staticmethod
    def save_snapshot(snapshots: Sequence[TaskSnapshot], path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = [s.to_dict() for s in snapshots]
        path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
        logger.info("Saved %d snapshots to %s", len(snapshots), path)

    @staticmethod
    def load_snapshot(path: str | Path) -> List[TaskSnapshot]:
        path = Path(path)
        data = json.loads(path.read_text(encoding="utf-8"))
        return [
            TaskSnapshot(
                task_id=d["task_id"],
                score=d["score"],
                grader_scores=d.get("grader_scores", {}),
                failure_category=d.get("failure_category"),
                metadata=d.get("metadata", {}),
            )
            for d in data
        ]
