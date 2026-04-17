"""
MultiTrialRunner – runs the same task N times and aggregates results.

Each run produces a Trial (Trajectory + Outcome + GraderResults).
The runner computes:
  - mean / variance / worst-case scores
  - Pass@k probability
  - stability flag (variance > threshold → UNSTABLE)
"""

from __future__ import annotations

import asyncio
import logging
import math
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from ..trajectory import (
    GradeResult,
    Outcome,
    TrackedMetrics,
    Trajectory,
    Trial,
)

logger = logging.getLogger(__name__)

# ── aggregated result ─────────────────────────────────────────────────────────


@dataclass
class MultiTrialResult:
    """Aggregated result across N trials of a single task."""

    task_id: str
    question: str
    trials: List[Trial] = field(default_factory=list)

    # Per-grader mean scores
    grader_scores: Dict[str, float] = field(default_factory=dict)

    # Combined score statistics
    mean_score: float = 0.0
    variance: float = 0.0
    worst_case_score: float = 0.0
    best_score: float = 0.0

    # Pass@k  {k: probability}
    pass_at_k: Dict[int, float] = field(default_factory=dict)

    # Aggregated runtime metrics
    aggregated_metrics: Dict[str, Any] = field(default_factory=dict)

    # Stability flag: True when variance > threshold
    is_unstable: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_id": self.task_id,
            "question": self.question,
            "n_trials": len(self.trials),
            "grader_scores": self.grader_scores,
            "mean_score": self.mean_score,
            "variance": self.variance,
            "worst_case_score": self.worst_case_score,
            "best_score": self.best_score,
            "pass_at_k": self.pass_at_k,
            "aggregated_metrics": self.aggregated_metrics,
            "is_unstable": self.is_unstable,
            "trials": [t.to_dict() for t in self.trials],
        }


# ── runner ────────────────────────────────────────────────────────────────────


class MultiTrialRunner:
    """Runs a task multiple times and aggregates results.

    Args:
        run_fn: Async callable ``(task) -> Trial``.  This is typically
            provided by the EvaluationEngine or AgentTrialBench and handles
            the full agent call + grading pipeline.
        num_trials: How many times to run each task (default: 3).
        aggregation: One of ``"pass_at_k"`` or ``"mean"`` (default: pass_at_k).
        pass_threshold: Minimum combined score to count a trial as "passed"
            for Pass@k calculation (default: 0.7).
        variance_threshold: Trials whose score variance exceeds this value
            are flagged as ``UNSTABLE`` (default: 0.15).
        max_concurrent: Maximum number of concurrent trial executions
            (default: 3).
    """

    DEFAULT_K_VALUES = (1, 2, 3)

    def __init__(
        self,
        run_fn: Callable,
        *,
        num_trials: int = 3,
        aggregation: str = "pass_at_k",
        pass_threshold: float = 0.7,
        variance_threshold: float = 0.15,
        max_concurrent: int = 3,
    ) -> None:
        self.run_fn = run_fn
        self.num_trials = max(1, num_trials)
        self.aggregation = aggregation
        self.pass_threshold = pass_threshold
        self.variance_threshold = variance_threshold
        self.max_concurrent = max_concurrent

    # ── public API ────────────────────────────────────────────────────────────

    async def run(self, task: Any) -> MultiTrialResult:
        """Run ``task`` num_trials times and return aggregated results."""
        semaphore = asyncio.Semaphore(self.max_concurrent)
        trials: List[Optional[Trial]] = [None] * self.num_trials

        async def _one_trial(idx: int) -> None:
            async with semaphore:
                try:
                    trial = await self.run_fn(task, trial_id=idx)
                    trial.trial_id = idx
                    trials[idx] = trial
                except Exception as exc:
                    logger.warning("Trial %d for task %s failed: %s", idx, task.task_id, exc)
                    trials[idx] = _make_failed_trial(idx, task.task_id, str(exc))

        await asyncio.gather(*[_one_trial(i) for i in range(self.num_trials)])

        valid_trials = [t for t in trials if t is not None]
        return self._aggregate(task, valid_trials)

    # ── aggregation ───────────────────────────────────────────────────────────

    def _aggregate(self, task: Any, trials: List[Trial]) -> MultiTrialResult:
        if not trials:
            return MultiTrialResult(
                task_id=getattr(task, "task_id", "unknown"),
                question=getattr(task, "question", ""),
            )

        # Combined score per trial (average of grader scores if present,
        # else fall back to outcome.confidence as a proxy)
        trial_scores = [_combined_score(t) for t in trials]

        mean_score = sum(trial_scores) / len(trial_scores)
        variance = _variance(trial_scores, mean_score)
        worst = min(trial_scores)
        best = max(trial_scores)

        # Per-grader means
        grader_names: set = set()
        for t in trials:
            grader_names.update(t.grader_results.keys())

        grader_means: Dict[str, float] = {}
        for gname in grader_names:
            vals = [
                t.grader_results[gname].score
                for t in trials
                if gname in t.grader_results
            ]
            grader_means[gname] = sum(vals) / len(vals) if vals else 0.0

        # Pass@k
        passes = [s >= self.pass_threshold for s in trial_scores]
        n = len(passes)
        c = sum(passes)
        pass_at_k = {k: _pass_at_k(n, c, k) for k in self.DEFAULT_K_VALUES if k <= n}

        # Aggregated runtime metrics
        agg_metrics = _aggregate_metrics([t.metrics for t in trials])

        # Stability
        is_unstable = variance > self.variance_threshold

        # Mark unstable trials
        if is_unstable:
            for t in trials:
                t.is_unstable = True

        return MultiTrialResult(
            task_id=getattr(task, "task_id", trials[0].task_id),
            question=getattr(task, "question", ""),
            trials=trials,
            grader_scores=grader_means,
            mean_score=round(mean_score, 4),
            variance=round(variance, 4),
            worst_case_score=round(worst, 4),
            best_score=round(best, 4),
            pass_at_k=pass_at_k,
            aggregated_metrics=agg_metrics,
            is_unstable=is_unstable,
        )


# ── helpers ───────────────────────────────────────────────────────────────────


def _combined_score(trial: Trial) -> float:
    if trial.grader_results:
        scores = [g.score for g in trial.grader_results.values()]
        return sum(scores) / len(scores)
    return trial.outcome.confidence if trial.outcome else 0.0


def _variance(values: List[float], mean: float) -> float:
    if len(values) < 2:
        return 0.0
    return sum((v - mean) ** 2 for v in values) / len(values)


def _pass_at_k(n: int, c: int, k: int) -> float:
    """Pass@k probability: at least one of k samples is correct."""
    if n == 0 or k == 0:
        return 0.0
    if n - c < k:
        return 1.0
    # 1 - C(n-c, k) / C(n, k)
    result = 1.0
    for i in range(k):
        result *= (n - c - i) / (n - i)
    return round(1.0 - result, 6)


def _aggregate_metrics(metrics_list: List[TrackedMetrics]) -> Dict[str, Any]:
    if not metrics_list:
        return {}
    keys = ("n_turns", "n_toolcalls", "tokens", "latency", "cost_usd")
    agg: Dict[str, float] = {}
    for key in keys:
        vals = [getattr(m, key, 0) for m in metrics_list]
        agg[f"mean_{key}"] = round(sum(vals) / len(vals), 4)
        agg[f"total_{key}"] = round(sum(vals), 4)
    return agg


def _make_failed_trial(trial_id: int, task_id: str, error: str) -> Trial:
    empty_traj = Trajectory(task_id=task_id, start_time=time.time())
    empty_outcome = Outcome(answer="", confidence=0.0)
    empty_metrics = TrackedMetrics()
    return Trial(
        trial_id=trial_id,
        task_id=task_id,
        trajectory=empty_traj,
        outcome=empty_outcome,
        metrics=empty_metrics,
        success=False,
        error=error,
    )
