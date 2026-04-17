"""
Failure Classifier – categorises why a Trial failed.

Failure categories (matching the plan specification):
  tool_error       – one or more tool calls failed or returned errors
  reasoning_error  – agent reached wrong conclusion despite good tools
  planning_error   – agent chose wrong tools / wrong execution order
  loop_detected    – agent entered a repetitive tool-call loop
  no_answer        – agent returned empty or unintelligible answer
  unknown          – cannot determine cause from available data
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Sequence

from ..trajectory import GradeResult, StepType, Trajectory, Trial

logger = logging.getLogger(__name__)


class FailureCategory(Enum):
    TOOL_ERROR = "tool_error"
    REASONING_ERROR = "reasoning_error"
    PLANNING_ERROR = "planning_error"
    LOOP_DETECTED = "loop_detected"
    NO_ANSWER = "no_answer"
    UNKNOWN = "unknown"


@dataclass
class ClassifiedFailure:
    """Result of classifying a single failed trial."""

    trial_id: int
    task_id: str
    category: FailureCategory
    confidence: float           # 0.0 – 1.0
    evidence: List[str] = field(default_factory=list)
    raw_grader_scores: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "trial_id": self.trial_id,
            "task_id": self.task_id,
            "category": self.category.value,
            "confidence": self.confidence,
            "evidence": self.evidence,
            "raw_grader_scores": self.raw_grader_scores,
        }


class FailureClassifier:
    """Classify failures in a batch of trials.

    Classification uses a rule-based cascade:
    1. Loop detection (highest priority – overrides everything else)
    2. No-answer check
    3. Tool error (tool_success_rate < threshold)
    4. Planning error (wrong tools used, excessive calls)
    5. Reasoning error (tools OK but wrong outcome)
    6. Unknown fallback
    """

    def __init__(
        self,
        tool_error_threshold: float = 0.7,
        pass_threshold: float = 0.6,
    ) -> None:
        self.tool_error_threshold = tool_error_threshold
        self.pass_threshold = pass_threshold

    def classify(self, trial: Trial) -> Optional[ClassifiedFailure]:
        """Classify a single trial.  Returns ``None`` if the trial passed."""
        if _trial_passed(trial, self.pass_threshold):
            return None

        traj = trial.trajectory
        evidence: List[str] = []

        # 1. Loop detection
        if traj and traj.has_loops:
            evidence.append(
                f"Repeated tool calls detected: {traj.n_toolcalls} calls with duplicates."
            )
            return ClassifiedFailure(
                trial_id=trial.trial_id,
                task_id=trial.task_id,
                category=FailureCategory.LOOP_DETECTED,
                confidence=0.95,
                evidence=evidence,
                raw_grader_scores=_grader_score_map(trial),
            )

        # 2. No answer
        answer = (trial.outcome.answer if trial.outcome else "").strip()
        if not answer or len(answer) < 3:
            evidence.append("Agent returned empty or very short answer.")
            return ClassifiedFailure(
                trial_id=trial.trial_id,
                task_id=trial.task_id,
                category=FailureCategory.NO_ANSWER,
                confidence=0.99,
                evidence=evidence,
                raw_grader_scores=_grader_score_map(trial),
            )

        # 3. Tool error
        tool_score = trial.grader_results.get("tool_calls")
        if tool_score and tool_score.score < self.tool_error_threshold:
            tsr = traj.tool_success_rate if traj else 0.0
            evidence.append(
                f"Tool call grader score={tool_score.score:.2f}, "
                f"tool success rate={tsr:.0%}."
            )
            # Check if it looks more like a planning error
            if traj and _is_planning_error(traj):
                evidence.append("Wrong tool selection pattern detected.")
                return ClassifiedFailure(
                    trial_id=trial.trial_id,
                    task_id=trial.task_id,
                    category=FailureCategory.PLANNING_ERROR,
                    confidence=0.75,
                    evidence=evidence,
                    raw_grader_scores=_grader_score_map(trial),
                )
            return ClassifiedFailure(
                trial_id=trial.trial_id,
                task_id=trial.task_id,
                category=FailureCategory.TOOL_ERROR,
                confidence=0.85,
                evidence=evidence,
                raw_grader_scores=_grader_score_map(trial),
            )

        # 4. Planning error (without tool failures)
        if traj and _is_planning_error(traj):
            evidence.append(
                f"Excessive or irrelevant tool calls: {traj.n_toolcalls} calls."
            )
            return ClassifiedFailure(
                trial_id=trial.trial_id,
                task_id=trial.task_id,
                category=FailureCategory.PLANNING_ERROR,
                confidence=0.70,
                evidence=evidence,
                raw_grader_scores=_grader_score_map(trial),
            )

        # 5. Reasoning error (tools OK, output wrong)
        det_score = trial.grader_results.get("deterministic_tests")
        llm_score = trial.grader_results.get("llm_rubric")
        if (det_score and det_score.score < 0.5) or (llm_score and llm_score.score < 0.5):
            evidence.append(
                f"deterministic={det_score.score if det_score else 'n/a'}, "
                f"llm_rubric={llm_score.score if llm_score else 'n/a'}."
            )
            return ClassifiedFailure(
                trial_id=trial.trial_id,
                task_id=trial.task_id,
                category=FailureCategory.REASONING_ERROR,
                confidence=0.70,
                evidence=evidence,
                raw_grader_scores=_grader_score_map(trial),
            )

        # Fallback
        return ClassifiedFailure(
            trial_id=trial.trial_id,
            task_id=trial.task_id,
            category=FailureCategory.UNKNOWN,
            confidence=0.3,
            evidence=["Could not determine failure cause from available signals."],
            raw_grader_scores=_grader_score_map(trial),
        )

    def classify_batch(
        self, trials: Sequence[Trial]
    ) -> List[ClassifiedFailure]:
        """Classify all failed trials in the batch."""
        results: List[ClassifiedFailure] = []
        for trial in trials:
            cf = self.classify(trial)
            if cf is not None:
                results.append(cf)
        return results


# ── helpers ───────────────────────────────────────────────────────────────────


def _trial_passed(trial: Trial, threshold: float) -> bool:
    if trial.success is False:
        return False
    if not trial.grader_results:
        return trial.outcome.confidence >= threshold if trial.outcome else False
    mean = sum(g.score for g in trial.grader_results.values()) / len(trial.grader_results)
    return mean >= threshold


def _grader_score_map(trial: Trial) -> Dict[str, float]:
    return {k: v.score for k, v in trial.grader_results.items()}


def _is_planning_error(traj: Trajectory) -> bool:
    """Heuristic: many tool calls with no apparent progress."""
    if traj.n_toolcalls == 0:
        return False
    # If more than 8 tool calls but still failed, likely a planning issue
    if traj.n_toolcalls > 8:
        return True
    # More reasoning steps than tool results – the agent over-thought
    if traj.n_reasoning_steps > traj.n_toolcalls * 3:
        return True
    return False
