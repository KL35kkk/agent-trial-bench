"""
Trajectory, Step, Trial, Outcome, and TrackedMetrics data structures.

These align with Anthropic's Agent Evaluation architecture:
- Trajectory: the full execution record (messages, tool_calls, reasoning)
- Step: a single action within a trajectory
- Trial: one complete run attempt of a task
- Outcome: the final environment state after a trial
- TrackedMetrics: runtime performance metrics
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class StepType(Enum):
    MESSAGE = "message"
    REASONING = "reasoning"
    TOOL_CALL = "tool_call"
    TOOL_RESULT = "tool_result"
    ERROR = "error"


@dataclass
class Step:
    """A single step within an agent's execution trajectory."""

    step_id: int
    step_type: StepType
    timestamp: float
    content: str

    # Tool call fields
    tool_name: Optional[str] = None
    tool_input: Optional[Dict[str, Any]] = None
    tool_output: Optional[Any] = None
    tool_success: bool = True

    # Reasoning fields
    reasoning: Optional[str] = None
    confidence: float = 1.0

    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "step_id": self.step_id,
            "step_type": self.step_type.value,
            "timestamp": self.timestamp,
            "content": self.content,
            "tool_name": self.tool_name,
            "tool_input": self.tool_input,
            "tool_output": self.tool_output,
            "tool_success": self.tool_success,
            "reasoning": self.reasoning,
            "confidence": self.confidence,
            "metadata": self.metadata,
        }


@dataclass
class Trajectory:
    """Complete execution trace of an agent run.

    Corresponds to Anthropic's Trajectory concept: the full record of
    messages, tool_calls, and reasoning steps.
    """

    task_id: str
    steps: List[Step] = field(default_factory=list)

    # Aggregate counters (auto-maintained via add_step)
    n_turns: int = 0
    n_toolcalls: int = 0
    n_reasoning_steps: int = 0
    n_errors: int = 0

    start_time: Optional[float] = None
    end_time: Optional[float] = None
    total_duration: float = 0.0

    def add_step(self, step: Step) -> None:
        self.steps.append(step)
        if step.step_type == StepType.TOOL_CALL:
            self.n_toolcalls += 1
        elif step.step_type == StepType.REASONING:
            self.n_reasoning_steps += 1
        elif step.step_type == StepType.ERROR:
            self.n_errors += 1
        elif step.step_type == StepType.MESSAGE:
            self.n_turns += 1

    def finalize(self) -> None:
        """Mark the trajectory as complete and compute duration."""
        if self.end_time is None:
            self.end_time = time.time()
        if self.start_time is not None:
            self.total_duration = self.end_time - self.start_time

    @property
    def has_loops(self) -> bool:
        """Detect repeated identical tool calls (potential infinite loops)."""
        calls = [
            (s.tool_name, str(s.tool_input))
            for s in self.steps
            if s.step_type == StepType.TOOL_CALL
        ]
        return len(calls) != len(set(calls))

    @property
    def tool_success_rate(self) -> float:
        tool_calls = [s for s in self.steps if s.step_type == StepType.TOOL_CALL]
        if not tool_calls:
            return 1.0
        return sum(1 for s in tool_calls if s.tool_success) / len(tool_calls)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_id": self.task_id,
            "steps": [s.to_dict() for s in self.steps],
            "n_turns": self.n_turns,
            "n_toolcalls": self.n_toolcalls,
            "n_reasoning_steps": self.n_reasoning_steps,
            "n_errors": self.n_errors,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "total_duration": self.total_duration,
        }

    @classmethod
    def from_agent_response(cls, task_id: str, response: Dict[str, Any]) -> "Trajectory":
        """Parse a Trajectory from an agent's structured JSON response.

        Expected response format::

            {
              "answer": "...",
              "trajectory": {
                "steps": [
                  {"step_id": 1, "type": "reasoning", "content": "...", "timestamp": 0.0},
                  {"step_id": 2, "type": "tool_call", "tool_name": "web_search",
                   "tool_input": {...}, "tool_output": {...}, "success": true, "timestamp": 0.0},
                  ...
                ],
                "n_turns": 3,
                "total_duration": 2.5
              }
            }
        """
        traj = cls(task_id=task_id, start_time=time.time())

        raw = response.get("trajectory") or {}
        raw_steps = raw.get("steps", [])

        type_map = {
            "message": StepType.MESSAGE,
            "reasoning": StepType.REASONING,
            "thought": StepType.REASONING,
            "tool_call": StepType.TOOL_CALL,
            "tool_result": StepType.TOOL_RESULT,
            "error": StepType.ERROR,
        }

        for idx, raw_step in enumerate(raw_steps):
            step_type_str = raw_step.get("type", "message").lower()
            step_type = type_map.get(step_type_str, StepType.MESSAGE)

            step = Step(
                step_id=raw_step.get("step_id", idx + 1),
                step_type=step_type,
                timestamp=raw_step.get("timestamp", time.time()),
                content=raw_step.get("content", ""),
                tool_name=raw_step.get("tool_name"),
                tool_input=raw_step.get("tool_input"),
                tool_output=raw_step.get("tool_output"),
                tool_success=raw_step.get("success", True),
                reasoning=raw_step.get("reasoning"),
                confidence=raw_step.get("confidence", 1.0),
                metadata=raw_step.get("metadata", {}),
            )
            traj.add_step(step)

        # Honour explicit counts if the agent provided them
        if "n_turns" in raw:
            traj.n_turns = raw["n_turns"]
        if "total_duration" in raw:
            traj.total_duration = raw["total_duration"]

        return traj


@dataclass
class Outcome:
    """The final environment state produced by a trial.

    Corresponds to Anthropic's Outcome concept: not just the answer text,
    but the full observable state after the agent has run.
    """

    answer: str
    confidence: float = 0.0
    state: Dict[str, Any] = field(default_factory=dict)
    sources: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "answer": self.answer,
            "confidence": self.confidence,
            "state": self.state,
            "sources": self.sources,
            "metadata": self.metadata,
        }

    @classmethod
    def from_agent_response(cls, response: Dict[str, Any]) -> "Outcome":
        state = response.get("state", {})
        sources = state.get("sources_consulted", []) if isinstance(state, dict) else []
        return cls(
            answer=str(response.get("answer", "")),
            confidence=float(response.get("confidence", 0.0)),
            state=state,
            sources=sources,
            metadata=response.get("metadata", {}),
        )


@dataclass
class TrackedMetrics:
    """Runtime performance metrics for a single trial.

    Corresponds to Anthropic's Tracked Metrics: n_turns, n_toolcalls,
    tokens, latency.
    """

    n_turns: int = 0
    n_toolcalls: int = 0
    tokens: int = 0
    latency: float = 0.0

    # Extended metrics
    prompt_tokens: int = 0
    completion_tokens: int = 0
    tool_success_rate: float = 1.0
    error_count: int = 0
    retry_count: int = 0
    cost_usd: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "n_turns": self.n_turns,
            "n_toolcalls": self.n_toolcalls,
            "tokens": self.tokens,
            "latency": self.latency,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "tool_success_rate": self.tool_success_rate,
            "error_count": self.error_count,
            "retry_count": self.retry_count,
            "cost_usd": self.cost_usd,
        }

    @classmethod
    def from_trajectory_and_response(
        cls,
        trajectory: Trajectory,
        response: Dict[str, Any],
        latency: float = 0.0,
    ) -> "TrackedMetrics":
        token_usage = response.get("token_usage", {})
        prompt_tokens = int(token_usage.get("prompt", 0))
        completion_tokens = int(token_usage.get("completion", 0))
        total_tokens = int(token_usage.get("total", prompt_tokens + completion_tokens))

        return cls(
            n_turns=trajectory.n_turns,
            n_toolcalls=trajectory.n_toolcalls,
            tokens=total_tokens,
            latency=latency,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            tool_success_rate=trajectory.tool_success_rate,
            error_count=trajectory.n_errors,
        )


@dataclass
class GradeResult:
    """Score produced by a single Grader."""

    grader_name: str
    score: float
    passed: bool
    reasoning: str
    details: Dict[str, Any] = field(default_factory=dict)
    assertions: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "grader_name": self.grader_name,
            "score": self.score,
            "passed": self.passed,
            "reasoning": self.reasoning,
            "details": self.details,
            "assertions": self.assertions,
        }


@dataclass
class Trial:
    """A single execution attempt of a task.

    Corresponds to Anthropic's Trial: one run that produces a Trajectory,
    an Outcome, and collected Metrics.
    """

    trial_id: int
    task_id: str
    trajectory: Trajectory
    outcome: Outcome
    metrics: TrackedMetrics

    grader_results: Dict[str, GradeResult] = field(default_factory=dict)

    started_at: float = field(default_factory=time.time)
    completed_at: float = 0.0
    success: bool = True
    error: Optional[str] = None

    # Stability flag set during multi-trial aggregation
    is_unstable: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "trial_id": self.trial_id,
            "task_id": self.task_id,
            "trajectory": self.trajectory.to_dict(),
            "outcome": self.outcome.to_dict(),
            "metrics": self.metrics.to_dict(),
            "grader_results": {k: v.to_dict() for k, v in self.grader_results.items()},
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "success": self.success,
            "error": self.error,
            "is_unstable": self.is_unstable,
        }
