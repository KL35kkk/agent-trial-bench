"""
Graders package – four parallel graders that evaluate Outcome and Trajectory.

Grader          | Type         | Evaluates
----------------|--------------|----------------------------
deterministic   | code-based   | Outcome
llm_rubric      | model-based  | Trajectory + Outcome
state_check     | code-based   | Outcome (environment state)
tool_calls      | code-based   | Trajectory
"""

from .base import BaseGrader
from .deterministic import DeterministicTestsGrader
from .llm_rubric import LLMRubricGrader
from .state_check import StateCheckGrader
from .tool_calls import ToolCallsGrader

GRADER_REGISTRY: dict = {
    "deterministic_tests": DeterministicTestsGrader,
    "llm_rubric": LLMRubricGrader,
    "state_check": StateCheckGrader,
    "tool_calls": ToolCallsGrader,
}


def build_grader(name: str, config: dict = None) -> BaseGrader:
    """Instantiate a grader by registry name."""
    cls = GRADER_REGISTRY.get(name)
    if cls is None:
        raise ValueError(f"Unknown grader '{name}'. Available: {list(GRADER_REGISTRY)}")
    return cls(config)


__all__ = [
    "BaseGrader",
    "DeterministicTestsGrader",
    "LLMRubricGrader",
    "StateCheckGrader",
    "ToolCallsGrader",
    "GRADER_REGISTRY",
    "build_grader",
]
