"""
ToolCallsGrader – evaluates the agent's tool usage within a Trajectory.

Checks: tool call success rate, selection relevance (vs. expected tool sets
contributed by the domain registry), absence of redundant calls / loops, and
domain-aware parameter well-formedness.

No domain is hardcoded: tool patterns and parameter validators come from the
pluggable ``DomainPlugin`` system, so the same grader works for Web3, general
research agents, coding agents, etc.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Set

from ..domains import expected_tools_for, validate_params_across_domains
from ..trajectory import GradeResult, Outcome, StepType, Trajectory
from .base import BaseGrader


class ToolCallsGrader(BaseGrader):
    """Grades tool usage quality from the ``Trajectory``.

    Task-context keys:
        expected_tools: Iterable[str] – authoritative per-task expected tool
            set.  If supplied, overrides domain-registry lookup.
    """

    name = "tool_calls"

    async def grade(
        self,
        outcome: Outcome,
        trajectory: Optional[Trajectory] = None,
        task: Optional[Any] = None,
    ) -> GradeResult:
        if not trajectory or not trajectory.steps:
            return self._make_degraded_result("No trajectory available for tool call analysis.")

        tool_steps = [s for s in trajectory.steps if s.step_type == StepType.TOOL_CALL]

        if not tool_steps:
            return GradeResult(
                grader_name=self.name,
                score=0.7,
                passed=True,
                reasoning="No tool calls in trajectory (acceptable for knowledge-based tasks).",
            )

        assertions: List[Dict[str, Any]] = []
        category = str(getattr(task, "category", "")) if task else ""
        if hasattr(category, "value"):
            category = category.value

        # 1. Tool call success rate
        success_rate = sum(1 for s in tool_steps if s.tool_success) / len(tool_steps)
        assertions.append(
            {
                "name": "tool_success_rate",
                "passed": success_rate >= self.config.get("min_success_rate", 0.8),
                "value": success_rate,
            }
        )

        # 2. Tool selection relevance (task-level wins over domain-level)
        expected_tools: Set[str] = set()
        task_ctx = (task.context if task else {}) or {}
        override = task_ctx.get("expected_tools") or self.config.get("expected_tools")
        if override:
            expected_tools = set(override)
        elif category:
            expected_tools = expected_tools_for(category)

        if expected_tools:
            used_tools = {s.tool_name for s in tool_steps if s.tool_name}
            relevance = (
                len(used_tools & expected_tools) / len(used_tools) if used_tools else 0.0
            )
            assertions.append(
                {
                    "name": "tool_selection_relevance",
                    "passed": relevance >= self.config.get("min_relevance", 0.5),
                    "value": relevance,
                    "expected_tools": sorted(expected_tools),
                }
            )

        # 3. Redundancy detection
        call_signatures = [(s.tool_name, str(s.tool_input)) for s in tool_steps]
        unique_ratio = (
            len(set(call_signatures)) / len(call_signatures) if call_signatures else 1.0
        )
        assertions.append(
            {
                "name": "no_redundant_calls",
                "passed": unique_ratio >= self.config.get("min_unique_ratio", 0.5),
                "value": unique_ratio,
            }
        )

        # 4. Trajectory-level loop detection
        assertions.append(
            {"name": "no_infinite_loop", "passed": not trajectory.has_loops}
        )

        # 5. Domain-aware parameter format validation
        param_failures: List[str] = []
        for step in tool_steps:
            if step.tool_input and not validate_params_across_domains(step.tool_input):
                param_failures.append(f"step_{step.step_id}")
        assertions.append(
            {
                "name": "valid_tool_params",
                "passed": len(param_failures) == 0,
                "failed_steps": param_failures,
            }
        )

        # 6. Step count efficiency
        max_steps = self.config.get("max_tool_calls")
        if max_steps:
            assertions.append(
                {
                    "name": "step_count_within_limit",
                    "passed": len(tool_steps) <= max_steps,
                    "value": len(tool_steps),
                }
            )

        passed_count = sum(1 for a in assertions if a["passed"])
        score = passed_count / len(assertions)

        return GradeResult(
            grader_name=self.name,
            score=round(score, 4),
            passed=score >= self.config.get("pass_threshold", 0.6),
            reasoning=(
                f"Tool calls: {passed_count}/{len(assertions)} checks passed. "
                f"{len(tool_steps)} tool calls, success rate {success_rate:.0%}."
            ),
            assertions=assertions,
            details={
                "n_tool_calls": len(tool_steps),
                "tool_success_rate": success_rate,
                "unique_ratio": unique_ratio,
            },
        )
