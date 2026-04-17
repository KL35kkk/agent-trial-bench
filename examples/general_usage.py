"""
General-purpose evaluation example.

Demonstrates how to use DAB as a **domain-agnostic** agent benchmark –
no Web3 assumptions anywhere.  We:

1. Build a Trajectory / Outcome for a research agent answering a fact QA
   task (deterministic, offline – no network required).
2. Grade it with the same four graders used for Web3 agents.  The only
   domain-specific knowledge is what the dataset authors put in the task
   context (``check_format``, ``expected_tools``, ``state_format_checks``).
3. Show how adding a custom DomainPlugin extends validators / tool
   patterns without touching the core.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from agent_trial_bench import (
    BaseGrader,
    DeterministicTestsGrader,
    DomainPlugin,
    LLMRubricGrader,  # noqa: F401  (documented; not used in this offline demo)
    Outcome,
    StateCheckGrader,
    Step,
    StepType,
    TaskCategory,
    ToolCallsGrader,
    TrackedMetrics,
    Trajectory,
    Trial,
    register_domain,
)


# ── 1. Optional: register a custom domain plugin ──────────────────────────────


class ResearchDomain(DomainPlugin):
    """Extra validators and trusted sources for academic research agents."""

    name = "research"

    validators = {
        "doi": lambda s: s.startswith("10.") and "/" in s,
        "arxiv_id": lambda s: bool(s and s.replace(".", "").replace("v", "").isdigit()),
    }
    trusted_sources = {
        "arxiv.org",
        "nature.com",
        "sciencemag.org",
        "pubmed.ncbi.nlm.nih.gov",
        "acl-anthology.org",
    }
    tool_patterns = {
        "research": {
            "expected_tools": {"arxiv_search", "pubmed_search", "web_search", "summarize"},
            "required_params": ["query"],
        }
    }
    technical_terms = {"citation", "peer-review", "replication", "baseline", "ablation"}


register_domain(ResearchDomain())


# ── 2. Minimal task fixture (no dataset file needed) ──────────────────────────


@dataclass
class Task:
    question: str
    expected_answer: str
    category: TaskCategory
    context: Dict[str, Any]


task = Task(
    question="Who proposed the Transformer architecture and in which year?",
    expected_answer="Vaswani et al., 2017",
    category=TaskCategory.FACT_QA,
    context={
        "check_format": [],  # No strict format assertions for this QA
        "expected_tools": ["arxiv_search", "web_search"],
        "state_format_checks": {"primary_source": "url"},
    },
)


# ── 3. Simulate an agent's trajectory and outcome ─────────────────────────────


def build_trial() -> Trial:
    traj = Trajectory(task_id="demo-1")
    traj.add_step(
        Step(
            step_id=1,
            step_type=StepType.REASONING,
            content="I should search for the original Transformer paper.",
            timestamp=0.0,
        )
    )
    traj.add_step(
        Step(
            step_id=2,
            step_type=StepType.TOOL_CALL,
            timestamp=0.3,
            content="arxiv_search({'query': 'Attention is all you need'})",
            tool_name="arxiv_search",
            tool_input={"query": "Attention is all you need"},
            tool_output={"hits": 1},
            tool_success=True,
        )
    )
    traj.add_step(
        Step(
            step_id=3,
            step_type=StepType.REASONING,
            content="Found: Vaswani et al., NeurIPS 2017.",
            timestamp=0.6,
        )
    )
    traj.finalize()

    outcome = Outcome(
        answer="Vaswani et al., 2017",
        confidence=0.94,
        state={"primary_source": "https://arxiv.org/abs/1706.03762"},
        sources=["https://arxiv.org/abs/1706.03762", "https://nature.com/example"],
    )

    metrics = TrackedMetrics(n_turns=3, n_toolcalls=1, latency=0.62)

    return Trial(
        task_id="demo-1",
        trial_id=0,
        trajectory=traj,
        outcome=outcome,
        metrics=metrics,
    )


# ── 4. Run every grader and print a neat summary ──────────────────────────────


async def main() -> None:
    trial = build_trial()

    graders: List[BaseGrader] = [
        DeterministicTestsGrader(),
        StateCheckGrader(),
        ToolCallsGrader(),
    ]

    print(f"Q: {task.question}")
    print(f"A: {trial.outcome.answer}\n")
    print(f"{'Grader':<22}  {'Score':<6}  Passed  Reasoning")
    print("-" * 80)
    for g in graders:
        result = await g.grade(trial.outcome, trial.trajectory, task)
        print(
            f"{g.name:<22}  {result.score:<6.2f}  "
            f"{str(result.passed):<6}  {result.reasoning}"
        )


if __name__ == "__main__":
    asyncio.run(main())
