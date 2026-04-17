"""
Scenario Generator – injects perturbations into tasks to simulate real-world
chaos and test agent robustness.

The generator wraps any HubTask and produces a ScenarioTask that carries
the original task plus injected perturbations.  The MockToolRegistry is
configured based on the perturbation type so the agent runner experiences
realistic failure conditions.

Supported perturbations (5 types):
  TOOL_FAILURE   – mock API timeout or error on one or more tools
  MISSING_INFO   – remove key context fields from the task
  CONFLICTING    – inject a contradictory fact into the context
  PRICE_CHANGE   – mid-run data staleness (price / value changes)
  USER_REDIRECT  – user changes their question mid-conversation
"""

from __future__ import annotations

import copy
import logging
import random
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Sequence

from .datasets.task_hub import HubTask, TaskTier
from .tools.mock_registry import MockToolRegistry, ToolFailureMode, register_default_tools

logger = logging.getLogger(__name__)


# ── perturbation enum ─────────────────────────────────────────────────────────


class Perturbation(Enum):
    TOOL_FAILURE = "tool_failure"
    MISSING_INFO = "missing_info"
    CONFLICTING = "conflicting"
    PRICE_CHANGE = "price_change"
    USER_REDIRECT = "user_redirect"


# ── scenario task ─────────────────────────────────────────────────────────────


@dataclass
class ScenarioTask:
    """A HubTask augmented with scenario perturbation metadata.

    Attributes:
        base_task:        The original unmodified task.
        perturbed_task:   A deep copy of ``base_task`` with injected context.
        perturbations:    Which perturbation types were applied.
        injected_context: What was injected / removed.
        mock_registry:    Pre-configured MockToolRegistry for this scenario.
        expected_recovery: Whether the agent is expected to recover gracefully.
        difficulty_delta:  How much harder this scenario is (+1 per perturbation).
    """

    base_task: HubTask
    perturbed_task: HubTask
    perturbations: List[Perturbation] = field(default_factory=list)
    injected_context: Dict[str, Any] = field(default_factory=dict)
    mock_registry: Optional[MockToolRegistry] = None
    expected_recovery: bool = True
    difficulty_delta: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "base_task_id": self.base_task.task_id,
            "perturbed_task_id": self.perturbed_task.task_id,
            "perturbations": [p.value for p in self.perturbations],
            "injected_context": self.injected_context,
            "expected_recovery": self.expected_recovery,
            "difficulty_delta": self.difficulty_delta,
        }


# ── generator ─────────────────────────────────────────────────────────────────


class ScenarioGenerator:
    """Generates perturbed scenario tasks from base HubTasks.

    Args:
        seed: RNG seed for reproducibility.
        cache_dir: Passed through to MockToolRegistry for replay caching.
    """

    def __init__(
        self,
        seed: Optional[int] = None,
        cache_dir: Optional[str] = None,
    ) -> None:
        self._rng = random.Random(seed)
        self._cache_dir = cache_dir

    # ── public API ────────────────────────────────────────────────────────────

    def generate(
        self,
        task: HubTask,
        perturbations: Sequence[Perturbation],
    ) -> ScenarioTask:
        """Apply a list of perturbations to ``task`` and return a ScenarioTask."""
        perturbed = copy.deepcopy(task)
        perturbed.task_id = f"{task.task_id}_scenario"
        perturbed.tier = TaskTier.SYNTHETIC
        perturbed.perturbations = [p.value for p in perturbations]

        registry = MockToolRegistry(cache_dir=self._cache_dir)
        register_default_tools(registry)

        injected: Dict[str, Any] = {}
        difficulty_delta = 0

        for p in perturbations:
            delta, ctx = self._apply(p, perturbed, registry)
            injected.update(ctx)
            difficulty_delta += delta

        perturbed.injected_context = injected

        expected_recovery = difficulty_delta <= 2  # 3+ perturbations → probably unrecoverable

        return ScenarioTask(
            base_task=task,
            perturbed_task=perturbed,
            perturbations=list(perturbations),
            injected_context=injected,
            mock_registry=registry,
            expected_recovery=expected_recovery,
            difficulty_delta=difficulty_delta,
        )

    def generate_batch(
        self,
        tasks: Sequence[HubTask],
        perturbations_per_task: Sequence[Sequence[Perturbation]],
    ) -> List[ScenarioTask]:
        """Generate scenarios for multiple tasks.

        Args:
            tasks: Base tasks.
            perturbations_per_task: One list of perturbations per task.
                If lengths differ, extras are ignored / missing entries use [].
        """
        scenarios: List[ScenarioTask] = []
        for i, task in enumerate(tasks):
            perts = perturbations_per_task[i] if i < len(perturbations_per_task) else []
            scenarios.append(self.generate(task, perts))
        return scenarios

    def generate_all_combinations(
        self,
        task: HubTask,
        perturbation_pool: Optional[List[Perturbation]] = None,
        max_combinations: int = 10,
    ) -> List[ScenarioTask]:
        """Exhaustively (or randomly up to ``max_combinations``) generate
        single-perturbation and two-perturbation scenarios from the pool."""
        from itertools import combinations

        pool = perturbation_pool or list(Perturbation)
        combos: List[tuple] = []
        for size in (1, 2):
            combos.extend(combinations(pool, size))

        if len(combos) > max_combinations:
            combos = self._rng.sample(combos, max_combinations)

        return [self.generate(task, list(combo)) for combo in combos]

    # ── perturbation handlers ─────────────────────────────────────────────────

    def _apply(
        self,
        perturbation: Perturbation,
        task: HubTask,
        registry: MockToolRegistry,
    ) -> tuple[int, Dict[str, Any]]:
        """Apply a single perturbation.  Returns (difficulty_delta, injected_ctx)."""
        handlers = {
            Perturbation.TOOL_FAILURE: self._apply_tool_failure,
            Perturbation.MISSING_INFO: self._apply_missing_info,
            Perturbation.CONFLICTING: self._apply_conflicting,
            Perturbation.PRICE_CHANGE: self._apply_price_change,
            Perturbation.USER_REDIRECT: self._apply_user_redirect,
        }
        return handlers[perturbation](task, registry)

    def _apply_tool_failure(
        self, task: HubTask, registry: MockToolRegistry
    ) -> tuple[int, Dict[str, Any]]:
        """Inject a tool failure for one of the expected tools."""
        failure_modes = [
            ToolFailureMode.TIMEOUT,
            ToolFailureMode.HTTP_ERROR,
            ToolFailureMode.RATE_LIMIT,
            ToolFailureMode.EMPTY_RESULT,
        ]
        mode = self._rng.choice(failure_modes)

        # Pick a tool to fail based on the task category.  We consult the
        # domain registry so that Web3, general-purpose and any third-party
        # domain plugins all contribute candidate tools.
        from .domains import expected_tools_for

        candidates = sorted(expected_tools_for(task.category)) or [
            "web_search",
            "document_fetch",
        ]
        target_tool = self._rng.choice(candidates)

        registry.inject_failure(target_tool, mode)

        ctx = {
            "tool_failure": {
                "tool": target_tool,
                "mode": mode.value,
            }
        }
        return 1, ctx

    def _apply_missing_info(
        self, task: HubTask, registry: MockToolRegistry
    ) -> tuple[int, Dict[str, Any]]:
        """Remove one context key to simulate incomplete information."""
        removable = [k for k in task.context if k not in ("question", "category")]
        if not removable:
            return 0, {}

        key = self._rng.choice(removable)
        removed_value = task.context.pop(key, None)

        ctx = {"missing_info": {"removed_key": key, "original_value": str(removed_value)[:100]}}
        return 1, ctx

    def _apply_conflicting(
        self, task: HubTask, registry: MockToolRegistry
    ) -> tuple[int, Dict[str, Any]]:
        """Inject a contradictory fact into the context."""
        if task.expected_answer:
            # Insert a plausible but wrong alternative answer hint
            fake = _generate_conflicting_value(task.expected_answer, self._rng)
            task.context["conflicting_hint"] = fake
            ctx = {"conflicting": {"injected_hint": fake, "correct_answer": task.expected_answer}}
        else:
            task.context["conflicting_hint"] = "Data unavailable from primary source."
            ctx = {"conflicting": {"injected_hint": task.context["conflicting_hint"]}}

        return 2, ctx  # Higher difficulty – agent must resolve contradiction

    def _apply_price_change(
        self, task: HubTask, registry: MockToolRegistry
    ) -> tuple[int, Dict[str, Any]]:
        """Simulate mid-run data change (e.g. price moved)."""
        # Override price_query to return a different price on subsequent calls
        call_count = {"n": 0}
        initial_price = round(self._rng.uniform(100, 50000), 2)
        changed_price = round(initial_price * self._rng.uniform(0.85, 1.15), 2)

        def dynamic_price_query(
            token_address: str = "",
            token_symbol: str = "",
            currency: str = "USD",
            **_: Any,
        ) -> Dict[str, Any]:
            call_count["n"] += 1
            price = initial_price if call_count["n"] == 1 else changed_price
            return {
                "token": token_symbol or token_address,
                "price_usd": price,
                "currency": currency,
                "note": "price_changed" if call_count["n"] > 1 else "initial",
            }

        registry.register("price_query", dynamic_price_query)

        ctx = {
            "price_change": {
                "initial_price": initial_price,
                "changed_price": changed_price,
                "pct_change": round((changed_price - initial_price) / initial_price * 100, 2),
            }
        }
        return 1, ctx

    def _apply_user_redirect(
        self, task: HubTask, registry: MockToolRegistry
    ) -> tuple[int, Dict[str, Any]]:
        """Modify the question to simulate a user changing their requirement."""
        original_question = task.question
        redirected = _redirect_question(original_question, self._rng)
        task.question = redirected
        task.context["original_question"] = original_question
        task.context["user_redirect"] = True

        ctx = {
            "user_redirect": {
                "original_question": original_question,
                "redirected_question": redirected,
            }
        }
        return 2, ctx


# ── helpers ───────────────────────────────────────────────────────────────────


def _generate_conflicting_value(correct: str, rng: random.Random) -> str:
    """Generate a plausible-looking but wrong alternative for ``correct``."""
    import re

    # Date: shift by a random number of days
    date_match = re.search(r"(\d{4})[/-](\d{1,2})[/-](\d{1,2})", correct)
    if date_match:
        year, month, day = date_match.groups()
        fake_day = (int(day) % 28) + 1
        return correct.replace(f"{year}/{month}/{day}", f"{year}/{month}/{fake_day}")

    # Number: add random offset
    num_match = re.search(r"\b(\d{5,})\b", correct)
    if num_match:
        original = int(num_match.group(1))
        fake = original + rng.randint(1000, 100000)
        return correct.replace(num_match.group(1), str(fake))

    # Generic: append a noise suffix
    noise_suffixes = [" (unverified)", " (estimate)", " (approximately)", " (reported)"]
    return correct + rng.choice(noise_suffixes)


def _redirect_question(question: str, rng: random.Random) -> str:
    """Produce a related but different question from the original."""
    redirects = [
        lambda q: f"Actually, ignore the previous question. Instead: {q} And also explain the risks.",
        lambda q: q.replace("日期", "时间区间") if "日期" in q else q + " (specifically for Q4)",
        lambda q: f"What is the CURRENT status of: {q}",
        lambda q: f"Compare and contrast: {q} vs its main alternative.",
    ]
    transform = rng.choice(redirects)
    try:
        return transform(question)
    except Exception:
        return question + " (Please also provide sources.)"
