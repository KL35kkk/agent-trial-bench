"""
Enums for the Agent Trial Bench.

``DAB`` is now a **Domain-Agnostic Benchmark** – Web3 is supported as a
first-class domain plugin, but the core framework is general-purpose and can
evaluate any kind of agent (research, reasoning, coding, tool-use, …).
"""

from enum import Enum


class TaskCategory(Enum):
    """Task category enumeration.

    The framework is domain-agnostic: add a new value here or – preferably –
    register a ``DomainPlugin`` that understands a new category string.
    Values are lowercase snake_case so they can also be loaded directly from
    dataset files without writing a mapping.
    """

    # ── General-purpose categories ───────────────────────────────────────
    GENERAL = "general"
    FACT_QA = "fact_qa"
    RESEARCH = "research"
    REASONING = "reasoning"
    TOOL_USE = "tool_use"
    CODING = "coding"
    CONVERSATIONAL = "conversational"
    MULTI_STEP = "multi_step"

    # ── Web-centric (shared across many verticals) ───────────────────────
    WEB_RETRIEVAL = "web_retrieval"

    # ── Web3 domain specialisation ───────────────────────────────────────
    WEB_ONCHAIN_RETRIEVAL = "web_onchain_retrieval"
    ONCHAIN_RETRIEVAL = "onchain_retrieval"

    @classmethod
    def from_value(cls, value: str) -> "TaskCategory":
        """Lenient loader that falls back to ``GENERAL`` for unknown strings."""
        if value is None:
            return cls.GENERAL
        normalized = str(value).strip().lower()
        for member in cls:
            if member.value == normalized:
                return member
        return cls.GENERAL


class EvaluationMethod(Enum):
    """Evaluation method enumeration."""
    RULE_BASED = "rule_based"
    LLM_BASED = "llm_based"
    HYBRID = "hybrid"
    CASCADE = "cascade"


class EvaluationStatus(Enum):
    """Evaluation status enumeration."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
