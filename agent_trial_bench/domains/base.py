"""
Domain plugin base class.

A ``DomainPlugin`` bundles the small amount of domain-specific knowledge the
framework needs to grade agents in a particular vertical:

* ``validators``        – named format checkers (``url``, ``email``, ``address`` …)
* ``trusted_sources``   – allow-list of reputable domains for citation checks
* ``tool_patterns``     – expected tools / required params per task category
* ``category_calibration`` – tiny score offsets applied by ``EnhancedScoringSystem``
* ``technical_terms``   – vocabulary that signals domain professionalism
* ``normalize``         – optional text normalisation (e.g. lowercase hex addresses)
* ``validate_params``   – sanity-checks tool arguments for the domain

New domains can be added without touching the core evaluator by subclassing this
class and calling :func:`agent_trial_bench.domains.register_domain`.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Iterable, Optional, Set


class DomainPlugin:
    """Base class for domain plugins.

    Subclasses should either override the class-level attributes or override
    the corresponding ``get_*`` methods for dynamic behaviour.
    """

    name: str = "base"

    # Static, declarative knowledge -------------------------------------------
    validators: Dict[str, Callable[[str], bool]] = {}
    trusted_sources: Set[str] = set()
    tool_patterns: Dict[str, Dict[str, Any]] = {}
    category_calibration: Dict[str, float] = {}
    technical_terms: Set[str] = set()

    # Optional hooks -----------------------------------------------------------

    def normalize(self, text: str) -> str:  # pragma: no cover - default no-op
        """Domain-specific text normalisation; default is identity."""
        return text

    def validate_params(self, params: Dict[str, Any]) -> bool:
        """Validate tool arguments for domain-specific well-formedness.

        Return ``False`` only if the domain is *certain* a value is malformed;
        return ``True`` otherwise so multiple domain validators can be combined
        with a simple ``all(...)``.
        """
        return True

    # Convenience accessors ----------------------------------------------------

    def get_validator(self, key: str) -> Optional[Callable[[str], bool]]:
        return self.validators.get(key)

    def get_expected_tools(self, category: str) -> Set[str]:
        entry = self.tool_patterns.get(category.lower(), {})
        tools = entry.get("expected_tools", set())
        return set(tools)

    def get_required_params(self, category: str) -> Iterable[str]:
        entry = self.tool_patterns.get(category.lower(), {})
        return list(entry.get("required_params", []))
