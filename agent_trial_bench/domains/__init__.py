"""
Domain plugin registry.

The evaluation framework is domain-agnostic at its core: every piece of
vertical-specific knowledge (regex validators, trusted sources, tool patterns,
scoring calibrations) lives in a ``DomainPlugin``.  Multiple plugins can be
active at the same time – their ``validators`` / ``trusted_sources`` /
``tool_patterns`` sets are merged on lookup.

Built-in plugins:

* ``general`` – URLs, emails, ISO dates, JSON, numeric ranges … (always on)
* ``web3``    – Ethereum addresses, tx hashes, ENS, etherscan/DeFi patterns

Third-party domains can register via :func:`register_domain`.
"""

from __future__ import annotations

from typing import Callable, Dict, Iterable, List, Optional, Set

from .base import DomainPlugin


# ── Registry ──────────────────────────────────────────────────────────────────


_REGISTRY: Dict[str, DomainPlugin] = {}


def register_domain(plugin: DomainPlugin) -> None:
    """Register (or replace) a domain plugin by name."""
    _REGISTRY[plugin.name.lower()] = plugin


def unregister_domain(name: str) -> None:
    _REGISTRY.pop(name.lower(), None)


def get_domain(name: str) -> Optional[DomainPlugin]:
    return _REGISTRY.get(name.lower())


def list_domains() -> List[str]:
    return sorted(_REGISTRY.keys())


def active_domains() -> List[DomainPlugin]:
    """Return all currently-registered domain plugins."""
    return list(_REGISTRY.values())


# ── Composite lookups ─────────────────────────────────────────────────────────


def get_validator(key: str) -> Optional[Callable[[str], bool]]:
    """Look up a named validator across all registered domains (first match)."""
    for plugin in _REGISTRY.values():
        fn = plugin.get_validator(key)
        if fn is not None:
            return fn
    return None


def all_trusted_sources() -> Set[str]:
    """Union of trusted-source allowlists from every registered domain."""
    merged: Set[str] = set()
    for plugin in _REGISTRY.values():
        merged.update(plugin.trusted_sources)
    return merged


def all_technical_terms() -> Set[str]:
    merged: Set[str] = set()
    for plugin in _REGISTRY.values():
        merged.update(plugin.technical_terms)
    return merged


def expected_tools_for(category: str) -> Set[str]:
    """Union of expected tools for ``category`` across every domain."""
    merged: Set[str] = set()
    for plugin in _REGISTRY.values():
        merged.update(plugin.get_expected_tools(category))
    return merged


def required_params_for(category: str) -> List[str]:
    merged: List[str] = []
    for plugin in _REGISTRY.values():
        for p in plugin.get_required_params(category):
            if p not in merged:
                merged.append(p)
    return merged


def validate_params_across_domains(params: Dict[str, object]) -> bool:
    """All registered domains must agree the params are well-formed."""
    return all(p.validate_params(dict(params)) for p in _REGISTRY.values())


def category_calibration_boost(category: str) -> float:
    for plugin in _REGISTRY.values():
        boost = plugin.category_calibration.get(category)
        if boost is not None:
            return boost
    return 0.0


def normalize_text(text: str) -> str:
    """Apply every domain's ``normalize`` hook in turn."""
    for plugin in _REGISTRY.values():
        text = plugin.normalize(text)
    return text


# ── Register built-ins on import ──────────────────────────────────────────────

from .general import GeneralDomain  # noqa: E402
from .web3 import Web3Domain  # noqa: E402

register_domain(GeneralDomain())
register_domain(Web3Domain())


__all__ = [
    "DomainPlugin",
    "register_domain",
    "unregister_domain",
    "get_domain",
    "list_domains",
    "active_domains",
    "get_validator",
    "all_trusted_sources",
    "all_technical_terms",
    "expected_tools_for",
    "required_params_for",
    "validate_params_across_domains",
    "category_calibration_boost",
    "normalize_text",
    "GeneralDomain",
    "Web3Domain",
]
