"""
Backward-compatibility shim.

The Web3-specific validators and tool patterns have moved to
:mod:`agent_trial_bench.domains.web3`.  Any existing import of
``agent_trial_bench.graders.web3_validators`` continues to work; new code should use
``from agent_trial_bench.domains import get_validator, expected_tools_for, ...``
instead.
"""

from __future__ import annotations

import warnings
from typing import Any, Dict, Set

from ..domains import (
    expected_tools_for,
    validate_params_across_domains,
)
from ..domains.general import normalize_number  # re-export
from ..domains.web3 import (  # re-export
    validate_address,
    validate_bytes32,
    validate_ens,
    validate_tx_hash,
)

# Legacy constant – kept for anyone who imported it directly.  New code should
# iterate the ``Web3Domain`` plugin.
from ..domains.web3 import _WEB3_TOOL_PATTERNS as WEB3_TOOL_PATTERNS  # noqa: F401


def get_expected_tools(category: str) -> Set[str]:
    """Return the expected tool set for ``category`` across all domains.

    .. deprecated::
        Import :func:`agent_trial_bench.domains.expected_tools_for` instead.
    """
    return expected_tools_for(category)


def validate_web3_params(params: Dict[str, Any]) -> bool:
    """Validate that every Web3-shaped parameter is well-formed.

    .. deprecated::
        Use :func:`agent_trial_bench.domains.validate_params_across_domains`.
    """
    return validate_params_across_domains(params)


__all__ = [
    "validate_address",
    "validate_tx_hash",
    "validate_ens",
    "validate_bytes32",
    "validate_web3_params",
    "get_expected_tools",
    "normalize_number",
    "WEB3_TOOL_PATTERNS",
]


def __getattr__(name: str):  # pragma: no cover - runtime shim
    warnings.warn(
        "agent_trial_bench.graders.web3_validators is a compatibility shim; "
        "import from agent_trial_bench.domains instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    raise AttributeError(name)
