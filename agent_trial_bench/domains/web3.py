"""
Web3 domain plugin.

Bundles Ethereum / EVM-specific format validators, trusted on-chain sources
and tool patterns.  Consumed transparently by the graders and scoring system
via :mod:`agent_trial_bench.domains`.
"""

from __future__ import annotations

import re
from typing import Any, Dict

from .base import DomainPlugin


# ── Regex patterns ────────────────────────────────────────────────────────────

_ADDRESS_RE = re.compile(r"^0x[a-fA-F0-9]{40}$")
_TX_HASH_RE = re.compile(r"^0x[a-fA-F0-9]{64}$")
_ENS_RE = re.compile(r"^[a-zA-Z0-9-]+\.eth$")
_BYTES32_RE = re.compile(r"^0x[a-fA-F0-9]{64}$")


# ── Validators ────────────────────────────────────────────────────────────────


def validate_address(value: str) -> bool:
    return bool(_ADDRESS_RE.match(value.strip()))


def validate_tx_hash(value: str) -> bool:
    return bool(_TX_HASH_RE.match(value.strip()))


def validate_ens(value: str) -> bool:
    return bool(_ENS_RE.match(value.strip()))


def validate_bytes32(value: str) -> bool:
    return bool(_BYTES32_RE.match(value.strip()))


# ── Web3 tool patterns ────────────────────────────────────────────────────────


_WEB3_TOOL_PATTERNS: Dict[str, Dict[str, Any]] = {
    "onchain_retrieval": {
        "expected_tools": {
            "blockchain_query",
            "contract_read",
            "transaction_fetch",
            "etherscan_api",
            "get_block",
        },
        "required_params": ["chain_id", "address"],
    },
    "defi_analysis": {
        "expected_tools": {
            "price_query",
            "liquidity_check",
            "protocol_info",
            "dex_api",
        },
        "required_params": ["token_address", "protocol"],
    },
    "web_onchain_retrieval": {
        "expected_tools": {
            "web_search",
            "blockchain_query",
            "contract_read",
        },
        "required_params": ["query"],
    },
}


_TRUSTED_SOURCES = {
    "etherscan.io",
    "bscscan.com",
    "polygonscan.com",
    "arbiscan.io",
    "optimistic.etherscan.io",
    "ethereum.org",
    "eips.ethereum.org",
    "docs.uniswap.org",
    "compound.finance",
    "aave.com",
    "coindesk.com",
    "cointelegraph.com",
    "theblock.co",
    "defillama.com",
    "dune.com",
    "messari.io",
}


_TECHNICAL_TERMS = {
    "blockchain",
    "ethereum",
    "bitcoin",
    "contract",
    "transaction",
    "hash",
    "address",
    "token",
    "defi",
    "nft",
    "smart contract",
    "protocol",
    "consensus",
    "mining",
    "staking",
    "liquidity",
    "erc20",
    "erc721",
    "rpc",
}


class Web3Domain(DomainPlugin):
    """Ethereum / EVM validators and tool patterns."""

    name = "web3"

    validators = {
        "address": validate_address,
        "eth_address": validate_address,
        "tx_hash": validate_tx_hash,
        "transaction_hash": validate_tx_hash,
        "ens": validate_ens,
        "bytes32": validate_bytes32,
    }

    trusted_sources = _TRUSTED_SOURCES

    tool_patterns = _WEB3_TOOL_PATTERNS

    category_calibration = {
        "web_retrieval": 0.01,
        "web_onchain_retrieval": 0.0,
        "onchain_retrieval": -0.01,
    }

    technical_terms = _TECHNICAL_TERMS

    # ── Hooks ────────────────────────────────────────────────────────────────

    _ADDRESS_LOWERCASE_RE = re.compile(r"0x([a-fA-F0-9]{40})")

    def normalize(self, text: str) -> str:
        """Lowercase any hex address we detect (Web3 addresses are case-insensitive)."""
        return self._ADDRESS_LOWERCASE_RE.sub(lambda m: "0x" + m.group(1).lower(), text)

    def validate_params(self, params: Dict[str, Any]) -> bool:
        """Return ``False`` only when a value is *clearly* a malformed
        Ethereum address / tx hash / ENS name.

        We deliberately only look at strings that already *look* Web3-shaped
        (``0x`` prefix or ``.eth`` suffix) – otherwise a general agent passing
        ``{"street_address": "123 Main St"}`` would be reported as invalid.

        Any ``0x``-prefixed value is required to be either a well-formed
        address (42 chars) or tx hash / bytes32 (66 chars).  Values that are
        clearly not hex (e.g. ``0xnothash``) fail fast.
        """
        for _key, value in params.items():
            if not isinstance(value, str):
                continue
            stripped = value.strip()
            if stripped.startswith("0x"):
                if not (validate_address(stripped) or validate_tx_hash(stripped) or validate_bytes32(stripped)):
                    return False
            elif stripped.endswith(".eth"):
                if not validate_ens(stripped):
                    return False
        return True
