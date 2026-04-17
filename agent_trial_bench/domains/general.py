"""
General-purpose domain plugin.

Covers format validators (URLs, emails, ISO dates, UUIDs, JSON, numbers) and
common tool patterns / trusted sources that apply to most agent evaluations.

Web3 specifics live in :mod:`agent_trial_bench.domains.web3`.
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, Optional

from .base import DomainPlugin


# ── Regex patterns ────────────────────────────────────────────────────────────

_URL_RE = re.compile(r"^https?://[^\s]+$", re.IGNORECASE)
_EMAIL_RE = re.compile(r"^[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}$")
_ISO_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}(?:[T ]\d{2}:\d{2}(?::\d{2})?(?:Z|[+-]\d{2}:?\d{2})?)?$")
_UUID_RE = re.compile(r"^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$")
_INTEGER_RE = re.compile(r"^-?\d+$")
_FLOAT_RE = re.compile(r"^-?\d+(?:\.\d+)?$")
_PHONE_E164_RE = re.compile(r"^\+?[1-9]\d{6,14}$")


# ── Validators ────────────────────────────────────────────────────────────────


def validate_url(value: str) -> bool:
    return bool(_URL_RE.match(value.strip()))


def validate_email(value: str) -> bool:
    return bool(_EMAIL_RE.match(value.strip()))


def validate_iso_date(value: str) -> bool:
    return bool(_ISO_DATE_RE.match(value.strip()))


def validate_uuid(value: str) -> bool:
    return bool(_UUID_RE.match(value.strip()))


def validate_integer(value: str) -> bool:
    return bool(_INTEGER_RE.match(value.strip()))


def validate_float(value: str) -> bool:
    return bool(_FLOAT_RE.match(value.strip()))


def validate_phone(value: str) -> bool:
    return bool(_PHONE_E164_RE.match(value.strip()))


def validate_json(value: str) -> bool:
    try:
        json.loads(value)
        return True
    except (ValueError, TypeError):
        return False


def normalize_number(value: str) -> Optional[float]:
    """Strip common separators; return numeric value or ``None``."""
    try:
        cleaned = str(value).replace(",", "").replace("_", "").replace(" ", "")
        return float(cleaned) if "." in cleaned else float(int(cleaned))
    except (ValueError, TypeError):
        return None


# ── Tool patterns for common agent tasks ──────────────────────────────────────


_TOOL_PATTERNS: Dict[str, Dict[str, Any]] = {
    "general": {
        "expected_tools": set(),
        "required_params": [],
    },
    "fact_qa": {
        "expected_tools": {"web_search", "document_fetch", "knowledge_lookup"},
        "required_params": ["query"],
    },
    "research": {
        "expected_tools": {"web_search", "document_fetch", "summarize", "api_call"},
        "required_params": ["query"],
    },
    "reasoning": {
        "expected_tools": {"calculator", "python_exec", "knowledge_lookup"},
        "required_params": [],
    },
    "tool_use": {
        "expected_tools": {"web_search", "api_call", "calculator", "document_fetch"},
        "required_params": [],
    },
    "coding": {
        "expected_tools": {"python_exec", "code_search", "file_read", "file_write"},
        "required_params": [],
    },
    "conversational": {
        "expected_tools": set(),
        "required_params": [],
    },
    "multi_step": {
        "expected_tools": {"web_search", "api_call", "calculator"},
        "required_params": [],
    },
    # Common back-compat alias used by earlier datasets
    "web_retrieval": {
        "expected_tools": {"web_search", "api_call", "document_fetch"},
        "required_params": ["query"],
    },
}


_TRUSTED_SOURCES = {
    "wikipedia.org",
    "arxiv.org",
    "nature.com",
    "sciencemag.org",
    "sciencedirect.com",
    "nih.gov",
    "who.int",
    "un.org",
    "reuters.com",
    "apnews.com",
    "bbc.com",
    "bbc.co.uk",
    "nytimes.com",
    "theguardian.com",
    "ft.com",
    "bloomberg.com",
    "github.com",
    "stackoverflow.com",
    "docs.python.org",
    "developer.mozilla.org",
    "ieee.org",
    "acm.org",
}


_TECHNICAL_TERMS = {
    "algorithm",
    "function",
    "variable",
    "api",
    "endpoint",
    "schema",
    "dataset",
    "latency",
    "throughput",
    "metric",
    "benchmark",
    "model",
    "inference",
    "training",
    "hypothesis",
    "evidence",
    "methodology",
}


class GeneralDomain(DomainPlugin):
    """Cross-domain validators, trusted sources and tool patterns."""

    name = "general"

    validators = {
        "url": validate_url,
        "email": validate_email,
        "iso_date": validate_iso_date,
        "uuid": validate_uuid,
        "integer": validate_integer,
        "float": validate_float,
        "number": validate_float,
        "phone": validate_phone,
        "json": validate_json,
    }

    trusted_sources = _TRUSTED_SOURCES

    tool_patterns = _TOOL_PATTERNS

    category_calibration = {
        "general": 0.0,
        "fact_qa": 0.0,
        "research": 0.0,
        "reasoning": 0.0,
        "tool_use": 0.0,
        "coding": 0.0,
        "conversational": 0.0,
        "multi_step": 0.0,
    }

    technical_terms = _TECHNICAL_TERMS
