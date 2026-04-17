"""
Mock Tool Registry – deterministic, replayable tool execution for evaluation.

Key design goals:
1. **Determinism**: identical (tool_name, args) → identical response across runs.
2. **Failure injection**: inject timeout / wrong_data / empty_result at will.
3. **Replay**: every call is persisted to a JSON cache keyed by
   ``(tool_name, args_hash)``, enabling full re-run of any past scenario.
4. **Integration with mock:// agent**: the agent runner checks
   ``context["mock_tools"]`` and routes tool calls through this registry.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


# ── failure modes ─────────────────────────────────────────────────────────────


class ToolFailureMode(Enum):
    NONE = "none"
    TIMEOUT = "timeout"
    WRONG_DATA = "wrong_data"
    EMPTY_RESULT = "empty_result"
    HTTP_ERROR = "http_error"
    RATE_LIMIT = "rate_limit"


@dataclass
class MockToolResponse:
    """Encapsulates a mock tool's result."""

    tool_name: str
    args: Dict[str, Any]
    result: Any
    success: bool = True
    failure_mode: ToolFailureMode = ToolFailureMode.NONE
    latency_ms: float = 0.0
    cached: bool = False
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "tool_name": self.tool_name,
            "args": self.args,
            "result": self.result,
            "success": self.success,
            "failure_mode": self.failure_mode.value,
            "latency_ms": self.latency_ms,
            "cached": self.cached,
            "timestamp": self.timestamp,
        }


# ── registry ──────────────────────────────────────────────────────────────────


class MockToolRegistry:
    """Central registry that maps tool names to mock implementations.

    Args:
        cache_dir: Directory for persisting call recordings (enables replay).
            Set to ``None`` to disable persistence.
        default_latency_ms: Simulated network latency for all calls.
    """

    def __init__(
        self,
        cache_dir: Optional[str | Path] = None,
        default_latency_ms: float = 50.0,
    ) -> None:
        self._tools: Dict[str, Callable] = {}
        self._failure_overrides: Dict[str, ToolFailureMode] = {}
        self._call_log: List[MockToolResponse] = []
        self.default_latency_ms = default_latency_ms

        if cache_dir is not None:
            self._cache_dir = Path(cache_dir)
            self._cache_dir.mkdir(parents=True, exist_ok=True)
        else:
            self._cache_dir = None

    # ── registration ──────────────────────────────────────────────────────────

    def register(self, name: str, fn: Callable) -> "MockToolRegistry":
        """Register a mock implementation for a tool."""
        self._tools[name] = fn
        return self

    def unregister(self, name: str) -> "MockToolRegistry":
        self._tools.pop(name, None)
        return self

    def inject_failure(
        self,
        tool_name: str,
        mode: ToolFailureMode,
    ) -> "MockToolRegistry":
        """Force a specific failure mode for ``tool_name`` on the next call."""
        self._failure_overrides[tool_name] = mode
        return self

    def clear_failures(self) -> "MockToolRegistry":
        self._failure_overrides.clear()
        return self

    # ── execution ─────────────────────────────────────────────────────────────

    async def call(
        self,
        tool_name: str,
        args: Dict[str, Any],
        *,
        force_failure: Optional[ToolFailureMode] = None,
    ) -> MockToolResponse:
        """Execute a tool call, applying any injected failure modes."""
        start = time.time()

        failure_mode = force_failure or self._failure_overrides.pop(tool_name, ToolFailureMode.NONE)

        # Simulate latency
        if self.default_latency_ms > 0:
            await asyncio.sleep(self.default_latency_ms / 1000)

        # Handle failure modes
        if failure_mode == ToolFailureMode.TIMEOUT:
            resp = MockToolResponse(
                tool_name=tool_name,
                args=args,
                result=None,
                success=False,
                failure_mode=failure_mode,
                latency_ms=(time.time() - start) * 1000,
            )
            self._record(resp)
            return resp

        if failure_mode == ToolFailureMode.EMPTY_RESULT:
            resp = MockToolResponse(
                tool_name=tool_name,
                args=args,
                result={},
                success=True,
                failure_mode=failure_mode,
                latency_ms=(time.time() - start) * 1000,
            )
            self._record(resp)
            return resp

        if failure_mode == ToolFailureMode.HTTP_ERROR:
            resp = MockToolResponse(
                tool_name=tool_name,
                args=args,
                result={"error": "HTTP 500 Internal Server Error"},
                success=False,
                failure_mode=failure_mode,
                latency_ms=(time.time() - start) * 1000,
            )
            self._record(resp)
            return resp

        if failure_mode == ToolFailureMode.RATE_LIMIT:
            resp = MockToolResponse(
                tool_name=tool_name,
                args=args,
                result={"error": "Rate limit exceeded (429)"},
                success=False,
                failure_mode=failure_mode,
                latency_ms=(time.time() - start) * 1000,
            )
            self._record(resp)
            return resp

        # Check persistent cache first (replay path)
        cache_key = _cache_key(tool_name, args)
        cached = self._load_cache(cache_key)
        if cached is not None:
            cached.cached = True
            return cached

        # Execute mock implementation
        fn = self._tools.get(tool_name)
        if fn is None:
            logger.warning("No mock registered for tool '%s'", tool_name)
            result = {"error": f"Tool '{tool_name}' not registered in MockToolRegistry"}
            success = False
        else:
            try:
                if asyncio.iscoroutinefunction(fn):
                    result = await fn(**args)
                else:
                    result = fn(**args)
                success = True

                if failure_mode == ToolFailureMode.WRONG_DATA and isinstance(result, dict):
                    result = _corrupt_result(result)
            except Exception as exc:
                logger.warning("Mock tool '%s' raised: %s", tool_name, exc)
                result = {"error": str(exc)}
                success = False

        resp = MockToolResponse(
            tool_name=tool_name,
            args=args,
            result=result,
            success=success,
            failure_mode=failure_mode,
            latency_ms=(time.time() - start) * 1000,
        )
        self._save_cache(cache_key, resp)
        self._record(resp)
        return resp

    # ── call log ─────────────────────────────────────────────────────────────

    @property
    def call_log(self) -> List[MockToolResponse]:
        return list(self._call_log)

    def clear_log(self) -> None:
        self._call_log.clear()

    # ── cache helpers ─────────────────────────────────────────────────────────

    def _load_cache(self, key: str) -> Optional[MockToolResponse]:
        if self._cache_dir is None:
            return None
        path = self._cache_dir / f"{key}.json"
        if not path.exists():
            return None
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            return MockToolResponse(
                tool_name=data["tool_name"],
                args=data["args"],
                result=data["result"],
                success=data["success"],
                failure_mode=ToolFailureMode(data.get("failure_mode", "none")),
                latency_ms=data.get("latency_ms", 0.0),
                timestamp=data.get("timestamp", 0.0),
            )
        except Exception as exc:
            logger.debug("Cache load failed for %s: %s", key, exc)
            return None

    def _save_cache(self, key: str, resp: MockToolResponse) -> None:
        if self._cache_dir is None:
            return
        path = self._cache_dir / f"{key}.json"
        try:
            path.write_text(json.dumps(resp.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
        except Exception as exc:
            logger.debug("Cache save failed for %s: %s", key, exc)

    def _record(self, resp: MockToolResponse) -> None:
        self._call_log.append(resp)


# ── general-purpose tool implementations ─────────────────────────────────────


def _web_search(query: str, num_results: int = 5, **_: Any) -> Dict[str, Any]:
    """Stub web search – returns fixed placeholder results."""
    return {
        "query": query,
        "results": [
            {
                "title": f"Result {i + 1} for: {query}",
                "url": f"https://example.com/result/{i + 1}",
                "snippet": f"This is a mock search result snippet for '{query}'.",
            }
            for i in range(min(num_results, 3))
        ],
    }


def _document_fetch(url: str, **_: Any) -> Dict[str, Any]:
    return {
        "url": url,
        "content": f"Mock content fetched from {url}",
        "status_code": 200,
    }


def _knowledge_lookup(topic: str, **_: Any) -> Dict[str, Any]:
    return {
        "topic": topic,
        "summary": f"Mock knowledge summary about {topic}.",
        "source": "mock_knowledge_base",
    }


def _calculator(expression: str, **_: Any) -> Dict[str, Any]:
    """Safe arithmetic evaluator (no names, no calls)."""
    allowed = set("0123456789+-*/(). %")
    if not set(expression) <= allowed:
        return {"expression": expression, "error": "disallowed characters"}
    try:
        # Restricted eval – only numbers and basic operators remain.
        result = eval(expression, {"__builtins__": {}}, {})  # noqa: S307
        return {"expression": expression, "result": result}
    except Exception as exc:
        return {"expression": expression, "error": str(exc)}


def _python_exec(code: str, **_: Any) -> Dict[str, Any]:
    """Placeholder: mock a python executor without actually running code."""
    return {
        "code": code,
        "stdout": f"[mock] executed {len(code)} characters",
        "exit_code": 0,
    }


def _api_call(url: str = "", method: str = "GET", **_: Any) -> Dict[str, Any]:
    return {
        "url": url,
        "method": method,
        "status_code": 200,
        "body": {"mock": True},
    }


def _summarize(text: str = "", **_: Any) -> Dict[str, Any]:
    head = (text or "")[:120]
    return {"summary": f"Mock summary of: {head}…" if head else "Mock summary."}


# ── Web3-specific tool implementations ──────────────────────────────────────


def _blockchain_query(
    chain_id: str = "1",
    method: str = "eth_blockNumber",
    params: Optional[List] = None,
    **_: Any,
) -> Dict[str, Any]:
    """Stub blockchain RPC call."""
    mock_responses = {
        "eth_blockNumber": {"result": "0x12A05F2"},  # ~19529202
        "eth_chainId": {"result": hex(int(chain_id))},
        "eth_getBalance": {"result": "0xDE0B6B3A7640000"},
        "eth_getTransactionByHash": {
            "result": {
                "hash": "0xabcdef" + "0" * 58,
                "blockNumber": "0x12A05F2",
                "from": "0x" + "1" * 40,
                "to": "0x" + "2" * 40,
                "value": "0xDE0B6B3A7640000",
            }
        },
    }
    return mock_responses.get(method, {"result": None, "error": f"Unknown method: {method}"})


def _contract_read(
    address: str,
    function: str = "name",
    chain_id: str = "1",
    **_: Any,
) -> Dict[str, Any]:
    return {
        "address": address,
        "function": function,
        "chain_id": chain_id,
        "result": f"MockContractResult({function})",
    }


def _etherscan_api(
    module: str = "contract",
    action: str = "getsourcecode",
    address: str = "",
    **_: Any,
) -> Dict[str, Any]:
    return {
        "status": "1",
        "message": "OK",
        "result": [
            {
                "ContractName": "MockContract",
                "SourceCode": "// mock source",
                "ABI": "[]",
                "Proxy": "0",
                "Implementation": "",
            }
        ],
    }


def _price_query(
    token_address: str = "",
    token_symbol: str = "",
    currency: str = "USD",
    **_: Any,
) -> Dict[str, Any]:
    return {
        "token": token_symbol or token_address,
        "price_usd": 1.0,
        "currency": currency,
        "timestamp": int(time.time()),
        "source": "mock",
    }


# ── registration helpers ─────────────────────────────────────────────────────


def register_general_tools(registry: MockToolRegistry) -> MockToolRegistry:
    """Register domain-agnostic tool mocks (search / fetch / calculator / …)."""
    registry.register("web_search", _web_search)
    registry.register("document_fetch", _document_fetch)
    registry.register("knowledge_lookup", _knowledge_lookup)
    registry.register("calculator", _calculator)
    registry.register("python_exec", _python_exec)
    registry.register("api_call", _api_call)
    registry.register("summarize", _summarize)
    return registry


def register_web3_tools(registry: MockToolRegistry) -> MockToolRegistry:
    """Register Web3-flavoured tool mocks (RPC, contract, explorer, price)."""
    registry.register("blockchain_query", _blockchain_query)
    registry.register("contract_read", _contract_read)
    registry.register("etherscan_api", _etherscan_api)
    registry.register("price_query", _price_query)
    # Aliases commonly seen in Web3 datasets.
    registry.register("transaction_fetch", _blockchain_query)
    registry.register("get_block", _blockchain_query)
    return registry


def register_default_tools(
    registry: MockToolRegistry,
    *,
    include_web3: bool = True,
) -> MockToolRegistry:
    """Register the default set of tool mocks.

    General-purpose tools are always registered.  Web3 tools are included by
    default for back-compat, but callers running purely non-Web3 evaluations
    can pass ``include_web3=False``.
    """
    register_general_tools(registry)
    if include_web3:
        register_web3_tools(registry)
    return registry


# ── helpers ───────────────────────────────────────────────────────────────────


def _cache_key(tool_name: str, args: Dict[str, Any]) -> str:
    payload = json.dumps({"tool": tool_name, "args": args}, sort_keys=True, ensure_ascii=False)
    return hashlib.md5(payload.encode()).hexdigest()


def _corrupt_result(result: Dict[str, Any]) -> Dict[str, Any]:
    """Subtly corrupt a dict result for wrong_data failure testing."""
    corrupted = dict(result)
    for k, v in corrupted.items():
        if isinstance(v, str) and len(v) > 2:
            corrupted[k] = v[:-2] + "XX"
            break
        if isinstance(v, (int, float)):
            corrupted[k] = v + 1
            break
    return corrupted
