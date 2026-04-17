"""
Tools package – mock tool registry for deterministic agent evaluation.
"""

from .mock_registry import (
    MockToolRegistry,
    MockToolResponse,
    ToolFailureMode,
    register_default_tools,
    register_general_tools,
    register_web3_tools,
)

__all__ = [
    "MockToolRegistry",
    "ToolFailureMode",
    "MockToolResponse",
    "register_default_tools",
    "register_general_tools",
    "register_web3_tools",
]
