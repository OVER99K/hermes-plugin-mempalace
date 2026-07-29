"""Minimal Hermes interface stub for running this public plugin's tests standalone."""

import sys
from types import ModuleType

try:
    from agent.memory_provider import MemoryProvider as _MemoryProvider  # noqa: F401
except ModuleNotFoundError:
    agent = ModuleType("agent")
    agent.__path__ = []
    memory_provider = ModuleType("agent.memory_provider")

    class MemoryProvider:
        pass

    setattr(memory_provider, "MemoryProvider", MemoryProvider)
    setattr(agent, "memory_provider", memory_provider)
    sys.modules.setdefault("agent", agent)
    sys.modules["agent.memory_provider"] = memory_provider
