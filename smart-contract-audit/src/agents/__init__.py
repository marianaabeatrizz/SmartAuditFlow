"""Multi-agent smart contract audit.

Replaces the fixed five-node linear pipeline in ``src/workflow`` with a
hierarchical orchestrator, a parallel swarm of per-vulnerability specialists,
an adversarial validator and a ranking aggregator, wired as a LangGraph graph
with a bounded replanning loop.
"""
from src.agents.graph import build_graph
from src.agents.llm import LLMClient
from src.agents.state import AuditState, Finding, Task, new_state

__all__ = ["build_graph", "LLMClient", "AuditState", "Finding", "Task", "new_state"]
