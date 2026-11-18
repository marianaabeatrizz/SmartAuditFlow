"""Drop-in replacement for ``WorkflowEngine`` backed by the agent graph.

Deliberately mirrors the legacy constructor and ``execute_workflow_sync``
signature so ``audit_api.py`` can switch engines without the route code
changing shape, and so the two can be run side by side for comparison.
"""
from __future__ import annotations

import logging
import time
from typing import Any, Callable, Dict, List, Optional

from src.agents.aggregator import to_legacy_report
from src.agents.graph import build_graph
from src.agents.llm import LLMClient, role_model
from src.agents.state import new_state
from src.storage.memory import storage

logger = logging.getLogger(__name__)

# Coarse progress markers. The graph is not a fixed-length pipeline any more,
# so these are reported as phases reached rather than as steps completed.
PHASES = {
    "context": (1, "Context Briefing", 15),
    "plan": (2, "Planning & Delegation", 30),
    "specialist": (3, "Specialist Audit", 55),
    "validate": (4, "Adversarial Validation", 72),
    "refine": (5, "Refinement", 82),
    "replan": (6, "Coverage Review", 92),
    "aggregate": (7, "Ranking & Report", 98),
}


class MultiAgentEngine:
    """Runs the audit graph for one session."""

    def __init__(
        self,
        session_id: str,
        model_choice: str = "auto",
        specialties: Optional[List[str]] = None,
        max_rounds: int = 2,
        max_refine_rounds: int = 1,
        refine_uncertain: bool = True,
        strict: bool = False,
        judge_models: Optional[List[str]] = None,
        panel_size: int = 1,
        strict_independence: bool = False,
    ):
        self.session_id = session_id
        self.model_choice = model_choice
        self.max_rounds = max_rounds
        self.max_refine_rounds = max_refine_rounds
        self.refine_uncertain = refine_uncertain
        # Interactive runs default to non-strict so a single provider hiccup
        # degrades instead of failing the session. Benchmark runs pass
        # strict=True, where silent substitution would invalidate the result.
        self.client = LLMClient(strict=strict)
        self.specialties = specialties
        self._model_override = None if model_choice == "auto" else model_choice
        self.panel_size = panel_size
        self.strict_independence = strict_independence
        self.graph = build_graph(
            client=self.client,
            specialties=specialties,
            strict=strict,
            model=self._model_override,
            judge_models=judge_models,
            panel_size=panel_size,
            strict_independence=strict_independence,
        )

    def execute_workflow_sync(
        self,
        code_snippet: str,
        static_tool: str = "",
        progress_callback: Optional[Callable] = None,
    ) -> Dict[str, Any]:
        started = time.time()
        storage.update_session_status(self.session_id, "running")

        state = new_state(
            code_snippet=code_snippet,
            static_tool=static_tool,
            session_id=self.session_id,
            max_rounds=self.max_rounds,
            max_refine_rounds=self.max_refine_rounds,
            refine_uncertain=self.refine_uncertain,
        )

        try:
            merged: Dict[str, Any] = dict(state)
            # Two stream modes at once: "updates" names the node that just
            # finished (so we can report progress without hard-coding the
            # graph's shape), while "values" carries the properly *reduced*
            # state. Folding the updates by hand would be wrong -- it would
            # overwrite the accumulator fields instead of applying their
            # reducers, so token_usage would report only the last node's cost.
            for mode, chunk in self.graph.stream(
                state, stream_mode=["updates", "values"]
            ):
                if mode == "updates":
                    for node_name in chunk:
                        self._report(node_name, progress_callback)
                    # The API can stop a session at any point.
                    session = storage.get_session(self.session_id)
                    if session and session.get("status") == "stopped":
                        logger.info("session %s stopped by user", self.session_id)
                        return {"status": "stopped", "message": "Audit stopped by user."}
                else:
                    merged = chunk

            report = to_legacy_report(merged)
            elapsed = time.time() - started

            report["model_used"] = self.model_choice
            report["ExecutionSummary"]["total_execution_time"] = elapsed
            report["ExecutionSummary"]["session_id"] = self.session_id
            report["ExecutionSummary"]["architecture"] = "multi-agent"
            report["ExecutionSummary"]["refine_rounds"] = merged.get("refine_round", 0)

            storage.save_result(
                session_id=self.session_id,
                findings=report["Findings"],
                full_report=report.get("ContextBrief", ""),
                model_used=self.model_choice,
                execution_summary=report["ExecutionSummary"],
                execution_time=elapsed,
            )

            if progress_callback:
                progress_callback(
                    step=6, name="Completed", percentage=100, status="completed"
                )
            storage.update_session_status(self.session_id, "completed")
            logger.info(
                "session %s complete: %s findings in %.1fs",
                self.session_id, report["FindingNumber"], elapsed,
            )
            return report

        except Exception as exc:  # noqa: BLE001
            elapsed = time.time() - started
            logger.error("session %s failed: %s", self.session_id, exc, exc_info=True)
            storage.update_session_status(self.session_id, "failed")
            storage.save_result(
                session_id=self.session_id,
                findings=[{
                    "Issue": "Audit Execution Error",
                    "Severity": "High",
                    "Description": (
                        f"The multi-agent audit failed: {exc}. "
                        "This error appears in the auditing result only."
                    ),
                    "Impact": "The security analysis did not complete.",
                    "Location": "MultiAgentEngine",
                }],
                full_report=f"Execution failed: {exc}",
                model_used=self.model_choice,
                execution_summary={"error": True, "architecture": "multi-agent"},
                execution_time=elapsed,
            )
            raise

    def _report(self, node_name: str, callback: Optional[Callable]) -> None:
        if not callback or node_name not in PHASES:
            return
        step, name, pct = PHASES[node_name]
        try:
            callback(step=step, name=name, percentage=pct)
        except Exception:  # noqa: BLE001
            logger.debug("progress callback raised; continuing", exc_info=True)

    def describe(self) -> Dict[str, Any]:
        """What this run is configured to do -- useful for the API and logs."""
        from src.agents.specialists import DEFAULT_SPECIALTIES

        return {
            "architecture": "multi-agent",
            "specialists": self.specialties or DEFAULT_SPECIALTIES,
            "max_rounds": self.max_rounds,
            "refinement": {
                "max_refine_rounds": self.max_refine_rounds,
                "refine_uncertain": self.refine_uncertain,
            },
            "models": {
                role: role_model(role, self._model_override)
                for role in ("orchestrator", "specialist")
            },
            # Judges are resolved per finding against the proposer, so there is
            # no single judge model to report here -- only the policy.
            "judging": {
                "panel_size": self.panel_size,
                "strict_independence": self.strict_independence,
            },
        }
