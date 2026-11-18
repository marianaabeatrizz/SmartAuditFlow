"""LangGraph wiring for the multi-agent audit.

    START -> context -> plan -> [specialist x N in parallel] -> validate
                 ^                                              |   ^
                 |                                              |   |
                 |                            did not pass ->  refine
                 |                                              |
                 |                                              v
                 +------------------ replan <-------------------+
                                       |
                                       v (no further leads, or budget spent)
                                   aggregate -> END

Two loops, deliberately nested:

* **Inner** (``validate -> refine -> validate``): a finding the judging panel
  did not confirm goes back to the specialist that proposed it, with the
  objection attached, and is verified again. Bounded by ``max_refine_rounds``.
  This is the cheaper loop, and the one that moves Top-1, because it repairs
  findings we already have rather than commissioning new ones.
* **Outer** (``replan -> plan``): the orchestrator widens the plan when a
  concrete coverage gap remains. Bounded by ``max_rounds``.

The fan-out uses ``Send``, so the number of specialists is decided at runtime by
the planner rather than baked into the graph. ``raw_findings`` carries an
``operator.add`` reducer, which is what makes the concurrent writes safe.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from langgraph.graph import END, START, StateGraph
from langgraph.types import Send

from src.agents.aggregator import finalize
from src.agents.llm import LLMClient
from src.agents.orchestrator import ContextAgent, PlannerAgent, ReplanAgent
from src.agents.refiner import refine_node_factory
from src.agents.validator import _identity as _key
from src.agents.specialists import DEFAULT_SPECIALTIES, build_specialists
from src.agents.state import AuditState
from src.agents.validator import ValidatorAgent

logger = logging.getLogger(__name__)


def build_graph(
    client: Optional[LLMClient] = None,
    specialties: Optional[List[str]] = None,
    strict: bool = True,
    always_dispatch: bool = True,
    model: Optional[str] = None,
    judge_models: Optional[List[str]] = None,
    panel_size: int = 1,
    strict_independence: bool = False,
    judge_top_n: int = 0,
):
    """Compile the audit graph.

    Args:
        client: LLM facade. Built with the given ``strict`` setting if omitted.
        specialties: subset of the catalogue to enable. Defaults to all,
            ordered by benchmark weight.
        strict: fail loudly rather than silently answering with another
            provider's model. Keep this on for anything you intend to report.
        always_dispatch: run every specialty regardless of planner opinion.
        model: pin every *proposing* role to this one model. Leave as None to
            use the per-role routing in ``llm.ROLE_PREFERENCES``. Pinning is
            what you want for a like-for-like benchmark against a single-model
            baseline; per-role routing is what you want in production.
            Note this does **not** pin the judges -- see below.
        judge_models: candidate judges. Defaults to every configured model.
            Judges are chosen per finding to be independent of whichever model
            proposed it, so pinning ``model`` does not drag the judge along
            with it: a model grading its own output favours it, and that bias
            inflates exactly the confidence score our ranking depends on.
        panel_size: judges per finding. 1 is a single independent reviewer;
            >1 convenes a panel drawn from different provider families and
            keeps their agreement as a confidence signal, at N times the cost.
        strict_independence: raise rather than fall back to self-validation
            when no independent judge is configured. Turn this on for runs
            whose numbers you intend to report.
    """
    client = client or LLMClient(strict=strict)
    specialties = specialties or DEFAULT_SPECIALTIES

    context_agent = ContextAgent(client, model)
    planner = PlannerAgent(
        client, model, available=specialties, always_dispatch=always_dispatch
    )
    specialists = build_specialists(client, specialties, model)
    validator = ValidatorAgent(
        client,
        model,
        judge_models=judge_models,
        panel_size=panel_size,
        strict_independence=strict_independence,
        judge_top_n=judge_top_n,
    )
    replanner = ReplanAgent(client, model)

    def context_node(state: AuditState) -> Dict[str, Any]:
        return context_agent.run(dict(state))

    def plan_node(state: AuditState) -> Dict[str, Any]:
        return planner.run(dict(state))

    def specialist_node(state: Dict[str, Any]) -> Dict[str, Any]:
        task = state.get("current_task") or {}
        agent = specialists.get(task.get("specialty", ""))
        if agent is None:
            return {"raw_findings": []}
        return agent.run(state)

    def validate_node(state: AuditState) -> Dict[str, Any]:
        return validator.run(dict(state))

    def replan_node(state: AuditState) -> Dict[str, Any]:
        return replanner.run(dict(state))

    refine_node = refine_node_factory(client, model)

    def aggregate_node(state: AuditState) -> Dict[str, Any]:
        return finalize(dict(state))

    # --- fan-out -----------------------------------------------------------
    def dispatch(state: AuditState):
        plan = state.get("plan") or []
        if not plan:
            # Nothing to do this round; skip straight to validation so the run
            # still terminates cleanly.
            return "validate"
        return [
            Send("specialist", {**dict(state), "current_task": task}) for task in plan
        ]

    # --- after verification: refine, replan, or finish ---------------------
    def after_validation(state: AuditState) -> str:
        """The inner loop comes first: verification -> execution -> verification.

        A claim the panel did not confirm goes back to the specialist that made
        it before the orchestrator is asked whether the *plan* needs widening.
        Fixing a finding we already have is cheaper, and more likely to move
        Top-1, than commissioning new ones.
        """
        max_refine = int(state.get("max_refine_rounds", 1))
        rnd = int(state.get("refine_round", 0))
        if max_refine <= 0 or rnd >= max_refine:
            return "replan"

        failing = {"rejected"}
        if state.get("refine_uncertain", True):
            failing.add("uncertain")
        pending = [
            f for f in (state.get("validated_findings") or [])
            if f.get("verdict") in failing
            and not f.get("withdrawn")
            and int(f.get("refine_attempts", 0) or 0) < max_refine
        ]
        if pending:
            logger.info(
                "refinement round %s: %s finding(s) go back to execution",
                rnd + 1, len(pending),
            )
            return "refine"
        return "replan"

    def after_refine(state: AuditState) -> str:
        """Re-verify revisions; go straight on when nothing was revised."""
        if state.get("raw_findings"):
            unvalidated = {
                _key(f) for f in (state.get("raw_findings") or [])
            } - {_key(f) for f in (state.get("validated_findings") or [])}
            if unvalidated:
                return "validate"
        return "replan"

    def should_replan(state: AuditState) -> str:
        gaps = state.get("coverage_gaps") or []
        rnd = int(state.get("round", 0))
        max_rounds = int(state.get("max_rounds", 2))
        if gaps and rnd < max_rounds:
            logger.info("replanning: round %s, %s gap(s)", rnd + 1, len(gaps))
            return "plan"
        return "aggregate"

    graph = StateGraph(AuditState)
    graph.add_node("context", context_node)
    graph.add_node("plan", plan_node)
    graph.add_node("specialist", specialist_node)
    graph.add_node("validate", validate_node)
    graph.add_node("refine", refine_node)
    graph.add_node("replan", replan_node)
    graph.add_node("aggregate", aggregate_node)

    graph.add_edge(START, "context")
    graph.add_edge("context", "plan")
    graph.add_conditional_edges("plan", dispatch, ["specialist", "validate"])
    graph.add_edge("specialist", "validate")
    # Inner loop (the professor's increment): a claim that fails verification
    # returns to execution and is verified again, bounded by max_refine_rounds.
    graph.add_conditional_edges("validate", after_validation, ["refine", "replan"])
    graph.add_conditional_edges("refine", after_refine, ["validate", "replan"])
    # Outer loop: widen the plan when coverage gaps remain.
    graph.add_conditional_edges("replan", should_replan, ["plan", "aggregate"])
    graph.add_edge("aggregate", END)

    return graph.compile()
