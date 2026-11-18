"""Typed state shared by every node of the multi-agent audit graph.

This replaces the legacy ``context`` dict keyed by magic node ids
(``context.get('1728113785964')``) used by the linear pipeline in
``src/workflow/``. State is now explicit and typed, so adding or reordering an
agent no longer silently breaks a downstream consumer.
"""
from __future__ import annotations

import operator
from typing import Annotated, Any, Dict, List, Literal, TypedDict

Severity = Literal["High", "Medium", "Low"]

# Verdicts a validator can return for a candidate finding.
Verdict = Literal["confirmed", "uncertain", "rejected"]


class Task(TypedDict, total=False):
    """A unit of audit work produced by the orchestrator and routed to one agent."""

    id: str
    specialty: str          # key into the specialist registry
    instruction: str        # what this agent must check, in plain language
    focus: str              # function / region of the contract to concentrate on
    priority: Literal["high", "medium", "low"]
    round: int              # which planning round produced this task


class Finding(TypedDict, total=False):
    """A candidate vulnerability.

    The five capitalised fields are the legacy report schema and are kept
    byte-compatible so the existing frontend and the evaluation harness keep
    working unchanged. Everything below them is new multi-agent metadata.
    """

    # --- legacy schema (do not rename) ---
    Issue: str
    Severity: Severity
    Description: str
    Impact: str
    Location: str

    # --- multi-agent metadata ---
    # Stable identity, assigned once at creation. Everything downstream keys
    # off this rather than off the text, because the text legitimately changes
    # when a revision answers an objection -- and deriving identity from
    # mutable content made a revised claim look like a brand new one.
    uid: str
    agent: str              # which specialist proposed it
    specialty: str
    task_id: str
    round: int

    proposer_model: str     # the model that actually produced this claim

    # --- validation metadata ---
    verdict: Verdict
    confidence: float       # 0.0 - 1.0, drives the final ranking
    rationale: str          # why the validator reached that verdict
    exploit_path: str       # concrete attack sequence, when one exists
    corroborations: int     # how many independent agents reported this issue

    # --- judge provenance -------------------------------------------------
    # Recorded so a result can be audited for self-evaluation bias rather than
    # assumed clean. A model grading its own output is known to favour it, so
    # "who judged this, and how independent were they?" is part of the result,
    # not an implementation detail.
    judged_by: List[str]        # judge model(s) that returned a verdict
    independence: str           # weakest link: cross-provider | cross-model | self
    self_validated: bool        # True when the proposer judged its own claim
    judge_agreement: float      # 1.0 = unanimous panel; lower = split verdicts
    judge_verdicts: List[str]   # each judge's raw verdict, for inspection

    # --- refinement loop --------------------------------------------------
    # A finding that fails verification goes back to the specialist that
    # proposed it, with the judge's objection, and is then re-verified. These
    # fields make that history auditable and let an ablation measure whether
    # the loop earns its cost.
    refine_attempts: int        # how many revision rounds this claim survived
    refined: bool               # True once a revision replaced the original
    withdrawn: bool             # the specialist conceded instead of revising
    objection: str              # the judge's reason the revision had to answer
    prior_judges: List[str]     # judges that already ruled; re-validation
                                # prefers them, so a revision cannot shop for
                                # a friendlier judge


class AuditState(TypedDict, total=False):
    """State threaded through the whole graph.

    ``raw_findings`` carries an ``operator.add`` reducer because specialists run
    concurrently and LangGraph must merge their writes instead of letting the
    last one win.
    """

    # --- inputs ---
    code_snippet: str
    static_tool: str
    session_id: str

    # --- orchestrator ---
    context_brief: str              # shared understanding handed to every agent
    plan: List[Task]
    round: int
    max_rounds: int
    coverage_gaps: List[str]        # what the orchestrator still wants probed

    # --- refinement loop (verification -> execution -> verification) ---
    refine_round: int               # revision rounds spent so far
    max_refine_rounds: int          # budget; 0 disables the loop entirely
    refine_uncertain: bool          # also send back 'uncertain', not just
                                    # 'rejected' -- that is where a real
                                    # finding most often gets stuck

    # --- parallel specialist fan-in ---
    raw_findings: Annotated[List[Finding], operator.add]

    # --- validation / aggregation ---
    validated_findings: List[Finding]
    findings: List[Finding]         # final, deduplicated, ranked

    # --- bookkeeping ---
    agent_log: Annotated[List[Dict[str, Any]], operator.add]
    token_usage: Annotated[int, operator.add]
    errors: Annotated[List[str], operator.add]

    # --- report summary ---
    # LangGraph discards any key a node returns that is not declared on the
    # state schema, so the aggregator's summary counters must be declared here
    # or they vanish silently between the node and the caller.
    FindingNumber: int
    ConfirmedNumber: int
    RejectedNumber: int
    DuplicatesMerged: int
    CriticalIssues: int
    MediumIssues: int
    LowIssues: int
    SelfValidatedCount: int     # findings judged by their own proposer
    UnjudgedCount: int          # findings no judge answered on at all
    IndependenceLevel: str      # weakest independence across the whole run
    RefinedCount: int           # findings a revision round repaired
    WithdrawnCount: int         # claims the specialist conceded


def new_state(
    code_snippet: str,
    static_tool: str = "",
    session_id: str = "",
    max_rounds: int = 2,
    max_refine_rounds: int = 1,
    refine_uncertain: bool = True,
) -> AuditState:
    """Build a fresh state with every accumulator initialised.

    The reducer fields must start as real containers; LangGraph will not create
    them for us on the first concurrent write.
    """
    return AuditState(
        code_snippet=code_snippet,
        static_tool=static_tool,
        session_id=session_id,
        context_brief="",
        plan=[],
        round=0,
        max_rounds=max_rounds,
        coverage_gaps=[],
        refine_round=0,
        max_refine_rounds=max_refine_rounds,
        refine_uncertain=refine_uncertain,
        raw_findings=[],
        validated_findings=[],
        findings=[],
        agent_log=[],
        token_usage=0,
        errors=[],
    )
