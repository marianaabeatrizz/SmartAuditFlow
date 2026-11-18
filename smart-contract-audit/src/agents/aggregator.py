"""Deduplication, corroboration merging and final ranking.

Twelve specialists working in parallel will sometimes land on the same issue
from different angles. Collapsing those duplicates matters twice over: the
report stops repeating itself, and agreement between agents that cannot see
each other's work becomes a usable confidence signal.
"""
from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Tuple

from src.agents.llm import (
    INDEPENDENCE_CROSS_MODEL,
    INDEPENDENCE_CROSS_PROVIDER,
    INDEPENDENCE_SELF,
    family,
)
from src.agents.state import Finding
from src.agents.validator import rank_score

logger = logging.getLogger(__name__)

# Words that carry no discriminating signal when comparing two finding titles.
_STOPWORDS = {
    "the", "a", "an", "in", "on", "of", "to", "for", "and", "or", "is", "are",
    "can", "be", "with", "without", "issue", "vulnerability", "potential",
    "possible", "risk", "problem", "contract", "function", "due", "may", "that",
    "this", "it", "its", "from", "by", "at", "as", "not", "no",
}

_TOKEN = re.compile(r"[a-z0-9_]+")

# Above this Jaccard similarity on title+location tokens, two findings are
# treated as the same issue. Tuned to merge genuine restatements while keeping
# distinct issues in the same function apart.
MERGE_THRESHOLD = 0.62


def _tokens(text: str) -> set:
    return {
        t for t in _TOKEN.findall((text or "").lower())
        if t not in _STOPWORDS and len(t) > 2
    }


def _similarity(a: Finding, b: Finding) -> float:
    """Jaccard overlap on title tokens, with location agreement as a tiebreak."""
    ta, tb = _tokens(a.get("Issue", "")), _tokens(b.get("Issue", ""))
    if not ta or not tb:
        return 0.0
    overlap = len(ta & tb) / len(ta | tb)

    la, lb = _tokens(a.get("Location", "")), _tokens(b.get("Location", ""))
    if la and lb:
        loc = len(la & lb) / len(la | lb)
        # Same place and similar words is strong evidence; different places
        # pulls two similar-sounding findings apart.
        overlap = 0.7 * overlap + 0.3 * loc
    return overlap


def deduplicate(findings: List[Finding]) -> List[Finding]:
    """Collapse restatements, keeping the best-evidenced version of each issue."""
    clusters: List[List[Finding]] = []

    for finding in findings:
        placed = False
        for cluster in clusters:
            if _similarity(finding, cluster[0]) >= MERGE_THRESHOLD:
                cluster.append(finding)
                placed = True
                break
        if not placed:
            clusters.append([finding])

    merged: List[Finding] = []
    for cluster in clusters:
        merged.append(_merge_cluster(cluster))
    return merged


def _merge_cluster(cluster: List[Finding]) -> Finding:
    """Pick the strongest member and fold in what the others add."""
    if len(cluster) == 1:
        winner = dict(cluster[0])
        winner["corroborations"] = 1
        winner["corroborating_families"] = 1
        return winner  # type: ignore[return-value]

    # The representative is the one the validator liked most.
    ordered = sorted(cluster, key=rank_score, reverse=True)
    winner = dict(ordered[0])

    distinct_agents = {f.get("agent", "") for f in cluster if f.get("agent")}
    winner["corroborations"] = len(distinct_agents) or len(cluster)
    # Agreement between two specialists running the *same* model is much
    # weaker evidence than it looks: they share training data, blind spots and
    # failure modes, so they will tend to hallucinate the same issue together.
    # Record how many distinct provider families actually agreed, so ranking
    # can discount same-model echo.
    winner["corroborating_families"] = len({
        family(f["proposer_model"]) for f in cluster if f.get("proposer_model")
    }) or 1

    # Keep the best exploit path available anywhere in the cluster.
    if not winner.get("exploit_path"):
        for f in ordered:
            if f.get("exploit_path"):
                winner["exploit_path"] = f["exploit_path"]
                break

    # Severity: take the highest any member confirmed, so a merge never
    # silently downgrades a confirmed High into a Low.
    order = {"High": 3, "Medium": 2, "Low": 1}
    confirmed = [f for f in cluster if f.get("verdict") == "confirmed"]
    pool = confirmed or cluster
    winner["Severity"] = max(
        (f.get("Severity", "Medium") for f in pool),
        key=lambda s: order.get(s, 2),
    )

    # Independence degrades to the weakest member: if any version of this
    # issue was self-judged, the merged finding inherits that caveat rather
    # than laundering it away behind a better-judged duplicate.
    levels = {f.get("independence", INDEPENDENCE_CROSS_PROVIDER) for f in cluster}
    if INDEPENDENCE_SELF in levels:
        winner["independence"] = INDEPENDENCE_SELF
    elif INDEPENDENCE_CROSS_MODEL in levels:
        winner["independence"] = INDEPENDENCE_CROSS_MODEL
    winner["self_validated"] = winner.get("independence") == INDEPENDENCE_SELF
    winner["judged_by"] = sorted(
        {j for f in cluster for j in (f.get("judged_by") or [])}
    )

    others = sorted(a for a in distinct_agents if a != winner.get("agent"))
    if others:
        winner["Description"] = (
            f"{winner.get('Description', '')}\n\n"
            f"[Independently corroborated by: {', '.join(others)}]"
        ).strip()
    return winner  # type: ignore[return-value]


def finalize(state: Dict[str, Any]) -> Dict[str, Any]:
    """Deduplicate, rank, and shape the report the API and harness consume."""
    validated: List[Finding] = state.get("validated_findings", []) or []

    merged = deduplicate(validated)
    merged.sort(key=rank_score, reverse=True)

    findings: List[Finding] = []
    for i, finding in enumerate(merged, start=1):
        out = dict(finding)
        out["rank"] = i
        out["score"] = round(rank_score(finding), 4)
        findings.append(out)  # type: ignore[arg-type]

    confirmed = [f for f in findings if f.get("verdict") == "confirmed"]
    rejected = [f for f in findings if f.get("verdict") == "rejected"]
    self_validated = [f for f in findings if f.get("self_validated")]

    levels = {f.get("independence", INDEPENDENCE_SELF) for f in findings}
    if not findings:
        overall = INDEPENDENCE_CROSS_PROVIDER
    elif INDEPENDENCE_SELF in levels:
        overall = INDEPENDENCE_SELF
    elif INDEPENDENCE_CROSS_MODEL in levels:
        overall = INDEPENDENCE_CROSS_MODEL
    else:
        overall = INDEPENDENCE_CROSS_PROVIDER

    return {
        "findings": findings,
        "FindingNumber": len(findings),
        "ConfirmedNumber": len(confirmed),
        "RejectedNumber": len(rejected),
        "DuplicatesMerged": len(validated) - len(findings),
        "CriticalIssues": len([f for f in findings if f.get("Severity") == "High"]),
        "MediumIssues": len([f for f in findings if f.get("Severity") == "Medium"]),
        "LowIssues": len([f for f in findings if f.get("Severity") == "Low"]),
        "SelfValidatedCount": len(self_validated),
        "IndependenceLevel": overall,
        # Refinement-loop outcomes, so an ablation can tell whether the
        # verification -> execution -> verification loop earned its cost.
        "RefinedCount": len([f for f in findings if f.get("refined")]),
        "WithdrawnCount": len([f for f in findings if f.get("withdrawn")]),
    }


def to_legacy_report(state: Dict[str, Any]) -> Dict[str, Any]:
    """Project the new state onto the report shape the existing API returns.

    The frontend and the published evaluation CSVs both expect the five
    capitalised fields, so the contract is preserved exactly and the new
    metadata rides alongside.
    """
    findings: List[Finding] = state.get("findings", []) or []
    legacy = [
        {
            "Issue": f.get("Issue", ""),
            "Severity": f.get("Severity", "Medium"),
            "Description": f.get("Description", ""),
            "Impact": f.get("Impact", ""),
            "Location": f.get("Location", ""),
            # extras are additive; existing consumers ignore them
            "Rank": f.get("rank"),
            "Confidence": f.get("confidence"),
            "Verdict": f.get("verdict"),
            "Agent": f.get("agent"),
            "Corroborations": f.get("corroborations", 1),
            "ExploitPath": f.get("exploit_path", ""),
            # Judge provenance travels with the finding so a reader can tell
            # whether its confidence was earned independently.
            "JudgedBy": f.get("judged_by", []),
            "Independence": f.get("independence", ""),
            "SelfValidated": bool(f.get("self_validated")),
            "JudgeAgreement": f.get("judge_agreement"),
            "ProposerModel": f.get("proposer_model", ""),
            "Refined": bool(f.get("refined")),
            "Withdrawn": bool(f.get("withdrawn")),
            "RefineAttempts": int(f.get("refine_attempts", 0) or 0),
        }
        for f in findings
    ]

    return {
        "Findings": legacy,
        "FindingNumber": len(legacy),
        "CriticalIssues": len([f for f in legacy if f["Severity"] == "High"]),
        "MediumIssues": len([f for f in legacy if f["Severity"] == "Medium"]),
        "LowIssues": len([f for f in legacy if f["Severity"] == "Low"]),
        "ContextBrief": state.get("context_brief", ""),
        "ExecutionSummary": {
            "rounds": state.get("round", 0),
            "total_tokens": state.get("token_usage", 0),
            "agents_run": len(state.get("agent_log", []) or []),
            "confirmed": len([f for f in findings if f.get("verdict") == "confirmed"]),
            "rejected": len([f for f in findings if f.get("verdict") == "rejected"]),
            # A result judged by the same model that produced it is not an
            # independent evaluation; say so in the report rather than leaving
            # the reader to infer it.
            "independence": state.get("IndependenceLevel", ""),
            "self_validated": state.get("SelfValidatedCount", 0),
            "refined": state.get("RefinedCount", 0),
            "withdrawn": state.get("WithdrawnCount", 0),
            "refine_rounds": state.get("refine_round", 0),
            "judges": sorted(
                {j for f in findings for j in (f.get("judged_by") or [])}
            ),
            "errors": state.get("errors", []),
        },
        "AgentLog": state.get("agent_log", []),
    }
