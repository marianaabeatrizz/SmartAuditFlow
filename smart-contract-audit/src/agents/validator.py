"""Adversarial validation.

This is the component the benchmark rewards. On ``standardVuln`` the published
SmartAuditFlow run already reaches Top-5 = 0.993 while Top-1 is only 0.664: the
true vulnerability is nearly always *found*, it is just not *ranked first*. So
the win is not in finding more, it is in deciding which of the candidates is
real and ordering accordingly.

A deliberate design choice: a rejected finding is **demoted, never deleted**.
Top-5 and Top-max measure whether the true issue is present at all, so dropping
a wrongly-rejected finding would destroy recall we cannot recover, while
demoting it costs nothing but rank.
"""
from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional

from src.agents.base import Agent, extract_json
from src.agents.llm import (
    INDEPENDENCE_CROSS_MODEL,
    INDEPENDENCE_CROSS_PROVIDER,
    INDEPENDENCE_SELF,
    LLMClient,
    INDEPENDENCE_UNJUDGED,
    independence_of,
    select_judges,
)
from src.agents.state import Finding

logger = logging.getLogger(__name__)

# How much each verdict is trusted when scoring. Chosen so that a confirmed
# finding always outranks an uncertain one, which always outranks a rejected
# one, regardless of the confidence the proposing specialist claimed.
VERDICT_WEIGHT = {"confirmed": 1.0, "uncertain": 0.45, "rejected": 0.08}
SEVERITY_BONUS = {"High": 0.10, "Medium": 0.04, "Low": 0.0}


def _identity(finding: Finding) -> tuple:
    """Stable key for 'is this the same candidate I already validated?'.

    Deliberately exact rather than fuzzy: near-duplicate merging is the
    aggregator's job, and conflating two distinct candidates here would skip a
    validation we actually owe.
    """
    uid = finding.get("uid")
    if uid:
        return ("uid", uid)
    # Fallback for findings built by hand (tests, fixtures) that carry no uid.
    return (
        finding.get("agent", ""),
        finding.get("task_id", ""),
        finding.get("Issue", ""),
        finding.get("Location", ""),
        int(finding.get("refine_attempts", 0) or 0),
    )


class ValidatorAgent(Agent):
    """Independent judging panel: tries to refute each candidate, then scores it.

    The judge is chosen per finding to be independent of whichever model
    proposed it (see ``llm.select_judges``). This matters because an LLM asked
    to grade its own output systematically favours it, so a validator sharing
    the proposer's model measures agreement with itself rather than the truth
    of the claim -- and the bias runs in the dangerous direction here, inflating
    confidence in findings that should have been rejected.

    ``panel_size`` > 1 asks several judges from different provider families and
    keeps their level of agreement as a signal: a finding three independent
    models confirm is worth more than one that splits them.
    """

    role = "validator"
    name = "validator"

    def __init__(
        self,
        client: LLMClient,
        model: Optional[str] = None,
        judge_models: Optional[List[str]] = None,
        panel_size: int = 1,
        strict_independence: bool = False,
        judge_top_n: int = 0,
    ):
        super().__init__(client, model)
        self.panel_size = max(1, int(panel_size))
        self.strict_independence = strict_independence
        # 0 judges everything. A positive value judges only that many of the
        # most confident candidates per contract -- the only ones whose
        # verdict can move Top-1/Top-5/MRR.
        self.judge_top_n = max(0, int(judge_top_n or 0))
        # Candidate judges: whatever was configured, else every model the
        # provider setup actually has keys for.
        self._configured_judges = judge_models
        self._warned_self = False

    def judge_pool(self) -> List[str]:
        if self._configured_judges:
            return [m for m in self._configured_judges if self.client.available(m)]
        available = self.client.get_available_models() if hasattr(
            self.client, "get_available_models"
        ) else {}
        return [m for m, ok in available.items() if ok]

    def judges_for(self, finding: Finding) -> List[str]:
        proposer = finding.get("proposer_model") or self.model
        pool = self.judge_pool()

        # A revised claim goes back to the judge that objected to it. Letting a
        # revision draw a fresh judge would turn the refinement loop into judge
        # shopping: retry until someone agrees. The reviewer who raised the
        # objection is the one who decides whether it was answered.
        prior = [j for j in (finding.get("prior_judges") or []) if j in pool]
        if prior and finding.get("refined"):
            still_independent = [
                j for j in prior
                if independence_of(proposer, j) != INDEPENDENCE_SELF
                or not self.strict_independence
            ]
            if still_independent:
                return still_independent[: self.panel_size]

        judges = select_judges(
            proposer, pool, self.panel_size, strict=self.strict_independence
        )
        if judges and independence_of(proposer, judges[0]) == INDEPENDENCE_SELF:
            if not self._warned_self:
                logger.warning(
                    "no model independent of '%s' is configured; findings will be "
                    "self-validated and flagged accordingly. Configure a second "
                    "provider to remove this bias.",
                    proposer,
                )
                self._warned_self = True
        return judges

    @property
    def system_prompt(self) -> str:
        return (
            "You are an adversarial reviewer on a smart contract audit. Another "
            "auditor claims to have found a vulnerability. Your job is to try to "
            "prove them wrong by reading the code. You are not hostile, you are "
            "rigorous: you confirm genuine findings readily and reject plausible-"
            "sounding claims that the code does not actually support. False "
            "positives are the main failure mode of automated auditing, and you "
            "are the last line of defence against them."
        )

    def _judge_once(
        self, finding: Finding, state: Dict[str, Any], judge_model: str
    ) -> Dict[str, Any]:
        """Ask one judge to rule on one claim. Returns a raw judgement."""
        code = state.get("code_snippet", "")
        brief = state.get("context_brief", "")

        prompt = f"""## Contract briefing
{brief}

## Contract
```solidity
{code}
```

## The claim to examine
- **Issue**: {finding.get('Issue', '')}
- **Severity claimed**: {finding.get('Severity', '')}
- **Location**: {finding.get('Location', '')}
- **Description**: {finding.get('Description', '')}
- **Impact claimed**: {finding.get('Impact', '')}
- **Evidence offered**: {finding.get('rationale', '')}
- **Exploit path offered**: {finding.get('exploit_path', '') or '(none given)'}

## Your task
Work through these in order:

1. **Does the cited code exist as described?** If the location or the quoted
   code is wrong or invented, that alone is grounds for rejection.
2. **Do the preconditions actually hold?** Check guards, modifiers, the
   Solidity version, and anything upstream that already prevents this.
3. **Is there a concrete exploit?** Write the ordered steps an attacker takes.
   If you cannot write them, the claim is at best theoretical.
4. **Is the severity honest?** A gas note is not High. Loss of funds is not Low.

Return ONLY JSON:

{{
  "verdict": "confirmed | uncertain | rejected",
  "confidence": 0.0,
  "reason": "what decided it, citing the code",
  "exploit_path": "ordered attacker steps, or \\"\\" if none",
  "corrected_severity": "High | Medium | Low",
  "corrected_issue": "a sharper title, or \\"\\" to keep the original"
}}

- "confirmed": the code supports it and an exploit path exists.
- "uncertain": plausible, but depends on context you cannot see from this code.
- "rejected": the code does not support the claim, or a guard already prevents it.
- "confidence" is your estimated probability that this is a real, exploitable
  issue. It must be consistent with your verdict."""

        try:
            response = self.client.complete(
                prompt,
                model=judge_model,
                system=self.system_prompt,
                timeout=90,
                max_tokens=1500,
                temperature=0.0,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "judge %s failed on %r: %s", judge_model, finding.get("Issue"), exc
            )
            # An infrastructure failure must never silently bury a finding.
            return {
                "judge": judge_model,
                "ok": False,
                "verdict": "uncertain",
                "confidence": float(finding.get("confidence", 0.5)) * 0.8,
                "reason": f"judge unavailable: {exc}",
                "tokens": 0,
            }

        parsed = extract_json(response.text, expect="object") or {}

        verdict = str(parsed.get("verdict", "uncertain")).strip().lower()
        if verdict not in VERDICT_WEIGHT:
            verdict = "uncertain"

        try:
            confidence = float(parsed.get("confidence", 0.5))
        except (TypeError, ValueError):
            confidence = 0.5
        confidence = min(max(confidence, 0.0), 1.0)

        severity = str(parsed.get("corrected_severity", "")).strip().title()
        return {
            "judge": judge_model,
            "ok": True,
            "verdict": verdict,
            "confidence": confidence,
            "reason": str(parsed.get("reason", "")).strip(),
            "exploit_path": str(parsed.get("exploit_path", "")).strip(),
            "severity": severity if severity in ("High", "Medium", "Low") else "",
            "issue": str(parsed.get("corrected_issue", "")).strip(),
            "tokens": response.total_tokens,
        }

    def validate(self, finding: Finding, state: Dict[str, Any]) -> Finding:
        """Run the independent panel over one finding and merge their rulings."""
        judges = self.judges_for(finding)
        proposer = finding.get("proposer_model") or self.model

        if len(judges) == 1:
            rulings = [self._judge_once(finding, state, judges[0])]
        else:
            # Judges must not see each other's answers, so they run in
            # parallel and independently -- no sequential anchoring.
            with ThreadPoolExecutor(max_workers=len(judges)) as pool:
                futures = [
                    pool.submit(self._judge_once, finding, state, j) for j in judges
                ]
                rulings = [f.result() for f in futures]

        return self._merge_rulings(finding, rulings, proposer)

    def _merge_rulings(
        self, finding: Finding, rulings: List[Dict[str, Any]], proposer: str
    ) -> Finding:
        out: Dict[str, Any] = dict(finding)
        usable = [r for r in rulings if r.get("ok")] or rulings

        verdicts = [r.get("verdict", "uncertain") for r in usable]
        tally = {v: verdicts.count(v) for v in set(verdicts)}
        top = max(tally.values())
        tied = [v for v, c in tally.items() if c == top]
        # A tie is an absence of majority, so the panel's honest answer is
        # "uncertain". Resolving a tie to "rejected" would be actively harmful:
        # rejection demotes a finding to the bottom of the report, so a split
        # panel over a *real* vulnerability would bury it and cost us the
        # Top-1/MRR we are trying to win. Rejection must be earned by a
        # majority, not inherited from deadlock.
        verdict = tied[0] if len(tied) == 1 else "uncertain"

        agreement = top / len(verdicts) if verdicts else 1.0
        confidence = sum(float(r.get("confidence", 0.5)) for r in usable) / len(usable)
        # A split panel is itself evidence of uncertainty, so pull confidence
        # toward the middle when the judges disagree.
        confidence *= 0.6 + 0.4 * agreement

        out["verdict"] = verdict
        out["confidence"] = min(max(confidence, 0.0), 1.0)
        out["rationale"] = " | ".join(
            f"{r['judge']}: {r.get('reason', '')}" for r in usable if r.get("reason")
        )
        # What a revision would have to answer. Kept separate from `rationale`
        # so a later round still has the objection verbatim after the rationale
        # has been overwritten by the new ruling.
        if verdict in ("rejected", "uncertain"):
            out["objection"] = " | ".join(
                f"{r['judge']}: {r.get('reason', '')}"
                for r in usable
                if r.get("reason") and r.get("verdict") in ("rejected", "uncertain")
            ) or out["rationale"]

        for r in usable:
            if r.get("exploit_path"):
                out["exploit_path"] = r["exploit_path"]
                break

        severities = [r["severity"] for r in usable if r.get("severity")]
        if severities:
            order = {"High": 3, "Medium": 2, "Low": 1}
            out["Severity"] = max(severities, key=lambda s: order.get(s, 2))
        for r in usable:
            if r.get("issue"):
                out["Issue"] = r["issue"]
                break

        # Only a judge that actually answered has judged anything. Counting a
        # failed call as a judgement reported a whole 143-contract run as
        # "cross-provider" while two thirds of its judges had returned 429 --
        # an infrastructure outage dressed up as a validated result.
        answered = [r["judge"] for r in rulings if r.get("ok")]
        attempted = [r["judge"] for r in rulings]
        out["judged_by"] = answered
        out["judges_attempted"] = attempted
        out["judges_failed"] = [r["judge"] for r in rulings if not r.get("ok")]
        out["judge_verdicts"] = verdicts
        out["judge_agreement"] = round(agreement, 3)
        # The weakest link decides how independent the judgement really was.
        levels = [independence_of(proposer, j) for j in answered]
        if not answered:
            out["independence"] = INDEPENDENCE_UNJUDGED
        elif INDEPENDENCE_SELF in levels:
            out["independence"] = INDEPENDENCE_SELF
        elif INDEPENDENCE_CROSS_MODEL in levels:
            out["independence"] = INDEPENDENCE_CROSS_MODEL
        else:
            out["independence"] = INDEPENDENCE_CROSS_PROVIDER
        out["self_validated"] = out["independence"] == INDEPENDENCE_SELF
        out["unjudged"] = not answered

        out["_tokens"] = sum(int(r.get("tokens", 0) or 0) for r in rulings)
        return out  # type: ignore[return-value]

    def run(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Validate every candidate, concurrently.

        ``raw_findings`` accumulates across rounds (it has an ``operator.add``
        reducer), so on a replan round we must validate only what is new --
        otherwise round one's findings are paid for twice.
        """
        all_candidates: List[Finding] = state.get("raw_findings", []) or []
        already: List[Finding] = state.get("validated_findings", []) or []
        seen = {_identity(f) for f in already}
        candidates = [f for f in all_candidates if _identity(f) not in seen]

        if not candidates:
            return {
                "validated_findings": already,
                "agent_log": [
                    self.log_entry(status="ok", validated=0, skipped=len(already))
                ],
            }

        # Judging every candidate is mostly waste: Top-1, Top-5 and MRR depend
        # only on the head of the ranking, so the 18th finding's verdict can
        # never move a metric. Judging the most confident N instead cuts judge
        # calls by roughly half and buys a run that fits a free tier -- the
        # difference between an independent judge and no judge at all.
        # Deferred findings keep their proposer's confidence and are flagged.
        deferred: List[Finding] = []
        if self.judge_top_n and len(candidates) > self.judge_top_n:
            ordered = sorted(
                candidates,
                key=lambda f: float(f.get("confidence", 0.5) or 0.5),
                reverse=True,
            )
            candidates, deferred = (ordered[:self.judge_top_n],
                                    ordered[self.judge_top_n:])
            for f in deferred:
                f["judged_by"] = []
                f["judges_attempted"] = []
                f["judges_failed"] = []
                f["independence"] = INDEPENDENCE_UNJUDGED
                f["self_validated"] = False
                # Not "unjudged": that flag marks an outage. This is a
                # deliberate, documented budget decision, and the finding was
                # outside the ranking window the metrics look at.
                f["unjudged"] = False
                f["judge_deferred"] = True
                f.setdefault("verdict", "uncertain")
            logger.info(
                "judging the %d most confident of %d candidates; %d deferred",
                len(candidates), len(candidates) + len(deferred), len(deferred),
            )

        max_workers = int(state.get("validator_concurrency", 6) or 6)
        validated: List[Finding] = list(deferred)
        tokens = 0

        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = {pool.submit(self.validate, f, state): f for f in candidates}
            for future in as_completed(futures):
                try:
                    result = future.result()
                except Exception as exc:  # noqa: BLE001
                    original = futures[future]
                    logger.warning("validator worker crashed: %s", exc)
                    fallback = dict(original)
                    fallback["verdict"] = "uncertain"
                    result = fallback  # type: ignore[assignment]
                tokens += int(result.pop("_tokens", 0) or 0)
                validated.append(result)  # type: ignore[arg-type]

        combined = already + validated

        counts = {v: 0 for v in VERDICT_WEIGHT}
        for f in combined:
            key = f.get("verdict", "uncertain")
            counts[key] = counts.get(key, 0) + 1

        self_validated = [f for f in combined if f.get("self_validated")]
        unjudged = [f for f in combined if f.get("unjudged")]
        deferred_out = [f for f in combined if f.get("judge_deferred")]
        # A deliberately deferred finding says nothing about how independent
        # the judging was; only the findings we actually sent to a judge do.
        judged = [f for f in combined if not f.get("judge_deferred")]
        levels = {f.get("independence", INDEPENDENCE_SELF) for f in judged}
        # Report the weakest independence in the run, not the best: one
        # self-judged finding is enough to qualify the whole result.
        if INDEPENDENCE_UNJUDGED in levels:
            overall = INDEPENDENCE_UNJUDGED
        elif INDEPENDENCE_SELF in levels:
            overall = INDEPENDENCE_SELF
        elif INDEPENDENCE_CROSS_MODEL in levels:
            overall = INDEPENDENCE_CROSS_MODEL
        else:
            overall = INDEPENDENCE_CROSS_PROVIDER

        judges_used = sorted({j for f in combined for j in (f.get("judged_by") or [])})
        if self_validated:
            logger.warning(
                "%d of %d findings were judged by their own proposer; "
                "treat confidence on those as inflated",
                len(self_validated), len(combined),
            )

        errors = []
        if self_validated:
            errors.append(
                f"{len(self_validated)}/{len(combined)} findings were self-validated "
                "(no independent judge configured); confidence on those is biased upward"
            )
        if unjudged:
            # Loud, because a run in this state is not a measurement: the
            # ranking rests on unchecked proposer confidence.
            logger.error(
                "%d of %d findings were never judged -- every judge call "
                "failed. This run cannot be reported as validated.",
                len(unjudged), len(combined),
            )
            errors.append(
                f"{len(unjudged)}/{len(combined)} findings went UNJUDGED because "
                "every judge call failed (quota or outage). Their ranking rests "
                "on unchecked proposer confidence; do not report this run."
            )

        return {
            "validated_findings": combined,
            "token_usage": tokens,
            "SelfValidatedCount": len(self_validated),
            "UnjudgedCount": len(unjudged),
            "IndependenceLevel": overall,
            "errors": errors,
            "agent_log": [
                self.log_entry(
                    status="ok",
                    validated=len(validated),
                    skipped=len(already),
                    tokens=tokens,
                    panel_size=self.panel_size,
                    judges=judges_used,
                    independence=overall,
                    self_validated=len(self_validated),
                    **counts,
                )
            ],
        }


def rank_score(finding: Finding) -> float:
    """Final ordering key. Higher ranks first.

    Driven primarily by the validator's verdict and confidence, nudged by
    severity and by how many independent specialists reported the same issue.
    Corroboration is a genuine signal here precisely because the specialists
    work from disjoint checklists and cannot see each other's output.
    """
    verdict = finding.get("verdict", "uncertain")
    confidence = float(finding.get("confidence", 0.5) or 0.0)
    severity = finding.get("Severity", "Medium")
    corroborations = int(finding.get("corroborations", 1) or 1)
    families = int(finding.get("corroborating_families", 1) or 1)

    score = VERDICT_WEIGHT.get(verdict, 0.45) * confidence
    score += SEVERITY_BONUS.get(severity, 0.04)
    # Diminishing: the second agent agreeing matters more than the fourth.
    # And agreement is only worth much when it crosses model families --
    # two specialists on the same model share training data and blind spots,
    # so they tend to be wrong together. Same-model echo gets a third of the
    # credit that genuine cross-provider agreement earns.
    per_agreement = 0.03 if families > 1 else 0.01
    score += min(corroborations - 1, 3) * per_agreement
    # A finding with a concrete exploit path is more actionable than one without.
    if finding.get("exploit_path"):
        score += 0.05
    return score
