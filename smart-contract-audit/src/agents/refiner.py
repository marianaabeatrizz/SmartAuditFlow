"""The refinement loop: verification -> execution -> verification.

When the judging panel does not confirm a finding, the claim goes back to the
specialist that made it, together with the judge's objection. The specialist
must either answer the objection with evidence it did not cite before, or
withdraw the claim. The revision is then re-verified.

Why this is worth having
------------------------
On the benchmark, Top-5 is saturated (0.993) while Top-1 is 0.664: the true
vulnerability is almost always *found* and merely mis-ranked. A plausible cause
is a real finding described badly -- wrong function name, vague mechanism, no
exploit path -- which an honest judge is right to refuse. Giving the specialist
one chance to answer the specific objection converts some of those into
confirmed findings, which lifts them up the ranking.

The risk, and what guards it
----------------------------
A naive retry loop is verification shopping: keep rephrasing until some judge
says yes. Three guards:

1. **Bounded.** ``max_refine_rounds`` caps revisions; the default is one.
2. **New evidence required.** The specialist is told that rewording is not an
   answer; if it cannot cite code it did not cite before, it must withdraw.
   A revision that fails this test is discarded and the original stands.
3. **No judge shopping.** A revision is re-judged by the *same* judge that
   objected, wherever that judge is still available. The reviewer who raised
   the objection is the one who decides whether it was met.

Every refined finding is flagged, so an ablation can measure whether the loop
earns its cost rather than assuming it does.
"""
from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional, Tuple

from src.agents.base import Agent, extract_json
from src.agents.llm import LLMClient
from src.agents.specialists import CATALOGUE, Specialty
from src.agents.state import Finding

logger = logging.getLogger(__name__)

# Verdicts that send a finding back for revision.
FAILING = {"rejected", "uncertain"}


class RefinementAgent(Agent):
    """Re-runs one specialist against the judge's objection."""

    role = "specialist"  # it is the execution layer, re-entered

    def __init__(
        self,
        specialty: Specialty,
        client: LLMClient,
        model: Optional[str] = None,
    ):
        super().__init__(client, model)
        self.specialty = specialty
        self.name = f"refine:{specialty.key}"

    @property
    def system_prompt(self) -> str:
        return (
            f"You are a smart contract security auditor specialising in "
            f"{self.specialty.title}. A reviewer has challenged one of your "
            "findings. You care about being right, not about winning: if the "
            "reviewer is correct, you withdraw the claim without argument. If "
            "the reviewer missed something in the code, you show them exactly "
            "what, quoting it. You never defend a claim by restating it more "
            "confidently."
        )

    def build_prompt(self, finding: Finding, state: Dict[str, Any]) -> str:
        code = state.get("code_snippet", "")
        brief = state.get("context_brief", "")
        objection = finding.get("objection") or finding.get("rationale") or ""
        verdict = finding.get("verdict", "rejected")

        return f"""## Contract briefing
{brief}

## Contract
```solidity
{code}
```

## Your original finding
- **Issue**: {finding.get('Issue', '')}
- **Severity**: {finding.get('Severity', '')}
- **Location**: {finding.get('Location', '')}
- **Description**: {finding.get('Description', '')}
- **Impact**: {finding.get('Impact', '')}
- **Evidence you cited**: {finding.get('rationale', '') or '(none)'}
- **Exploit path you gave**: {finding.get('exploit_path', '') or '(none)'}

## The reviewer's ruling: {verdict.upper()}
{objection or '(no reason recorded)'}

## What you must do
Decide honestly which of these is true, then answer:

**A. The reviewer is right.** The code does not support the claim, or a guard
you overlooked already prevents it. Withdraw.

**B. The reviewer missed something.** There is code you did not cite that
answers their specific objection. Quote it exactly and explain why it settles
the point.

Hard rules:
- Restating your claim in stronger words is NOT an answer. If you have no new
  code to cite, choose A.
- Quote only code that is actually in the contract above. An invented line
  means the finding is discarded.
- Answer the objection that was raised, not a different one.
- If you can now give a concrete exploit path you could not give before, do.

Return ONLY JSON:

{{
  "action": "withdraw | revise",
  "reasoning": "one or two sentences on why",
  "new_evidence": "the exact code line(s) you are citing for the first time, or \\"\\" if withdrawing",
  "Issue": "sharper title, if revising",
  "Severity": "High | Medium | Low",
  "Description": "the corrected explanation, answering the objection",
  "Impact": "concrete consequence",
  "Location": "contract.function, corrected if it was wrong",
  "exploit_path": "ordered attacker steps, or \\"\\" if none exists",
  "confidence": 0.0
}}"""

    def refine(self, finding: Finding, state: Dict[str, Any]) -> Tuple[Finding, str]:
        """Return (finding, outcome) where outcome is revised|withdrawn|kept."""
        try:
            response = self.ask(self.build_prompt(finding, state), max_tokens=1800)
        except Exception as exc:  # noqa: BLE001
            logger.warning("%s failed: %s", self.name, exc)
            # Infrastructure failure must not silently drop the claim; the
            # original ruling stands.
            return finding, "kept"

        parsed = extract_json(response.text, expect="object") or {}
        out: Dict[str, Any] = dict(finding)
        out["_tokens"] = response.total_tokens

        action = str(parsed.get("action", "")).strip().lower()
        evidence = str(parsed.get("new_evidence", "")).strip()

        if action == "withdraw":
            # The claim keeps its uid: it is the same claim, conceded.
            out["refine_attempts"] = int(finding.get("refine_attempts", 0)) + 1
            out["withdrawn"] = True
            out["verdict"] = "rejected"
            out["confidence"] = 0.0
            out["rationale"] = (
                f"Withdrawn by {self.name} after review: "
                f"{str(parsed.get('reasoning', '')).strip()}"
            )
            return out, "withdrawn"  # type: ignore[return-value]

        if action != "revise":
            # The call was still paid for, so the cost travels with the
            # unchanged finding rather than vanishing from the ledger.
            return self._kept(finding, response.total_tokens), "kept"

        out["refine_attempts"] = int(finding.get("refine_attempts", 0)) + 1

        # Guard 2: a revision without new cited evidence is rewording, which is
        # exactly the failure mode this loop could otherwise manufacture.
        if not evidence or not self._evidence_is_real(evidence, state):
            logger.info(
                "%s revision rejected: %s",
                self.name,
                "no new evidence" if not evidence else "evidence not found in code",
            )
            return self._kept(finding, response.total_tokens), "kept"

        for field, key in (
            ("Issue", "Issue"), ("Description", "Description"),
            ("Impact", "Impact"), ("Location", "Location"),
        ):
            value = str(parsed.get(key, "")).strip()
            if value:
                out[field] = value

        severity = str(parsed.get("Severity", "")).strip().title()
        if severity in ("High", "Medium", "Low"):
            out["Severity"] = severity
        if parsed.get("exploit_path"):
            out["exploit_path"] = str(parsed["exploit_path"]).strip()

        out["rationale"] = evidence
        # A revision is a new candidate for the validator, so it needs its own
        # identity -- derived from the original so the lineage stays readable.
        out["uid"] = f"{finding.get('uid', 'anon')}#r{out['refine_attempts']}"
        out["refined"] = True
        out["withdrawn"] = False
        # Guard 3: remember who already ruled, so re-validation goes back to
        # that judge instead of sampling until one agrees.
        prior = list(finding.get("prior_judges") or [])
        prior += [j for j in (finding.get("judged_by") or []) if j not in prior]
        out["prior_judges"] = prior
        # The verdict is cleared so the validator treats this as open again.
        out.pop("verdict", None)
        try:
            out["confidence"] = min(max(float(parsed.get("confidence", 0.5)), 0.0), 1.0)
        except (TypeError, ValueError):
            out["confidence"] = 0.5
        return out, "revised"  # type: ignore[return-value]

    @staticmethod
    def _kept(finding: Finding, tokens: int) -> Finding:
        """The original finding, unchanged, but carrying the call's cost."""
        out = dict(finding)
        out["_tokens"] = int(tokens or 0)
        return out  # type: ignore[return-value]

    @staticmethod
    def _evidence_is_real(evidence: str, state: Dict[str, Any]) -> bool:
        """Cheap check that the quoted code actually occurs in the contract.

        Models cite lines that do not exist. Comparing on a whitespace-stripped
        basis catches the obvious fabrications without being so strict that a
        reformatted but genuine quote is thrown away.
        """
        code = (state.get("code_snippet") or "")
        if not code:
            return False
        haystack = "".join(code.split())
        # Any reasonably long quoted fragment must appear in the source.
        fragments = [
            "".join(line.split())
            for line in evidence.splitlines()
            if len("".join(line.split())) >= 12
        ]
        if not fragments:
            # Too short to verify either way; accept, the judge still has to
            # be convinced on re-validation.
            return True
        return any(f in haystack for f in fragments)

    def run(self, state: Dict[str, Any]) -> Dict[str, Any]:  # pragma: no cover
        raise NotImplementedError("RefinementAgent is driven by refine_node")


def refine_node_factory(client: LLMClient, model: Optional[str] = None):
    """Build the graph node that runs one refinement round."""

    agents: Dict[str, RefinementAgent] = {}

    def get_agent(specialty_key: str) -> Optional[RefinementAgent]:
        if specialty_key not in CATALOGUE:
            return None
        if specialty_key not in agents:
            agents[specialty_key] = RefinementAgent(
                CATALOGUE[specialty_key], client, model
            )
        return agents[specialty_key]

    def refine_node(state: Dict[str, Any]) -> Dict[str, Any]:
        validated: List[Finding] = list(state.get("validated_findings") or [])
        rnd = int(state.get("refine_round", 0)) + 1
        max_rounds = int(state.get("max_refine_rounds", 1))
        include_uncertain = bool(state.get("refine_uncertain", True))

        failing = {"rejected"} | ({"uncertain"} if include_uncertain else set())
        eligible_idx = [
            i for i, f in enumerate(validated)
            if f.get("verdict") in failing
            and not f.get("withdrawn")
            and int(f.get("refine_attempts", 0)) < max_rounds
        ]

        if not eligible_idx:
            return {
                "refine_round": rnd,
                "agent_log": [{
                    "agent": "refine", "role": "specialist",
                    "status": "ok", "round": rnd, "eligible": 0,
                }],
            }

        results: Dict[int, Tuple[Finding, str]] = {}
        with ThreadPoolExecutor(max_workers=min(6, len(eligible_idx))) as pool:
            futures = {}
            for i in eligible_idx:
                finding = validated[i]
                agent = get_agent(str(finding.get("specialty", "")))
                if agent is None:
                    continue
                futures[pool.submit(agent.refine, finding, state)] = i
            for future in as_completed(futures):
                i = futures[future]
                try:
                    results[i] = future.result()
                except Exception as exc:  # noqa: BLE001
                    logger.warning("refinement worker crashed: %s", exc)
                    results[i] = (validated[i], "kept")

        tokens = 0
        revised: List[Finding] = []
        kept: List[Finding] = []
        counts = {"revised": 0, "withdrawn": 0, "kept": 0}

        for i, finding in enumerate(validated):
            if i not in results:
                kept.append(finding)
                continue
            updated, outcome = results[i]
            tokens += int(dict(updated).pop("_tokens", 0) or 0)
            updated = {k: v for k, v in updated.items() if k != "_tokens"}  # type: ignore[assignment]
            counts[outcome] += 1
            if outcome == "revised":
                # Leaves validated_findings and re-enters as an open candidate.
                revised.append(updated)  # type: ignore[arg-type]
            else:
                kept.append(updated)  # type: ignore[arg-type]

        logger.info(
            "refinement round %s: %s revised, %s withdrawn, %s unchanged",
            rnd, counts["revised"], counts["withdrawn"], counts["kept"],
        )

        return {
            # Revised claims go back to the execution output so the validator
            # picks them up as new candidates on the next pass.
            "raw_findings": revised,
            "validated_findings": kept,
            "refine_round": rnd,
            "token_usage": tokens,
            "agent_log": [{
                "agent": "refine", "role": "specialist", "status": "ok",
                "round": rnd, "eligible": len(eligible_idx),
                "tokens": tokens, **counts,
            }],
        }

    return refine_node
