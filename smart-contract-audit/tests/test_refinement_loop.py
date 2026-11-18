"""The inner loop: verification -> execution -> verification.

A finding the panel does not confirm goes back to the specialist that made it,
with the objection attached, and is verified again.

These tests pin the guards that stop the loop becoming verification shopping:
it is bounded, a revision must cite code it did not cite before, a fabricated
quote is discarded, and a revision is re-judged by the judge that objected
rather than by a fresh one.
"""
from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.agents.graph import build_graph
from src.agents.llm import LLMResponse
from src.agents.state import new_state
from tests.test_agents_graph import CONTRACT, FakeLLM
from tests.test_judge_independence import MultiProviderLLM

SPECIALTIES = ["reentrancy", "arithmetic"]

# A line that really is in CONTRACT, so the evidence check passes.
REAL_EVIDENCE = "msg.sender.call.value(amount)(\"\");"
FAKE_EVIDENCE = "require(neverAppearsInThisContract == 42);"


class RefineLLM(MultiProviderLLM):
    """Judges reject on the first pass; the specialist then revises.

    `evidence` controls what the revision cites, and `second_verdict` what the
    judge says about the revision -- the two knobs the guards hinge on.
    """

    def __init__(self, models, action="revise", evidence=REAL_EVIDENCE,
                 second_verdict="confirmed", **kw):
        super().__init__(models, **kw)
        self.action = action
        self.evidence = evidence
        self.second_verdict = second_verdict
        self.refine_calls = []
        self._seen_claims = set()

    def _respond(self, prompt, system):
        # The refiner introduces itself as "A reviewer has challenged".
        if "reviewer has challenged" in system:
            with self._lock:
                self.refine_calls.append(system[:40])
            return json.dumps({
                "action": self.action,
                "reasoning": "resposta à objeção",
                "new_evidence": self.evidence,
                "Issue": "Reentrancy in withdraw (revisado)",
                "Severity": "High",
                "Description": "Estado escrito após a chamada externa.",
                "Impact": "Dreno de fundos.",
                "Location": "Vault.withdraw",
                "exploit_path": "1. deposit 2. withdraw 3. re-entra",
                "confidence": 0.9,
            })

        if "adversarial reviewer" in system:
            claim = ""
            for line in prompt.splitlines():
                if line.startswith("- **Issue**:"):
                    claim = line.split(":", 1)[1].strip()
                    break
            # A revised claim carries "(revisado)" in its title.
            if "revisado" in claim:
                return json.dumps({
                    "verdict": self.second_verdict, "confidence": 0.92,
                    "reason": "a objeção foi respondida",
                    "exploit_path": "re-entra pelo fallback",
                    "corrected_severity": "High", "corrected_issue": "",
                })
            return json.dumps({
                "verdict": "rejected", "confidence": 0.2,
                "reason": "você não citou a linha da chamada externa",
                "exploit_path": "", "corrected_severity": "Medium",
                "corrected_issue": "",
            })

        return super()._respond(prompt, system)


def _run(llm, max_refine_rounds=1, specialties=SPECIALTIES):
    graph = build_graph(
        client=llm, specialties=specialties, strict=False, model="openai",
    )
    state = new_state(
        CONTRACT, session_id="t", max_rounds=1,
        max_refine_rounds=max_refine_rounds,
    )
    return graph.invoke(state)


def test_failed_finding_goes_back_to_execution():
    """The professor's increment, end to end."""
    llm = RefineLLM(["openai", "claude", "gemini"])
    result = _run(llm)

    assert llm.refine_calls, "nothing was sent back to the specialist"
    assert result["refine_round"] == 1, result["refine_round"]
    refined = [f for f in result["findings"] if f.get("refined")]
    assert refined, "no finding was repaired by the loop"
    assert refined[0]["verdict"] == "confirmed", refined[0]["verdict"]
    assert result["RefinedCount"] >= 1
    print(f"  {len(llm.refine_calls)} devolvido(s); "
          f"{result['RefinedCount']} reparado(s)")


def test_refined_finding_climbs_the_ranking():
    """Repairing a finding is only worth it if it changes the order."""
    llm = RefineLLM(["openai", "claude", "gemini"])
    result = _run(llm)
    ranks = {f["Issue"]: f["rank"] for f in result["findings"]}
    refined = [f for f in result["findings"] if f.get("refined")]
    assert refined[0]["rank"] == 1, (
        f"repaired finding did not reach the top: {ranks}"
    )


def test_loop_is_bounded():
    """A judge that never confirms must not loop forever."""
    llm = RefineLLM(["openai", "claude", "gemini"], second_verdict="rejected")
    result = _run(llm, max_refine_rounds=2)
    assert result["refine_round"] <= 2, result["refine_round"]
    # Every finding stopped at or below the budget.
    assert all(
        int(f.get("refine_attempts", 0)) <= 2 for f in result["findings"]
    )
    print(f"  parou na rodada {result['refine_round']} (orçamento 2)")


def test_disabled_when_budget_is_zero():
    llm = RefineLLM(["openai", "claude", "gemini"])
    result = _run(llm, max_refine_rounds=0)
    assert not llm.refine_calls, "loop ran despite a zero budget"
    assert result.get("RefinedCount", 0) == 0


def test_rewording_without_new_evidence_is_refused():
    """The guard against verification shopping."""
    llm = RefineLLM(["openai", "claude", "gemini"], evidence="")
    result = _run(llm)
    assert llm.refine_calls, "the specialist was never asked"
    assert result["RefinedCount"] == 0, (
        "a revision with no new evidence was accepted"
    )
    # The original ruling stands.
    assert all(f["verdict"] != "confirmed" for f in result["findings"])
    print("  revisão sem evidência nova recusada")


def test_fabricated_evidence_is_refused():
    """A quote that is not in the contract must not buy a confirmation."""
    llm = RefineLLM(["openai", "claude", "gemini"], evidence=FAKE_EVIDENCE)
    result = _run(llm)
    assert result["RefinedCount"] == 0, (
        "evidence absent from the contract was accepted"
    )
    print("  citação inventada recusada")


def test_withdrawal_is_recorded():
    llm = RefineLLM(["openai", "claude", "gemini"], action="withdraw")
    result = _run(llm)
    withdrawn = [f for f in result["findings"] if f.get("withdrawn")]
    assert withdrawn, "a withdrawal was not recorded"
    assert all(f["verdict"] == "rejected" for f in withdrawn)
    assert result["WithdrawnCount"] >= 1
    print(f"  {result['WithdrawnCount']} retirada(s) pelo próprio especialista")


def test_revision_goes_back_to_the_judge_that_objected():
    """No judge shopping: the reviewer who objected rules on the answer."""
    llm = RefineLLM(["openai", "claude", "gemini"])
    result = _run(llm)

    refined = [f for f in result["findings"] if f.get("refined")]
    assert refined, "nothing was refined"
    finding = refined[0]
    prior = set(finding.get("prior_judges") or [])
    judged = set(finding.get("judged_by") or [])
    assert prior, "prior judges were not recorded"
    assert judged <= prior, (
        f"revision drew a fresh judge: judged={judged} prior={prior}"
    )
    print(f"  objetou {sorted(prior)} · rejulgou {sorted(judged)}")


def test_refined_findings_are_flagged_for_ablation():
    llm = RefineLLM(["openai", "claude", "gemini"])
    result = _run(llm)
    refined = [f for f in result["findings"] if f.get("refined")]
    assert refined
    for f in refined:
        assert f.get("refine_attempts", 0) >= 1
        assert "objection" in f or f.get("rationale")


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for fn in tests:
        try:
            fn()
            print(f"PASS  {fn.__name__}")
        except AssertionError as exc:
            failed += 1
            print(f"FAIL  {fn.__name__}: {exc}")
        except Exception as exc:  # noqa: BLE001
            failed += 1
            print(f"ERROR {fn.__name__}: {type(exc).__name__}: {exc}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    sys.exit(1 if failed else 0)
