"""Judge independence: no model may grade its own findings unflagged.

An LLM asked to evaluate its own output systematically prefers it. In this
pipeline that bias runs in the dangerous direction -- it inflates the
confidence score the whole ranking is built on, so a self-validated run would
report better Top-1/MRR without auditing any better.

These tests pin the guarantees: judges are drawn from a different provider
family than the proposer, self-validation is refused in strict mode and
flagged when permitted, and a split panel lowers confidence.
"""
from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.agents.graph import build_graph
from src.agents.llm import (
    INDEPENDENCE_CROSS_MODEL,
    INDEPENDENCE_CROSS_PROVIDER,
    INDEPENDENCE_SELF,
    LLMResponse,
    NoIndependentJudge,
    family,
    independence_of,
    select_judges,
)
from src.agents.state import new_state
from src.agents.validator import ValidatorAgent
from tests.test_agents_graph import CONTRACT, TEST_PROVIDERS, FakeLLM


# ----------------------------------------------------------- unit: policy ---
def test_independence_is_classified_by_provider_not_model_id():
    # Same family, different checkpoint: weak, not independent.
    assert independence_of("openai", "gpt-4o-mini") == INDEPENDENCE_CROSS_MODEL
    assert independence_of("claude", "claude-opus-4-0") == INDEPENDENCE_CROSS_MODEL
    # Different family: genuinely independent.
    assert independence_of("openai", "claude") == INDEPENDENCE_CROSS_PROVIDER
    assert independence_of("gemini", "deepseek") == INDEPENDENCE_CROSS_PROVIDER
    assert independence_of("openai", "openai") == INDEPENDENCE_SELF


def test_judge_selection_prefers_a_different_provider():
    judges = select_judges("gpt-4o-mini", ["openai", "gpt-4.1", "claude", "gemini"], 1)
    assert family(judges[0]) != "openai", f"picked a same-family judge: {judges}"


def test_panel_spreads_across_provider_families():
    pool = ["openai", "gpt-4.1", "claude", "claude-opus-4-0", "gemini", "deepseek"]
    judges = select_judges("openai", pool, panel_size=3)
    families = {family(j) for j in judges}
    assert len(families) == 3, f"panel collapsed into {families}: {judges}"
    assert "openai" not in families, "proposer's own family sat on its panel"


def test_same_family_used_only_when_nothing_else_exists():
    judges = select_judges("gpt-4o-mini", ["openai", "gpt-4.1"], 1)
    assert judges, "no judge chosen at all"
    assert independence_of("gpt-4o-mini", judges[0]) == INDEPENDENCE_CROSS_MODEL


def test_strict_mode_refuses_self_validation():
    try:
        select_judges("openai", ["openai"], 1, strict=True)
    except NoIndependentJudge as exc:
        assert "independent" in str(exc)
        return
    raise AssertionError("strict mode allowed a model to judge itself")


def test_non_strict_falls_back_to_self():
    assert select_judges("openai", ["openai"], 1, strict=False) == ["openai"]


# ------------------------------------------------- integration: the graph ---
class MultiProviderLLM(FakeLLM):
    """Fake client exposing several providers, recording who judged what."""

    def __init__(self, models, **kw):
        super().__init__(**kw)
        self._models = list(models)
        self.judgements = []  # (judge_model, claim_title)

    def available(self, model):
        return model in self._models

    def get_available_models(self):
        return {m: True for m in self._models}


    def complete(self, prompt, model, system="", timeout=90, max_tokens=4000,
                 temperature=0.0):
        if "adversarial reviewer" in system:
            claim = ""
            for line in prompt.splitlines():
                if line.startswith("- **Issue**:"):
                    claim = line.split(":", 1)[1].strip()
                    break
            with self._lock:
                self.judgements.append((model, claim))
        return super().complete(prompt, model, system, timeout, max_tokens,
                                temperature)


SPECIALTIES = ["reentrancy", "arithmetic"]


def _run(models, panel_size=1, strict_independence=False, pinned="openai"):
    llm = MultiProviderLLM(models)
    graph = build_graph(
        client=llm,
        specialties=SPECIALTIES,
        strict=False,
        model=pinned,
        panel_size=panel_size,
        strict_independence=strict_independence,
    )
    return llm, graph.invoke(new_state(CONTRACT, session_id="t", max_rounds=1))


def test_pinned_model_does_not_drag_the_judge_with_it():
    """The benchmark path pins one model; the judge must still be independent."""
    llm, result = _run(["openai", "claude", "gemini"], pinned="openai")

    assert llm.judgements, "nothing was judged"
    judges = {m for m, _ in llm.judgements}
    assert "openai" not in judges, (
        f"the pinned proposer model judged its own findings: {judges}"
    )
    for f in result["findings"]:
        assert f["independence"] == INDEPENDENCE_CROSS_PROVIDER, f["independence"]
        assert not f["self_validated"]
    assert result["SelfValidatedCount"] == 0
    assert result["IndependenceLevel"] == INDEPENDENCE_CROSS_PROVIDER
    print(f"  proposer=openai judged by {sorted(judges)}")


def test_single_provider_is_flagged_not_hidden():
    llm, result = _run(["openai"], pinned="openai", strict_independence=False)
    assert result["SelfValidatedCount"] == len(result["findings"]) > 0
    assert result["IndependenceLevel"] == INDEPENDENCE_SELF
    assert all(f["self_validated"] for f in result["findings"])
    # The caveat must reach the caller, not just the log.
    assert any("self-validated" in e for e in result.get("errors", [])), (
        "self-validation was not surfaced in errors"
    )
    print(f"  {result['SelfValidatedCount']} finding(s) flagged as self-judged")


def test_panel_records_every_judge():
    llm, result = _run(["openai", "claude", "gemini", "deepseek"], panel_size=3)
    for f in result["findings"]:
        assert len(f["judged_by"]) == 3, f["judged_by"]
        assert "openai" not in f["judged_by"]
        families = {family(j) for j in f["judged_by"]}
        assert len(families) == 3, f"panel not diverse: {f['judged_by']}"
    print(f"  panel: {result['findings'][0]['judged_by']}")


class ScriptedPanel(MultiProviderLLM):
    """Returns a fixed verdict per judge, to test panel aggregation."""

    def __init__(self, verdicts):
        super().__init__(["openai", "claude", "gemini", "deepseek"])
        self.verdicts = verdicts

    def complete(self, prompt, model, system="", timeout=90, max_tokens=4000,
                 temperature=0.0):
        verdict = self.verdicts.get(model, "confirmed")
        return LLMResponse(
            text=json.dumps({
                "verdict": verdict, "confidence": 0.9,
                "reason": f"{model} says {verdict}", "exploit_path": "",
                "corrected_severity": "High", "corrected_issue": "",
            }),
            model=model, prompt_tokens=5, completion_tokens=5,
        )


class DeadPanel(MultiProviderLLM):
    """Every judge call fails, as it does when a daily quota is exhausted."""

    def __init__(self):
        super().__init__(["openai", "claude", "gemini", "deepseek"])

    def complete(self, prompt, model, system="", timeout=90, max_tokens=4000,
                 temperature=0.0):
        if "adversarial reviewer" in system:
            raise RuntimeError("429 RESOURCE_EXHAUSTED: quota exceeded")
        return super().complete(prompt, model, system, timeout, max_tokens,
                                temperature)


def test_a_failed_judge_is_not_counted_as_having_judged():
    """Regression: the first full 143-contract run reported every contract as
    'cross-provider' while two thirds of its judge calls had returned 429.

    A finding nobody answered on keeps only its proposer's confidence, so
    calling that independently validated turns an outage into a result.
    """
    from src.agents.llm import INDEPENDENCE_UNJUDGED

    finding = {
        "Issue": "Reentrancy in withdraw", "Severity": "High",
        "Description": "x", "Impact": "y", "Location": "Vault.withdraw",
        "proposer_model": "openai", "confidence": 0.9, "agent": "a",
    }
    state = {"code_snippet": CONTRACT, "context_brief": ""}

    out = ValidatorAgent(DeadPanel(), panel_size=2).validate(finding, state)

    assert out["judged_by"] == [], f"a failed judge was credited: {out['judged_by']}"
    assert out["judges_failed"], "the failure was not recorded at all"
    assert out["independence"] == INDEPENDENCE_UNJUDGED, out["independence"]
    assert out["unjudged"] is True
    print(f"  independence={out['independence']} failed={out['judges_failed']}")


def test_unjudged_findings_are_surfaced_to_the_caller():
    """The caveat has to reach the run's output, not only the log."""
    llm = DeadPanel()
    graph = build_graph(client=llm, specialties=SPECIALTIES, strict=False,
                        model="openai", panel_size=1)
    result = graph.invoke(new_state(CONTRACT, session_id="t", max_rounds=1))

    assert result["UnjudgedCount"] > 0, "unjudged findings were not counted"
    assert any("UNJUDGED" in e for e in result.get("errors", [])), (
        f"no caveat reached the caller: {result.get('errors')}"
    )
    print(f"  {result['UnjudgedCount']} finding(s) flagged unjudged")


def test_work_is_spread_across_equally_independent_judges():
    """One judge per finding plus three free providers must not mean one
    provider doing everything: that drove the run to 20 min/contract while
    two providers sat idle."""
    pool = ["claude", "gemini", "deepseek"]
    picked = {select_judges("openai", pool, 1, seed=f"uid-{i}")[0]
              for i in range(40)}
    assert len(picked) > 1, f"all 40 findings went to one judge: {picked}"


def test_the_spread_is_reproducible():
    """Keyed on the finding, not on a counter: a benchmark whose judge
    allocation shifts with thread scheduling is not reproducible."""
    pool = ["claude", "gemini", "deepseek"]
    first = [select_judges("openai", pool, 1, seed=f"uid-{i}")[0] for i in range(20)]
    second = [select_judges("openai", pool, 1, seed=f"uid-{i}")[0] for i in range(20)]
    assert first == second


def test_spreading_never_picks_a_less_independent_judge():
    """Load balancing must not trade away independence."""
    pool = ["gpt-4.1", "claude", "gemini"]          # gpt-4.1 is same family
    for i in range(30):
        chosen = select_judges("openai", pool, 1, seed=f"uid-{i}")[0]
        assert independence_of("openai", chosen) == INDEPENDENCE_CROSS_PROVIDER


def test_judging_is_capped_to_the_most_confident_candidates():
    """Only the head of the ranking can move Top-1/Top-5/MRR, so judging the
    tail buys nothing and costs the quota that pays for independence."""
    llm = MultiProviderLLM(["openai", "claude", "gemini"])
    validator = ValidatorAgent(llm, model="openai", panel_size=1, judge_top_n=3)

    candidates = [
        {"Issue": f"finding {i}", "Severity": "Medium", "Description": "d",
         "Impact": "i", "Location": "L", "proposer_model": "openai",
         "confidence": i / 10.0, "agent": "a", "uid": f"u{i}"}
        for i in range(10)
    ]
    out = validator.run({"raw_findings": candidates, "code_snippet": CONTRACT,
                         "context_brief": "", "validated_findings": []})

    findings = out["validated_findings"]
    assert len(findings) == 10, "a deferred finding must not be dropped"

    judged = [f for f in findings if not f.get("judge_deferred")]
    deferred = [f for f in findings if f.get("judge_deferred")]
    assert len(judged) == 3, f"judged {len(judged)}, expected 3"
    assert len(deferred) == 7

    # It must be the most confident three that were judged.
    assert {f["Issue"] for f in judged} == {"finding 9", "finding 8", "finding 7"}
    # And deferring must not be reported as an outage.
    assert out["UnjudgedCount"] == 0, "a budget decision was reported as a failure"
    assert all(not f.get("unjudged") for f in deferred)
    print(f"  judged {len(judged)} of {len(findings)}, deferred {len(deferred)}")


def test_deferring_does_not_weaken_the_reported_independence():
    """The run's independence level describes the findings actually judged."""
    from src.agents.llm import INDEPENDENCE_CROSS_PROVIDER

    llm = MultiProviderLLM(["openai", "claude", "gemini"])
    validator = ValidatorAgent(llm, model="openai", panel_size=1, judge_top_n=2)
    candidates = [
        {"Issue": f"f{i}", "Severity": "Medium", "Description": "d",
         "Impact": "i", "Location": "L", "proposer_model": "openai",
         "confidence": 0.9 - i / 100.0, "agent": "a", "uid": f"u{i}"}
        for i in range(6)
    ]
    out = validator.run({"raw_findings": candidates, "code_snippet": CONTRACT,
                         "context_brief": "", "validated_findings": []})
    assert out["IndependenceLevel"] == INDEPENDENCE_CROSS_PROVIDER, (
        out["IndependenceLevel"]
    )
    assert not any("UNJUDGED" in e for e in out.get("errors", []))


def test_split_panel_lowers_confidence_versus_unanimous():
    """Disagreement among independent judges is itself evidence of doubt."""
    finding = {
        "Issue": "Reentrancy in withdraw", "Severity": "High",
        "Description": "x", "Impact": "y", "Location": "Vault.withdraw",
        "proposer_model": "openai", "confidence": 0.9,
        "agent": "specialist:reentrancy",
    }
    state = {"code_snippet": CONTRACT, "context_brief": ""}

    unanimous = ValidatorAgent(
        ScriptedPanel({"claude": "confirmed", "gemini": "confirmed",
                       "deepseek": "confirmed"}),
        panel_size=3,
    ).validate(finding, state)

    split = ValidatorAgent(
        ScriptedPanel({"claude": "confirmed", "gemini": "rejected",
                       "deepseek": "uncertain"}),
        panel_size=3,
    ).validate(finding, state)

    assert unanimous["judge_agreement"] == 1.0
    assert split["judge_agreement"] < 1.0
    assert split["confidence"] < unanimous["confidence"], (
        f"split panel ({split['confidence']:.3f}) not less confident than "
        f"unanimous ({unanimous['confidence']:.3f})"
    )
    # A three-way split is an absence of majority, so it must read as
    # "uncertain" -- not as confirmation, and not as rejection either, since
    # rejection would demote a possibly-real finding to the bottom.
    assert split["verdict"] == "uncertain", split["verdict"]
    print(
        f"  unanimous conf={unanimous['confidence']:.3f} / "
        f"split conf={split['confidence']:.3f} verdict={split['verdict']}"
    )


def test_rejection_requires_a_majority_not_a_deadlock():
    """A 1-1 split must not bury a finding that one judge confirmed."""
    finding = {
        "Issue": "Reentrancy in withdraw", "Severity": "High",
        "Description": "x", "Impact": "y", "Location": "Vault.withdraw",
        "proposer_model": "openai", "confidence": 0.9, "agent": "a",
    }
    state = {"code_snippet": CONTRACT, "context_brief": ""}

    deadlock = ValidatorAgent(
        ScriptedPanel({"claude": "confirmed", "gemini": "rejected"}), panel_size=2,
    ).validate(finding, state)
    assert deadlock["verdict"] == "uncertain", deadlock["verdict"]

    # A genuine majority against it, however, must still reject.
    majority = ValidatorAgent(
        ScriptedPanel({"claude": "rejected", "gemini": "rejected",
                       "deepseek": "confirmed"}),
        panel_size=3,
    ).validate(finding, state)
    assert majority["verdict"] == "rejected", majority["verdict"]
    print(f"  deadlock -> {deadlock['verdict']}, majority -> {majority['verdict']}")


def test_judges_run_concurrently_so_they_cannot_anchor():
    llm, _ = _run(["openai", "claude", "gemini", "deepseek"], panel_size=3)
    assert llm.max_concurrent > 1, "judges ran sequentially"


def test_same_model_corroboration_is_discounted():
    """Two agents on one model agreeing is weak evidence, not strong."""
    from src.agents.validator import rank_score

    base = {
        "verdict": "confirmed", "confidence": 0.8, "Severity": "Medium",
        "corroborations": 3,
    }
    echo = {**base, "corroborating_families": 1}       # same model, 3 times
    genuine = {**base, "corroborating_families": 3}    # 3 different providers

    assert rank_score(genuine) > rank_score(echo), (
        "cross-provider agreement scored no higher than same-model echo"
    )
    # And the echo must not outrank a solitary finding by much.
    alone = {**base, "corroborations": 1, "corroborating_families": 1}
    assert rank_score(echo) - rank_score(alone) < 0.05
    print(
        f"  alone={rank_score(alone):.3f} echo={rank_score(echo):.3f} "
        f"cross-provider={rank_score(genuine):.3f}"
    )


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
