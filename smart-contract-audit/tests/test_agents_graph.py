"""End-to-end exercise of the multi-agent graph against a scripted LLM.

No API keys and no network: the point is to prove the wiring -- parallel
fan-out, the reducer merge, validation, the replan loop, deduplication and
ranking -- independently of model quality, and to keep it provable in CI.

Run:  .venv/Scripts/python.exe -m pytest tests/test_agents_graph.py -v
  or: .venv/Scripts/python.exe tests/test_agents_graph.py
"""
from __future__ import annotations

import json
import os
import sys
import threading
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.agents.aggregator import deduplicate
from src.agents.graph import build_graph
from src.agents.llm import LLMResponse, make_provider, set_registry
from src.agents.state import new_state
from src.agents.validator import rank_score

# Providers these tests pretend to have. Installed explicitly so the suite
# never depends on what happens to be in the developer's .env.
TEST_PROVIDERS = {
    "openai": make_provider("openai", "openai"),
    "gpt-4o-mini": make_provider("gpt-4o-mini", "openai"),
    "gpt-4.1": make_provider("gpt-4.1", "openai"),
    "claude": make_provider("claude", "anthropic"),
    "claude-opus-4-0": make_provider("claude-opus-4-0", "anthropic"),
    "gemini": make_provider("gemini", "google"),
    "deepseek": make_provider("deepseek", "deepseek"),
}
set_registry(TEST_PROVIDERS)

CONTRACT = """
pragma solidity ^0.4.24;
contract Vault {
    mapping(address => uint) public balances;
    function deposit() public payable { balances[msg.sender] += msg.value; }
    function withdraw(uint amount) public {
        require(balances[msg.sender] >= amount);
        msg.sender.call.value(amount)("");
        balances[msg.sender] -= amount;
    }
}
"""


class FakeLLM:
    """Scripted stand-in for LLMClient.

    Dispatches on the system prompt so each role gets a plausible reply, and
    records call order so the test can assert on concurrency.
    """

    # A real call takes seconds; without some latency here every task would
    # finish before the next begins and concurrency would be unobservable
    # even though the graph does run them in parallel.
    LATENCY = 0.15

    def __init__(self, replan_once: bool = False):
        self.replan_once = replan_once
        self.calls = []
        self.tokens = 0
        self._lock = threading.Lock()
        self._replans = 0
        self.max_concurrent = 0
        self._active = 0

    def available(self, model):
        return True

    def get_available_models(self):
        return {name: True for name in TEST_PROVIDERS}

    def complete(self, prompt, model, system="", timeout=90, max_tokens=4000, temperature=0.0):
        with self._lock:
            self.calls.append(system[:40])
            self._active += 1
            self.max_concurrent = max(self.max_concurrent, self._active)
        try:
            time.sleep(self.LATENCY)
            text = self._respond(prompt, system)
        finally:
            with self._lock:
                self._active -= 1
        return LLMResponse(text=text, model=model, prompt_tokens=10, completion_tokens=5)

    def _respond(self, prompt, system):
        if "lead auditor opening an engagement" in system:
            return "Vault holds ether. Solidity 0.4.24, so arithmetic wraps. withdraw() calls out before updating state."

        if "allocating work" in system:
            return json.dumps([
                {"specialty": "reentrancy", "instruction": "Check withdraw()",
                 "focus": "Vault.withdraw", "priority": "high"},
                {"specialty": "unchecked_low_level_calls", "instruction": "Check the call",
                 "focus": "Vault.withdraw", "priority": "high"},
            ])

        if "adversarial reviewer" in system:
            # Dispatch on the claim's title only. Matching against the whole
            # prompt is wrong: the validator template itself contains words like
            # "gas", so every claim would fall into the same branch.
            claim = ""
            for line in prompt.splitlines():
                if line.startswith("- **Issue**:"):
                    claim = line.split(":", 1)[1].strip().lower()
                    break
            # Confirm reentrancy, reject the gas note, leave others uncertain.
            if "reentranc" in claim:
                return json.dumps({
                    "verdict": "confirmed", "confidence": 0.95,
                    "reason": "State written after the external call.",
                    "exploit_path": "1. deposit 2. withdraw 3. re-enter in fallback",
                    "corrected_severity": "High", "corrected_issue": "",
                })
            if "gas" in claim:
                return json.dumps({
                    "verdict": "rejected", "confidence": 0.05,
                    "reason": "Style note, not a vulnerability.",
                    "exploit_path": "", "corrected_severity": "Low", "corrected_issue": "",
                })
            return json.dumps({
                "verdict": "uncertain", "confidence": 0.5, "reason": "Depends on caller.",
                "exploit_path": "", "corrected_severity": "Medium", "corrected_issue": "",
            })

        if "reviewing the team's first pass" in system:
            with self._lock:
                self._replans += 1
                first = self._replans == 1
            if self.replan_once and first:
                return json.dumps({
                    "another_round": True, "reason": "deposit() unexamined",
                    "gaps": ["examine deposit() for overflow"],
                })
            return json.dumps({"another_round": False, "reason": "covered", "gaps": []})

        # Specialist replies, keyed off the specialty in the system prompt.
        if "Reentrancy" in system:
            return json.dumps([{
                "Issue": "Reentrancy in withdraw", "Severity": "High",
                "Description": "External call precedes the balance update.",
                "Impact": "Drain of all deposits.", "Location": "Vault.withdraw",
                "evidence": "msg.sender.call.value(amount)",
                "exploit_path": "re-enter via fallback", "confidence": 0.9,
            }])
        if "Unchecked Low-Level Calls" in system:
            # Same underlying bug, phrased differently -> must merge.
            return json.dumps([{
                "Issue": "Reentrancy in the withdraw function", "Severity": "High",
                "Description": "Return value ignored and state updated late.",
                "Impact": "Funds drained.", "Location": "Vault.withdraw",
                "evidence": "call.value", "exploit_path": "", "confidence": 0.8,
            }])
        if "Gas" in system:
            return json.dumps([{
                "Issue": "Gas optimization in deposit", "Severity": "Low",
                "Description": "Could cache storage.", "Impact": "Minor cost.",
                "Location": "Vault.deposit", "evidence": "",
                "exploit_path": "", "confidence": 0.3,
            }])
        if "Arithmetic" in system:
            return json.dumps([{
                "Issue": "Unchecked addition in deposit", "Severity": "Medium",
                "Description": "0.4.24 wraps on overflow.", "Impact": "Balance corruption.",
                "Location": "Vault.deposit", "evidence": "balances += msg.value",
                "exploit_path": "", "confidence": 0.6,
            }])
        return "[]"


def _run(specialties, replan_once=False, max_rounds=2):
    llm = FakeLLM(replan_once=replan_once)
    graph = build_graph(client=llm, specialties=specialties)
    state = new_state(CONTRACT, session_id="test", max_rounds=max_rounds)
    return llm, graph.invoke(state)


def test_graph_runs_and_ranks():
    specialties = ["reentrancy", "unchecked_low_level_calls", "arithmetic", "gas_efficiency"]
    llm, result = _run(specialties)

    findings = result["findings"]
    assert findings, "graph produced no findings"

    # Ranking: the confirmed High must come first, the rejected gas note last.
    assert findings[0]["verdict"] == "confirmed"
    assert "eentranc" in findings[0]["Issue"]
    assert findings[-1]["verdict"] == "rejected"

    # Scores must be monotonically non-increasing.
    scores = [f["score"] for f in findings]
    assert scores == sorted(scores, reverse=True), f"not ranked: {scores}"

    # And the verdict tiers must not interleave: every confirmed finding
    # outranks every uncertain one, which outranks every rejected one. This is
    # the property the Top-1/MRR metric actually rewards.
    tier = {"confirmed": 0, "uncertain": 1, "rejected": 2}
    tiers = [tier[f["verdict"]] for f in findings]
    assert tiers == sorted(tiers), f"verdict tiers interleaved: {tiers}"
    assert set(tiers) == {0, 1, 2}, f"fixture did not exercise all three tiers: {tiers}"

    # The two phrasings of the reentrancy bug must have merged.
    assert result["DuplicatesMerged"] >= 1
    assert findings[0]["corroborations"] >= 2
    print(f"  ranked: {[(f['rank'], f['Issue'][:38], f['verdict']) for f in findings]}")


def test_specialists_run_concurrently():
    """The headline win over the legacy serial `for task in tasks:` loop."""
    specialties = ["reentrancy", "unchecked_low_level_calls", "arithmetic", "gas_efficiency"]
    started = time.time()
    llm, result = _run(specialties)
    elapsed = time.time() - started

    assert llm.max_concurrent > 1, "specialists did not run in parallel"

    # Wall clock, not just a counter: the four specialists plus their
    # validations must cost far less than running every call back to back.
    serial = len(llm.calls) * FakeLLM.LATENCY
    assert elapsed < serial * 0.75, (
        f"no real speedup: {elapsed:.2f}s vs {serial:.2f}s serial"
    )
    print(
        f"  peak concurrency: {llm.max_concurrent} | "
        f"{elapsed:.2f}s vs {serial:.2f}s serial ({serial / elapsed:.1f}x)"
    )


def test_replan_loop_executes_and_terminates():
    specialties = ["reentrancy", "arithmetic"]
    llm, result = _run(specialties, replan_once=True, max_rounds=2)
    assert result["round"] == 2, f"expected 2 rounds, got {result['round']}"
    print(f"  rounds: {result['round']}")


def test_round_budget_is_respected():
    specialties = ["reentrancy"]
    # replan always asks for more, but max_rounds must stop it.
    llm = FakeLLM(replan_once=False)
    llm._respond_orig = llm._respond
    llm._respond = lambda p, s: (
        json.dumps({"another_round": True, "reason": "more", "gaps": ["again"]})
        if "reviewing the team's first pass" in s else llm._respond_orig(p, s)
    )
    graph = build_graph(client=llm, specialties=specialties)
    result = graph.invoke(new_state(CONTRACT, session_id="t", max_rounds=3))
    assert result["round"] <= 3, f"loop overran its budget: {result['round']}"
    print(f"  stopped at round {result['round']} (budget 3)")


def test_validation_is_not_repeated_across_rounds():
    specialties = ["reentrancy", "arithmetic"]
    llm, result = _run(specialties, replan_once=True, max_rounds=2)
    validator_calls = [c for c in llm.calls if "adversarial reviewer" in c]
    unique = len(result["validated_findings"])
    assert validator_calls, "validator never ran"
    assert len(validator_calls) == unique, (
        f"validator ran {len(validator_calls)}x for {unique} findings - round 1 was re-validated"
    )
    print(f"  {len(validator_calls)} validations for {unique} findings (no rework)")


def test_dedup_keeps_distinct_issues_apart():
    a = {"Issue": "Reentrancy in withdraw", "Location": "Vault.withdraw",
         "Severity": "High", "verdict": "confirmed", "confidence": 0.9, "agent": "x"}
    b = {"Issue": "Integer overflow in deposit", "Location": "Vault.deposit",
         "Severity": "Medium", "verdict": "uncertain", "confidence": 0.5, "agent": "y"}
    assert len(deduplicate([a, b])) == 2, "unrelated findings were wrongly merged"


def test_rejected_findings_are_demoted_not_deleted():
    """Recall must survive a wrong rejection: Top-5/Top-max depend on presence."""
    specialties = ["reentrancy", "gas_efficiency"]
    _, result = _run(specialties)
    verdicts = [f["verdict"] for f in result["findings"]]
    assert "rejected" in verdicts, "rejected finding was dropped instead of demoted"
    rejected_rank = next(f["rank"] for f in result["findings"] if f["verdict"] == "rejected")
    assert rejected_rank == len(result["findings"]), "rejected finding was not demoted to last"


def test_rank_score_ordering_properties():
    base = {"Severity": "Medium", "corroborations": 1}
    confirmed = {**base, "verdict": "confirmed", "confidence": 0.6}
    uncertain = {**base, "verdict": "uncertain", "confidence": 0.95}
    rejected = {**base, "verdict": "rejected", "confidence": 0.95}
    # A confirmed finding must outrank an uncertain one even when the uncertain
    # one claims higher confidence -- that is the whole point of validation.
    assert rank_score(confirmed) > rank_score(uncertain) > rank_score(rejected)


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
