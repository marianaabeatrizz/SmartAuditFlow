"""The specialist swarm.

Each specialist owns one vulnerability class and carries a prompt written for
that class alone. The legacy ``IterationNode`` sent one generic prompt for every
task, which gave the model no particular edge on any of them.

The catalogue deliberately mirrors the DASP-10 taxonomy used by the
``standardVuln`` benchmark (``unchecked_low_level_calls``, ``reentrancy``,
``access_control``, ``arithmetic``, ``bad_randomness``, ``denial_of_service``,
``time_manipulation``, ``front_running``, plus ``short_addresses``/``other``),
with three extra agents for the protocol-scale contracts in the code4rena and
sherlock sets. Aligning the swarm to the benchmark taxonomy means coverage is
auditable rather than assumed.
"""
from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from src.agents.base import Agent, extract_json
from src.agents.llm import LLMClient
from src.agents.state import Finding, Task

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Specialty:
    key: str
    title: str
    expertise: str      # what this agent is primed to see
    checklist: str      # concrete things to verify
    benchmark_weight: int = 0   # occurrences in standardVuln, for plan ordering


CATALOGUE: Dict[str, Specialty] = {
    "unchecked_low_level_calls": Specialty(
        key="unchecked_low_level_calls",
        title="Unchecked Low-Level Calls",
        expertise=(
            "the return values of call/delegatecall/callcode/send, and what "
            "happens when the callee reverts but the caller carries on regardless"
        ),
        checklist=(
            "- Every `.call(...)`, `.delegatecall(...)`, `.callcode(...)` and `.send(...)`: "
            "is the boolean return value checked, or silently discarded?\n"
            "- Does execution continue as if the transfer succeeded when it did not?\n"
            "- Is `transfer()` used where the 2300 gas stipend will break a contract recipient?\n"
            "- Does `delegatecall` run attacker-influenced code against this contract's storage?\n"
            "- Are return values decoded from tokens that return nothing (non-standard ERC-20)?"
        ),
        benchmark_weight=52,
    ),
    "reentrancy": Specialty(
        key="reentrancy",
        title="Reentrancy",
        expertise="control flow that leaves the contract before its state is settled",
        checklist=(
            "- Find every external call, then ask what state is written AFTER it.\n"
            "- Is checks-effects-interactions violated anywhere?\n"
            "- Cross-function reentrancy: can a different function be re-entered "
            "while this one holds inconsistent state?\n"
            "- Read-only reentrancy: can a view function be called mid-update and "
            "return a value another protocol trusts?\n"
            "- Is there a reentrancy guard, and does it actually cover every path?\n"
            "- ERC-777/ERC-721 hooks (`tokensReceived`, `onERC721Received`) that hand "
            "control to the attacker."
        ),
        benchmark_weight=31,
    ),
    "access_control": Specialty(
        key="access_control",
        title="Access Control & Authorization",
        expertise="who is allowed to call what, and the paths that bypass the intent",
        checklist=(
            "- Every state-changing function: what restricts it? If nothing, is that deliberate?\n"
            "- `tx.origin` used for authorization (phishable).\n"
            "- Constructor naming bugs / uninitialised owner / unprotected initializer.\n"
            "- `selfdestruct`, upgrade and withdrawal paths and who reaches them.\n"
            "- Missing or wrong modifier on privileged setters.\n"
            "- Default-public visibility and functions that should be internal.\n"
            "- Can ownership be transferred to address(0) or seized?"
        ),
        benchmark_weight=18,
    ),
    "arithmetic": Specialty(
        key="arithmetic",
        title="Arithmetic",
        expertise="integer overflow, underflow, precision loss and unit confusion",
        checklist=(
            "- Solidity version: before 0.8.0 arithmetic wraps silently. State the version.\n"
            "- `unchecked { }` blocks in 0.8+ and whether the bound is actually guaranteed.\n"
            "- Subtraction that can underflow (balance -= amount without a prior check).\n"
            "- Division before multiplication, causing truncation to zero.\n"
            "- Decimal/precision mismatches between tokens (6 vs 18 decimals).\n"
            "- Unsafe casts that silently truncate (uint256 -> uint128 -> uint64)."
        ),
        benchmark_weight=15,
    ),
    "bad_randomness": Specialty(
        key="bad_randomness",
        title="Weak Randomness",
        expertise="on-chain sources of entropy that a miner, validator or caller can predict",
        checklist=(
            "- `block.timestamp`, `block.number`, `blockhash`, `block.difficulty`/`prevrandao`, "
            "`block.coinbase` or `msg.sender` used as a randomness source.\n"
            "- Seeds that are readable on-chain before the draw resolves.\n"
            "- `blockhash` of a block older than 256 (returns zero).\n"
            "- Can a caller simulate the outcome and revert on a losing result?\n"
            "- Is the payoff large enough to make manipulation economically rational?"
        ),
        benchmark_weight=8,
    ),
    "denial_of_service": Specialty(
        key="denial_of_service",
        title="Denial of Service",
        expertise="paths an attacker can wedge permanently or make unaffordable",
        checklist=(
            "- Unbounded loops over arrays an attacker can grow.\n"
            "- Push-payment patterns where one reverting recipient blocks everyone.\n"
            "- External calls in a loop.\n"
            "- Gas-griefing: can an attacker force the caller to burn gas?\n"
            "- State that can be locked with no recovery path (owner set to a dead address).\n"
            "- Does the contract depend on an external call that can be made to always revert?"
        ),
        benchmark_weight=6,
    ),
    "time_manipulation": Specialty(
        key="time_manipulation",
        title="Timestamp Dependence",
        expertise="logic that trusts block time within a window a validator controls",
        checklist=(
            "- `block.timestamp`/`now` used in a comparison that gates value.\n"
            "- How much drift can a validator introduce, and is the window tight enough to matter?\n"
            "- `block.number` used as a clock with an assumed block time.\n"
            "- Deadlines, lock expiry, auction endings, rate accrual."
        ),
        benchmark_weight=5,
    ),
    "front_running": Specialty(
        key="front_running",
        title="Front-Running & MEV",
        expertise="what an observer of the mempool can do between submission and inclusion",
        checklist=(
            "- Is the outcome sensitive to transaction ordering?\n"
            "- Missing slippage / minimum-output parameters on swaps.\n"
            "- ERC-20 `approve` race condition.\n"
            "- Commit-reveal absent where the result is valuable and predictable.\n"
            "- Can a sandwich extract value from a user at the protocol's expense?"
        ),
        benchmark_weight=4,
    ),
    "data_handling": Specialty(
        key="data_handling",
        title="Data Encoding & Call Payloads",
        expertise="malformed input, short addresses, signatures and encoding ambiguity",
        checklist=(
            "- Short-address / padding assumptions on `msg.data` length.\n"
            "- Signature replay: is there a nonce, a chain id and a domain separator?\n"
            "- `ecrecover` returning address(0) on malformed input, unchecked.\n"
            "- Signature malleability (s-value not constrained).\n"
            "- `abi.encodePacked` with two dynamic types producing hash collisions."
        ),
        benchmark_weight=1,
    ),
    # --- beyond DASP-10: needed for the protocol-scale audit sets ---
    "business_logic": Specialty(
        key="business_logic",
        title="Business Logic & Accounting",
        expertise="whether the code does what the protocol economically intends",
        checklist=(
            "- Does each function preserve the protocol's accounting invariants?\n"
            "- Can a user end up with more value than they put in? Trace the path.\n"
            "- Are fee, reward and share calculations correct at the boundaries "
            "(first depositor, zero supply, full withdrawal)?\n"
            "- State machine: can a step be skipped, repeated or run out of order?\n"
            "- Rounding that consistently favours the user over the protocol."
        ),
    ),
    "oracle_pricing": Specialty(
        key="oracle_pricing",
        title="Oracle & Price Manipulation",
        expertise="where the contract learns prices and how that source can be moved",
        checklist=(
            "- Spot price read from an AMM reserve (manipulable within one transaction).\n"
            "- Flash-loan amplified manipulation of any price input.\n"
            "- Chainlink feeds without staleness or round-completeness checks.\n"
            "- Missing bounds on an oracle answer; negative or zero prices.\n"
            "- Single oracle with no fallback."
        ),
    ),
    "gas_efficiency": Specialty(
        key="gas_efficiency",
        title="Gas & Resource Efficiency",
        expertise="cost, not security -- reported at low severity",
        checklist=(
            "- Storage reads inside loops that could be cached in memory.\n"
            "- State variables that could be immutable or constant.\n"
            "- Redundant SLOADs and unnecessary storage writes.\n"
            "- Struct packing opportunities."
        ),
    ),
}

# Ordered by how much of the benchmark each class accounts for, so that when a
# run is budget-capped we keep the agents that matter most.
DEFAULT_SPECIALTIES: List[str] = [
    k for k, _ in sorted(CATALOGUE.items(), key=lambda kv: -kv[1].benchmark_weight)
]

_FINDING_SCHEMA = """
Return ONLY a JSON array. Each element must have exactly these keys:

[
  {
    "Issue": "short, specific title naming the actual flaw",
    "Severity": "High | Medium | Low",
    "Description": "what is wrong and why, citing the code you read",
    "Impact": "concrete consequence if exploited",
    "Location": "contract.function, and line numbers if determinable",
    "evidence": "the exact line(s) of code that prove it",
    "exploit_path": "ordered steps an attacker takes, or \\"\\" if none exists",
    "confidence": 0.0
  }
]

Rules:
- "confidence" is your own estimated probability from 0.0 to 1.0 that this is a
  real, exploitable flaw rather than a theoretical note. Be honest: a wrong
  high-confidence claim is worse than an honest low one.
- Report ONLY issues in your assigned specialty. Another agent covers the rest.
- If the contract is sound in your specialty, return exactly: []
- Never invent a line number or a function that is not in the code.
- No prose outside the JSON array.
"""


class SpecialistAgent(Agent):
    """Audits one vulnerability class."""

    role = "specialist"

    def __init__(self, specialty: Specialty, client: LLMClient, model: Optional[str] = None):
        super().__init__(client, model)
        self.specialty = specialty
        self.name = f"specialist:{specialty.key}"

    @property
    def system_prompt(self) -> str:
        return (
            f"You are a smart contract security auditor specialising in "
            f"{self.specialty.title}. Your expertise is {self.specialty.expertise}. "
            "You read code line by line and only report what the code actually "
            "supports. You never pad a report to look thorough -- reporting a "
            "non-issue is a failure, not caution."
        )

    def build_prompt(self, state: Dict[str, Any], task: Optional[Task] = None) -> str:
        code = state.get("code_snippet", "")
        static_tool = state.get("static_tool", "")
        brief = state.get("context_brief", "")

        sections = [
            f"# Your specialty: {self.specialty.title}\n",
            f"## What to verify\n{self.specialty.checklist}\n",
        ]
        if brief:
            sections.append(f"## Shared context from the orchestrator\n{brief}\n")
        if task and task.get("instruction"):
            sections.append(
                f"## Your specific assignment\n{task['instruction']}\n"
                + (f"Focus on: {task['focus']}\n" if task.get("focus") else "")
            )
        if static_tool:
            sections.append(
                f"## Static analyzer output\n{static_tool}\n"
                "Treat this as a hint, not as truth. Static tools produce false "
                "positives; confirm or dismiss each item against the code.\n"
            )
        sections.append(f"## Contract under audit\n```solidity\n{code}\n```\n")
        sections.append(_FINDING_SCHEMA)
        return "\n".join(sections)

    def run(self, state: Dict[str, Any]) -> Dict[str, Any]:
        task: Optional[Task] = state.get("current_task")
        rnd = int(state.get("round", 0))
        try:
            response = self.ask(self.build_prompt(state, task))
        except Exception as exc:  # noqa: BLE001
            logger.warning("%s failed: %s", self.name, exc)
            return {
                "raw_findings": [],
                "errors": [f"{self.name}: {exc}"],
                "agent_log": [self.log_entry(status="failed", error=str(exc))],
            }

        parsed = extract_json(response.text, expect="array")
        if parsed is None:
            # A specialist that returns no parseable JSON is treated as "found
            # nothing" rather than failing the run; the orchestrator sees the
            # gap in the log and can re-dispatch.
            logger.info("%s returned unparseable output", self.name)
            return {
                "raw_findings": [],
                "token_usage": response.total_tokens,
                "agent_log": [
                    self.log_entry(status="unparseable", chars=len(response.text))
                ],
            }

        findings = [
            self._normalise(item, task, rnd) for item in parsed if isinstance(item, dict)
        ]
        return {
            "raw_findings": findings,
            "token_usage": response.total_tokens,
            "agent_log": [
                self.log_entry(
                    status="ok",
                    findings=len(findings),
                    tokens=response.total_tokens,
                    seconds=round(response.duration, 2),
                )
            ],
        }

    def _normalise(self, item: Dict[str, Any], task: Optional[Task], rnd: int) -> Finding:
        severity = str(item.get("Severity", "Medium")).strip().title()
        if severity in ("Critical", "Severe"):
            severity = "High"
        if severity not in ("High", "Medium", "Low"):
            severity = "Medium"

        try:
            confidence = float(item.get("confidence", 0.5))
        except (TypeError, ValueError):
            confidence = 0.5
        confidence = min(max(confidence, 0.0), 1.0)

        return Finding(
            uid=uuid.uuid4().hex[:12],
            Issue=str(item.get("Issue", "Unspecified issue")).strip(),
            Severity=severity,
            Description=str(item.get("Description", "")).strip(),
            Impact=str(item.get("Impact", "")).strip(),
            Location=str(item.get("Location", "")).strip(),
            agent=self.name,
            specialty=self.specialty.key,
            task_id=(task or {}).get("id", ""),
            round=rnd,
            # The judge needs to know who proposed this in order to stay
            # independent of them.
            proposer_model=self.model,
            confidence=confidence,
            exploit_path=str(item.get("exploit_path", "")).strip(),
            rationale=str(item.get("evidence", "")).strip(),
            corroborations=1,
        )


def build_specialists(
    client: LLMClient,
    keys: Optional[List[str]] = None,
    model: Optional[str] = None,
) -> Dict[str, SpecialistAgent]:
    keys = keys or DEFAULT_SPECIALTIES
    return {
        k: SpecialistAgent(CATALOGUE[k], client, model) for k in keys if k in CATALOGUE
    }
