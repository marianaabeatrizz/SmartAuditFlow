"""Orchestration: shared grounding, dynamic delegation, and the replan loop.

The legacy engine had a fixed ``execution_order`` list, so the "dynamic
plan-execute" behaviour the project describes did not exist -- the same five
nodes ran in the same order regardless of what was found. Here the orchestrator
chooses which specialists to wake based on the contract in front of it, and
after validation decides whether a second round is warranted.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from src.agents.base import Agent, extract_json
from src.agents.llm import LLMClient
from src.agents.specialists import CATALOGUE, DEFAULT_SPECIALTIES
from src.agents.state import Task

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------
# Prompt variants for the prompt-optimisation ablation.
#
# The published ``promptOptimization/`` module evolves exactly one prompt: the
# one that reads a contract and extracts its structure (functions, state
# variables, modifiers, events). In this architecture that is the context
# briefing, so this node is where "with and without prompt optimisation" can
# be measured end to end.
#
# Both strings below are verbatim from the published artifact:
#   SEED     -- promptOptimization/apps/config.py, seed_prompts[0]
#   EVOLVED  -- promptOptimization/result/history.json, best of generation 4
#
# Worth knowing before reading the result: the published evolution run reports
# best fitness falling every generation (0.565 -> 0.345 -> 0.315 -> 0.265) and
# population diversity collapsing to 0.009, on a training set of two synthetic
# toy contracts. The "optimised" prompt is a shorter, vaguer paraphrase of its
# own seed. So this ablation is a test of whether that optimiser helped, not an
# assumption that it did.
CONTEXT_PROMPT_SEED = (
    "Analyze the provided smart contract. Focus on its function definitions, "
    "state variables, modifiers, and events. Be precise."
)
CONTEXT_PROMPT_EVOLVED = (
    "inspect the provided smart contract. Highlight functions, variables, "
    "modifiers, and events. Be precise."
)
CONTEXT_PROMPT_VARIANTS = ("default", "seed", "evolved")


class ContextAgent(Agent):
    """Produces the shared brief every specialist is grounded on.

    Running this once and handing the result to all specialists is much cheaper
    than making each of twelve agents independently work out what the contract
    does, and it stops them disagreeing about basic facts.

    ``prompt_variant`` selects which instruction the node uses. ``default`` is
    this architecture's own briefing; ``seed`` and ``evolved`` are the before
    and after of the paper's prompt optimiser, for the ablation. Only the task
    instruction changes between variants -- same model, same system prompt,
    same token budget -- so a difference in the result is attributable to the
    prompt and to nothing else.
    """

    role = "orchestrator"
    name = "orchestrator:context"

    def __init__(self, client: LLMClient, model: Optional[str] = None,
                 prompt_variant: str = "default"):
        super().__init__(client, model)
        if prompt_variant not in CONTEXT_PROMPT_VARIANTS:
            raise ValueError(
                f"unknown context prompt variant {prompt_variant!r}; "
                f"expected one of {CONTEXT_PROMPT_VARIANTS}"
            )
        self.prompt_variant = prompt_variant

    @property
    def system_prompt(self) -> str:
        return (
            "You are the lead auditor opening an engagement. You produce the "
            "briefing the rest of the team works from. You are precise and "
            "factual; you do not speculate about vulnerabilities here."
        )

    def build_prompt(self, code: str, static_tool: str = "") -> str:
        """The instruction sent to the model, per variant."""
        tool_block = (
            f"Static analyzer output:{chr(10)}{static_tool}{chr(10)}"
            if static_tool else ""
        )
        if self.prompt_variant != "default":
            instruction = (CONTEXT_PROMPT_SEED if self.prompt_variant == "seed"
                           else CONTEXT_PROMPT_EVOLVED)
            return f"""{instruction}

```solidity
{code}
```
{tool_block}"""

        return f"""Read this contract and write a briefing for the audit team.

```solidity
{code}
```
{tool_block}Cover, concisely:

1. **Purpose** - what the contract does, and which standards it implements.
2. **Solidity version** - state it exactly. It decides whether arithmetic wraps.
3. **Value flow** - where funds enter, where they leave, who can move them.
4. **Trust model** - privileged roles and what each can do.
5. **External surface** - every external call and every contract it depends on.
6. **State invariants** - what must always hold for this contract to be correct.

Be factual and specific. Name functions. Under 400 words. No vulnerability
analysis - that is the specialists' job."""

    def run(self, state: Dict[str, Any]) -> Dict[str, Any]:
        code = state.get("code_snippet", "")
        static_tool = state.get("static_tool", "")

        prompt = self.build_prompt(code, static_tool)

        try:
            response = self.ask(prompt, max_tokens=1200)
        except Exception as exc:  # noqa: BLE001
            logger.warning("context brief failed: %s", exc)
            return {
                "context_brief": "",
                "errors": [f"{self.name}: {exc}"],
                "agent_log": [self.log_entry(status="failed", error=str(exc))],
            }

        return {
            "context_brief": response.text.strip(),
            "token_usage": response.total_tokens,
            "agent_log": [
                self.log_entry(
                    status="ok",
                    tokens=response.total_tokens,
                    seconds=round(response.duration, 2),
                )
            ],
        }


class PlannerAgent(Agent):
    """Decides which specialists to dispatch, and what to tell each one."""

    role = "orchestrator"
    name = "orchestrator:planner"

    def __init__(
        self,
        client: LLMClient,
        model: Optional[str] = None,
        available: Optional[List[str]] = None,
        always_dispatch: bool = True,
    ):
        super().__init__(client, model)
        self.available = available or DEFAULT_SPECIALTIES
        # When True, every specialty runs regardless of the planner's opinion,
        # and the planner only supplies targeting instructions. Recall is the
        # thing we cannot recover later, so this defaults on; set it False to
        # trade coverage for cost.
        self.always_dispatch = always_dispatch

    @property
    def system_prompt(self) -> str:
        return (
            "You are the lead auditor allocating work. You know your team's "
            "specialties and you give each member a concrete, targeted brief "
            "rooted in this specific contract."
        )

    def run(self, state: Dict[str, Any]) -> Dict[str, Any]:
        rnd = int(state.get("round", 0)) + 1
        brief = state.get("context_brief", "")
        gaps: List[str] = state.get("coverage_gaps", []) or []

        roster = "\n".join(
            f"- {k}: {CATALOGUE[k].title} -- {CATALOGUE[k].expertise}"
            for k in self.available
        )
        gap_block = ""
        if gaps:
            gap_block = (
                "\n## Unresolved from the previous round\n"
                + "\n".join(f"- {g}" for g in gaps)
                + "\nTarget these specifically.\n"
            )

        prompt = f"""## Contract briefing
{brief}

## Your team
{roster}
{gap_block}
For each specialist that is genuinely relevant to THIS contract, write a
targeted assignment: which functions to examine and what you suspect is worth
checking there. Ground every assignment in something the briefing actually says.

Return ONLY a JSON array:

[
  {{
    "specialty": "one key from the team list above",
    "instruction": "what this specialist should examine, specific to this contract",
    "focus": "contract.function names to concentrate on",
    "priority": "high | medium | low"
  }}
]

Order the array by priority: what is most likely to be genuinely exploitable
here comes first. No prose outside the JSON."""

        instructions: Dict[str, Task] = {}
        tokens = 0
        status = "ok"
        try:
            response = self.ask(prompt, max_tokens=2000)
            tokens = response.total_tokens
            parsed = extract_json(response.text, expect="array") or []
            for i, item in enumerate(parsed):
                if not isinstance(item, dict):
                    continue
                key = str(item.get("specialty", "")).strip()
                if key not in CATALOGUE:
                    continue
                priority = str(item.get("priority", "medium")).strip().lower()
                instructions[key] = Task(
                    id=f"r{rnd}-{key}",
                    specialty=key,
                    instruction=str(item.get("instruction", "")).strip(),
                    focus=str(item.get("focus", "")).strip(),
                    priority=priority if priority in ("high", "medium", "low") else "medium",
                    round=rnd,
                )
        except Exception as exc:  # noqa: BLE001
            # A planner failure must not sink the run: fall back to dispatching
            # the full roster with no targeting.
            logger.warning("planner failed, falling back to full roster: %s", exc)
            status = "failed-open"

        plan = self._assemble(instructions, gaps, rnd)
        return {
            "plan": plan,
            "round": rnd,
            "token_usage": tokens,
            "agent_log": [
                self.log_entry(
                    status=status,
                    round=rnd,
                    dispatched=len(plan),
                    targeted=len(instructions),
                    tokens=tokens,
                )
            ],
        }

    def _assemble(
        self, instructions: Dict[str, Task], gaps: List[str], rnd: int
    ) -> List[Task]:
        """Merge planner targeting with the coverage guarantee."""
        if self.always_dispatch and not gaps:
            keys = list(self.available)
        elif gaps:
            # A replan round is narrow by design: only what the planner asked for.
            keys = [k for k in self.available if k in instructions]
        else:
            keys = [k for k in self.available if k in instructions] or list(self.available)

        plan: List[Task] = []
        for key in keys:
            plan.append(
                instructions.get(
                    key,
                    Task(
                        id=f"r{rnd}-{key}",
                        specialty=key,
                        instruction="",
                        focus="",
                        priority="medium",
                        round=rnd,
                    ),
                )
            )
        order = {"high": 0, "medium": 1, "low": 2}
        plan.sort(key=lambda t: order.get(t.get("priority", "medium"), 1))
        return plan


class ReplanAgent(Agent):
    """Decides, after validation, whether another round is worth its cost."""

    role = "orchestrator"
    name = "orchestrator:replan"

    @property
    def system_prompt(self) -> str:
        return (
            "You are the lead auditor reviewing the team's first pass. You are "
            "decisive and cost-aware: another round must be justified by a "
            "concrete, specific lead, not by a general wish to be thorough."
        )

    def run(self, state: Dict[str, Any]) -> Dict[str, Any]:
        rnd = int(state.get("round", 0))
        max_rounds = int(state.get("max_rounds", 2))
        validated = state.get("validated_findings", []) or []

        if rnd >= max_rounds:
            return {
                "coverage_gaps": [],
                "agent_log": [
                    self.log_entry(status="stop", reason="round budget exhausted", round=rnd)
                ],
            }

        summary = "\n".join(
            f"- [{f.get('Severity')}/{f.get('verdict', '?')}/conf={f.get('confidence', 0):.2f}] "
            f"{f.get('Issue')} ({f.get('Location')})"
            for f in validated[:40]
        ) or "(nothing confirmed)"

        prompt = f"""## Contract briefing
{state.get('context_brief', '')}

## What the first pass concluded
{summary}

Decide whether a second, targeted round would plausibly find something the
first pass missed. Justify it only with a specific lead -- an unexamined
function, an interaction between two findings, a rejected finding that
deserves a closer look.

Return ONLY JSON:

{{
  "another_round": true | false,
  "reason": "one sentence",
  "gaps": ["specific thing to re-examine", "..."]
}}

Default to false. Return true only if you can name a concrete lead."""

        try:
            response = self.ask(prompt, max_tokens=800)
            decision = extract_json(response.text, expect="object") or {}
        except Exception as exc:  # noqa: BLE001
            logger.warning("replan decision failed: %s", exc)
            return {
                "coverage_gaps": [],
                "agent_log": [self.log_entry(status="failed", error=str(exc))],
            }

        wants_more = bool(decision.get("another_round"))
        gaps = [str(g) for g in (decision.get("gaps") or []) if str(g).strip()]
        if not wants_more or not gaps:
            gaps = []

        return {
            "coverage_gaps": gaps,
            "token_usage": response.total_tokens,
            "agent_log": [
                self.log_entry(
                    status="ok",
                    another_round=bool(gaps),
                    reason=str(decision.get("reason", ""))[:200],
                    round=rnd,
                )
            ],
        }
