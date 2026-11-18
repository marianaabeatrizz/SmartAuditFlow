# Multi-Agent Architecture

How the audit engine works after the move from a linear pipeline to a
hierarchical multi-agent graph, and why each decision was made.

## Why the architecture changed

The original engine (`src/workflow/`) was described as a dynamic plan-execute
framework, but in practice it was a fixed five-node pipeline:

```
InitialAnalysis -> AuditPlan -> ParameterExtractor -> Iteration -> FormatConverter
```

Five properties limited it:

| Limitation | Where | Consequence |
|---|---|---|
| Tasks ran serially | `nodes.py` `for task in tasks:` | 12 tasks = 12 sequential API calls |
| One generic prompt for every task | `IterationNode` | No agent had an edge on any vulnerability class |
| No verification | — | Every candidate reached the report unchallenged |
| `execution_order` was a static list | `engine.py` | The "dynamic" replanning did not exist |
| One model for all stages | `set_preferred_model` on a singleton | Lost the plan-execute cost advantage, and two concurrent sessions clobbered each other's model choice |

## What the benchmark says the goal is

Re-scoring `evaluation/results/standardVuln` shows the published
SmartAuditFlow run reaching **Top-5 = 0.993** but **Top-1 = 0.664**.

The true vulnerability is almost always *found*. It is simply not ranked
first. Recall is saturated; all remaining headroom is in **ranking and false
positive suppression**. That single fact drives the design: the validator is
not a safety feature bolted on the side, it is the component that moves the
metric.

## The graph

```
START -> context -> plan -> [specialist x N, parallel] -> validate
             ^                                             |   ^
             |                                             |   |
             |                           did not pass ->  refine
             |                                             |
             |                                             v
             +-------------------- replan <----------------+
                                     |
                                     v  (no lead left, or budget spent)
                                 aggregate -> END
```

Two loops, nested on purpose:

* **Inner** (`validate -> refine -> validate`), bounded by `max_refine_rounds`:
  a claim the panel did not confirm goes back to the specialist that made it,
  with the objection attached. Repairing a finding we already have is cheaper
  than commissioning new ones, and it is what moves Top-1.
* **Outer** (`replan -> plan`), bounded by `max_rounds`: the orchestrator
  widens the plan when a concrete coverage gap remains.

| Node | Agent | Role |
|---|---|---|
| `context` | `ContextAgent` | One shared briefing: purpose, Solidity version, value flow, trust model, invariants. Computed once instead of twelve times. |
| `plan` | `PlannerAgent` | Writes a targeted assignment per specialist, grounded in the briefing. |
| `specialist` | `SpecialistAgent` x12 | One vulnerability class each, fanned out concurrently via `Send`. |
| `validate` | `ValidatorAgent` | An **independent** panel adversarially tries to refute each candidate, then scores confidence. |
| `refine` | `RefinementAgent` | Re-runs the proposing specialist against the judge's objection. It must answer with code it did not cite before, or withdraw. |
| `replan` | `ReplanAgent` | Decides whether a second round has a concrete lead worth paying for. |
| `aggregate` | `aggregator.finalize` | Deduplicates, merges corroboration, ranks. |

## The specialist catalogue

Deliberately mirrors the DASP-10 taxonomy that `standardVuln` uses, so
coverage is auditable rather than assumed. Ordered by share of the benchmark:

| Specialist | Benchmark share |
|---|---|
| `unchecked_low_level_calls` | 52 |
| `reentrancy` | 31 |
| `access_control` | 18 |
| `arithmetic` | 15 |
| `bad_randomness` | 8 |
| `denial_of_service` | 6 |
| `time_manipulation` | 5 |
| `front_running` | 4 |
| `data_handling` | 1 |
| `business_logic`, `oracle_pricing`, `gas_efficiency` | — (for protocol-scale audits) |

## Judge independence

A model asked to grade its own output systematically prefers it
(self-preference bias). Here that bias runs in the dangerous direction: it
inflates the confidence score the entire ranking is built on, so a
self-validated run would post better Top-1/MRR without auditing any better.

Judges are therefore selected **per finding**, against whoever proposed it:

| Level | Meaning |
|---|---|
| `cross-provider` | Judge is from a different provider family. The target. |
| `cross-model` | Same family, different checkpoint. Weak -- shared lineage. |
| `self` | The proposer judged itself. Biased, and flagged. |

Independence is classified at the **provider family** level, not the model id:
`gpt-4o` judging `gpt-4o-mini` shares training data and alignment, so the
self-preference effect does not vanish just because the checkpoint differs.

Pinning a model for a benchmark (`build_graph(model=...)`) pins the
*proposers* only. The judge is still resolved independently, so the
like-for-like comparison does not quietly reintroduce self-judgement in
exactly the run whose numbers get published.

`panel_size > 1` convenes several judges drawn from distinct families. They
run concurrently and never see each other's answers, so no judge is anchored
on a prior ruling. Their level of agreement is kept:

- Majority verdict wins.
- **A tie resolves to `uncertain`, never `rejected`.** Rejection demotes a
  finding to the bottom of the report, so resolving a deadlock as rejection
  would bury a possibly-real vulnerability and cost exactly the Top-1/MRR the
  design is chasing. Rejection must be earned by a majority.
- Disagreement pulls confidence toward the middle (`x0.6 + 0.4 x agreement`).

Every finding carries `judged_by`, `independence`, `self_validated` and
`judge_agreement`; the run reports `IndependenceLevel` (the weakest link, not
the best) and `SelfValidatedCount`. When no independent judge is configured,
`strict_independence=True` raises rather than proceeding; non-strict proceeds
but flags every affected finding and surfaces the caveat in `errors`.

### Corroboration is discounted the same way

Two specialists agreeing is only strong evidence when they are not the same
model. Same-model agreement shares blind spots and tends to be wrong together,
so `rank_score` pays a third of the credit for same-family corroboration that
it pays for cross-provider agreement (`corroborating_families`).

## Ranking

`validator.rank_score` orders the report:

```
score = verdict_weight x confidence
      + severity_bonus
      + corroboration_bonus   (diminishing, capped at 3 extra agents)
      + 0.05 if a concrete exploit path exists
```

Verdict weights (`confirmed` 1.0, `uncertain` 0.45, `rejected` 0.08) ensure a
confirmed finding always outranks an uncertain one regardless of the
confidence the proposing specialist claimed. The corroboration bonus is
discounted when the corroborating agents share a provider family.

**Rejected findings are demoted, never deleted.** Top-5 and Top-max measure
whether the true issue is present at all, so dropping a wrongly-rejected
finding destroys recall that cannot be recovered, while demoting it costs
only rank.

## The refinement loop

`Top-5 = 0.993` against `Top-1 = 0.664` says the true vulnerability is almost
always found and merely mis-ranked. A plausible cause is a real finding
*described badly* — wrong function name, vague mechanism, no exploit path —
which an honest judge is right to refuse. The loop gives the specialist one
chance to answer that specific objection.

The obvious failure mode is verification shopping: reword until some judge
says yes. Three guards:

1. **Bounded.** `max_refine_rounds` caps revisions; default 1, `0` disables it.
2. **New evidence required.** Rewording is explicitly not an answer. The
   revision must quote code it did not cite before, and that quote is checked
   against the contract source — a fabricated line is discarded and the
   original ruling stands.
3. **No judge shopping.** A revision goes back to the judge that objected,
   not to a fresh one. The reviewer who raised the objection decides whether
   it was met.

A specialist may also **withdraw**, which is recorded rather than hidden.
Every refined finding carries `refined`, `refine_attempts` and `objection`, and
the run reports `RefinedCount` / `WithdrawnCount`, so an ablation can measure
whether the loop earns its cost instead of assuming it does.

### Identity

Findings carry a `uid` assigned at creation, and everything downstream keys off
it. Deriving identity from the text does not work once revisions exist: a
revised claim legitimately changes its title and location, and content-derived
identity made it look like a brand-new finding — which duplicated it and lost
the original's verdict.

## Model routing

Each role resolves its model independently, with no shared mutable state:

```
override argument  >  SAF_MODEL_<ROLE> env var  >  DEFAULT_ROLE_MODELS
```

Defaults put a strong model on planning and judging and a cheaper one on the
bulk specialist work. Pin every role to one model with `build_graph(model=...)`
when you need a like-for-like comparison against a single-model baseline.

`LLMClient(strict=True)` refuses to silently answer with a different
provider's model. **Use strict mode for anything you intend to report** --
the legacy `call_model_sync` falls back across every configured provider, so a
run labelled `gpt-4o` could in fact have been answered by Claude.

## Running it

```bash
# API: architecture is selectable per request
curl -X POST localhost:5000/api/audit/start \
  -H 'Content-Type: application/json' \
  -d '{"code_snippet":"...", "architecture":"multi-agent", "max_rounds":2}'

# Tests (no API keys, no network)
.venv/Scripts/python.exe tests/test_agents_graph.py
.venv/Scripts/python.exe tests/test_agents_engine.py

# Benchmark: re-score published baselines, no cost
.venv/Scripts/python.exe -m benchmark.run --rescore-only

# Tests: judge independence
.venv/Scripts/python.exe tests/test_judge_independence.py

# Benchmark: run the new architecture and compare.
# Refuses to run if no judge independent of --model is configured.
.venv/Scripts/python.exe -m benchmark.run --limit 20 --model openai

# Convene a 3-judge panel from different providers
.venv/Scripts/python.exe -m benchmark.run --limit 20 --model openai --panel-size 3
```

Use `run_server.py`, not `src/main.py`, when verifying behaviour. `main.py`
runs with `debug=True`, and the werkzeug reloader has left orphaned processes
bound to port 5000 on this machine -- requests then get answered by a stale
server running old code, which is extremely confusing to debug.

## Measurement caveat

The published `Rank` column **cannot be reproduced** from the artifact:
matching the ground-truth `Issue` against the stored `Findings` reproduces
only ~61% of published ranks, and the disagreements are systematic. A further
42 of 145 rows have an unparseable `Findings` cell.

So the new numbers must not be compared against the published numbers
directly. `benchmark/run.py` instead re-scores the stored baseline findings
and the new findings with one identical, documented rule (`benchmark/scoring.py`).
That comparison is internally valid; the published one would not be.

Separately, four published result files (`claude_3_7_sonnet`,
`llama4_maverick`, `mistral_large_big`, `smartauditflow_gemini`) are
byte-identical in every metric column, as are `gemini20_flash` and
`gemini25_pro`. They appear to be copies rather than independent runs. Do not
cite them as separate baselines.
