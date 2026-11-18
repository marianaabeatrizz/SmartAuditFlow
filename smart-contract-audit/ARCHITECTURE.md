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

## The provider layer

A provider is configuration, not code. Anything with an OpenAI-compatible
endpoint becomes available by adding a key to `.env`; `check_providers.py`
reports what is configured, whether independent judging is possible at all,
and (with `--probe`) whether each key really answers. It exits non-zero when
no two configured providers come from different families, because a run in
that state cannot produce an independently judged number.

Each provider declares a `family`. That field, not the model id, is what
`independence_of()` uses, and keeping it honest matters: NVIDIA also serves
Gemma and Mistral, so judging with those would report `family=nvidia` while
the model's lineage was Google's or Mistral's. The registry therefore
defaults NVIDIA to a Nemotron, which is NVIDIA's own.

### Rate limits are measured, not trusted

Published limits were wrong twice here, and both times the run looked like a
result rather than an outage:

- Gemini's 500 requests/day was missing from the registry entirely. A
  143-contract run burned through it and reported every contract as
  independently judged while two thirds of its judge calls had failed.
- NVIDIA's "40/min", taken from a third-party list, produced 1,680
  rejections under sustained load — though 20 concurrent calls and ~77/min
  in short bursts both pass. The real constraint is sustained volume, which
  no `rpm` number expresses.

So `RateLimiter` measures. Every rate rejection halves the provider's
effective rate; 25 consecutive successes walk it back up by 25%, never past
what was configured. That is AIMD, the same principle TCP uses on a link of
unknown capacity. The floor of 2/min keeps a throttled run moving instead of
deadlocking.

The limiter also counts **tokens**, not just requests. Groq's binding limit
is 8k/min and 200k/day; a run can sit comfortably inside 30 req/min and still
exhaust the daily token budget, which is exactly what happened.

### Three kinds of 429

They look alike and must be handled differently. Conflating them cost most of
48 agent-hours in one run:

| meaning | example | response |
|---|---|---|
| slow down | `tokens per minute (TPM): Limit 8000` | throttle, back off, retry |
| daily allowance gone | `free_tier_requests, limit: 500 ... retry in 12h37m` | fail immediately |
| billing cap | `organization_spend_limit_exceeded` | fail immediately |

Only the first clears within a run. Backing off from the other two burns
wall-clock to fail anyway. A dropped TCP connection, by contrast, *is*
retryable and was not treated as such at first — each drop silently discarded
a finding, which reads as the model finding nothing rather than as an error.

### Cache

Responses are cached on disk, keyed on provider, model, system prompt, user
prompt, temperature and max_tokens — everything that changes the answer, so a
changed prompt is a miss rather than a stale hit. Debugging does not burn a
daily quota, and re-running after a fix only pays for what actually changed.
The cache is gitignored; it reaches tens of megabytes and holds contract
source inside prompts.

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

# What is configured, and does each key actually answer?
.venv/Scripts/python.exe check_providers.py --probe

# The two ablation arms. --judge pins the judge pool, which is needed when
# the automatically preferred provider has no quota left. Judging is capped
# to the 8 most confident findings: only the head of the ranking can move
# Top-1/Top-5/MRR, so a verdict on the 18th finding cannot change a metric,
# and the saved calls are what keeps an independent judge affordable.
.venv/Scripts/python.exe -m benchmark.run --model openai --judge nvidia \
    --judge-top-n 8 --max-refine-rounds 1   # arm A: refinement loop on
.venv/Scripts/python.exe -m benchmark.run --model openai --judge nvidia \
    --judge-top-n 8 --max-refine-rounds 0   # arm B: ablation
```

Findings deferred by `--judge-top-n` are flagged `judge_deferred` and keep
the proposer's confidence. That is deliberately distinct from `unjudged`,
which means every judge call failed and marks the whole run `[INVALID]`.
Collapsing the two would let an outage pass for a budget decision — the
failure mode this harness exists to prevent.

Use `run_server.py`, not `src/main.py`, when verifying behaviour. `main.py`
runs with `debug=True`, and the werkzeug reloader has left orphaned processes
bound to port 5000 on this machine -- requests then get answered by a stale
server running old code, which is extremely confusing to debug.

## The matching rule, and a correction to it

`match_ground_truth` decides where the true vulnerability sits in a report.
Its original implementation contradicted its own docstring: the text promised
"the rank of the **first** finding that describes the true vulnerability",
while the code took the candidate with the highest token overlap against the
reference text.

For a ranking metric the docstring was right. An auditor reading a report
top-down finds the issue at the first entry that describes it, whatever later
entries happen to be phrased more like the reference. The old rule
effectively **charged a report for describing the same issue twice** — it
scored the second copy and ignored the first — which penalises exactly the
high-recall behaviour this architecture produces.

The fix takes the earliest candidate. It is applied identically to both
sides, and **both sides gained**:

| on the same 72 contracts | Top-1 | Top-5 | Top-max | MRR |
|---|---|---|---|---|
| baseline, old rule | 0.726 | 0.973 | 0.973 | 0.829 |
| baseline, corrected | 0.792 | 0.972 | 0.972 | 0.870 |
| multi-agent, old rule | 0.329 | 0.808 | 0.973 | 0.523 |
| multi-agent, corrected | 0.639 | 0.917 | **0.986** | 0.756 |

That both improved is what makes the correction reportable: had only the new
architecture gained, the change would be indistinguishable from fitting the
metric to the result.

Note what the corrected numbers say. **Top-max is the one column the
multi-agent architecture wins** — it locates the labelled vulnerability more
often than the baseline. It still loses Top-1, and the reason is structural:
it reports a median of 14 findings per contract against the baseline's few,
on a benchmark that labels **one** vulnerability per contract. Other genuine
High-severity findings ranked above the labelled one are scored as failures.
That is a property of the evaluation protocol, not of the audit, and it is
worth stating plainly in any write-up.

### A second correction, larger than the first

The table above is computed on 72 contracts because that was all the
comparison could see. 42 of 143 baseline rows were being discarded as
unparseable, and that filter was **not** neutral: see `parse_findings`. The
cells are well-formed Python reprs that merely contain U+2028 LINE SEPARATOR
between elements, which `ast.literal_eval` rejects as non-printable source.

The baseline scores Top-1 0.822 on the 101 cells that always parsed and 0.268
on the 41 that did not, so discarding them deleted almost all of its failures:

| baseline, re-scored | n | Top-1 | Top-5 | Top-max | MRR |
|---|---|---|---|---|---|
| cells that always parsed | 101 | 0.822 | 0.970 | 0.970 | 0.885 |
| cells recovered by the fix | 41 | 0.268 | 0.634 | 0.659 | 0.415 |
| **all of them** | **142** | **0.662** | 0.873 | 0.880 | 0.749 |

Recovering them also **validates the matching rule**: 0.662 against the 0.664
the paper publishes from its own labels is agreement to 0.002, on a rule
written here without reference to those labels. The rule is stricter further
down the list — the paper reports Top-max 1.000 where this rule gives 0.880 —
but it is applied identically to both sides.

With the full dataset in scope, the standing comparison is:

| on the same 134 contracts | Top-1 | Top-5 | Top-max | MRR |
|---|---|---|---|---|
| baseline (paper, GPT) | 0.672 | **0.873** | 0.873 | **0.755** |
| multi-agent, loop off | **0.679** | 0.813 | **0.933** | 0.743 |
| multi-agent, loop on | 0.664 | 0.813 | **0.933** | 0.733 |

The architecture locates the labelled vulnerability in 125 of 134 contracts
against the baseline's 117 — 11 it finds that the baseline misses entirely,
against 3 the other way — and is level on Top-1. It loses Top-5 and MRR
because its extra detections land outside the top five. The baseline's Top-5
equals its Top-max exactly: when it finds the issue at all, it is always
already in the top five, because it reports so few findings.

Ranking weights were deliberately **not** tuned. Nine variants were measured
offline and all land between Top-1 0.257 and 0.310; picking the best of them
would be fitting the test set, and no such choice is in the shipped code.

## Measurement caveat

The published `Rank` column **cannot be reproduced** from the artifact:
matching the ground-truth `Issue` against the stored `Findings` reproduces
only ~61% of published ranks, and the disagreements are systematic. One of
143 rows still has an unreadable `Findings` cell (a stray U+2018); the other
41 that used to be unreadable are recovered — see `parse_findings`, and the
section above for why that mattered more than anything else measured here.

So the new numbers must not be compared against the published numbers
directly. `benchmark/run.py` instead re-scores the stored baseline findings
and the new findings with one identical, documented rule (`benchmark/scoring.py`).
That comparison is internally valid; the published one would not be.

Separately, four published result files (`claude_3_7_sonnet`,
`llama4_maverick`, `mistral_large_big`, `smartauditflow_gemini`) are
byte-identical in every metric column, as are `gemini20_flash` and
`gemini25_pro`. They appear to be copies rather than independent runs. Do not
cite them as separate baselines.
