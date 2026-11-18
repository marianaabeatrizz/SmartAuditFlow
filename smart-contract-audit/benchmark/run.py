"""Benchmark runner for the multi-agent architecture.

The dataset, the ground truth and the baseline's own findings all live in the
same published CSV, so both architectures are scored on identical inputs with
the identical matching rule in ``scoring.py``.

Usage
-----
Re-score the published baselines only (no API calls, no cost):

    .venv/Scripts/python.exe -m benchmark.run --rescore-only

Run the multi-agent architecture over N contracts and compare:

    .venv/Scripts/python.exe -m benchmark.run --limit 20 --model openai

Flags:
    --rescore-only   score the stored baseline findings and stop
    --limit N        audit only the first N contracts (default: all)
    --model NAME     pin every agent to one model, for a like-for-like
                     comparison against a single-model baseline
    --workers N      contracts audited concurrently (default 3)
    --max-rounds N   replanning budget (default 1 for benchmarking)
    --panel-size N   independent judges per finding (default 1)
    --allow-self-validation
                     permit a model to judge its own findings. Off by default:
                     an LLM grading its own output favours it, which inflates
                     the confidence score the ranking depends on, so a run that
                     needs this flag is not measuring what it claims to.
    --out PATH       where to write the per-contract CSV
"""
from __future__ import annotations

import argparse
import ast
import csv
import json
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from benchmark.scoring import Match, format_table, match_ground_truth, metrics

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RESULTS_DIR = os.path.join(REPO, "evaluation", "results", "standardVuln")

csv.field_size_limit(min(sys.maxsize, 2**31 - 1))


# ---------------------------------------------------------------- dataset ---
def load_dataset(path: Optional[str] = None) -> List[Dict[str, Any]]:
    """Ground truth plus contract source, taken from the published CSV.

    Using the CSV rather than the contracts directory guarantees the new runs
    see byte-identical input to what the baseline saw.
    """
    path = path or os.path.join(RESULTS_DIR, "smartauditflow_gpt_curated_datasets.csv")
    rows: List[Dict[str, Any]] = []
    with open(path, encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            name = (row.get("File Name") or "").strip()
            code = (row.get("code_snippet") or row.get("Source Code") or "").strip()
            vuln = (row.get("Vulnerability Type") or "").strip()
            if not name or not code or not vuln:
                continue
            rows.append({
                "file": name,
                "vuln_type": vuln,
                "issue": (row.get("Issue") or "").strip(),
                "code": code,
                "baseline_findings_raw": row.get("Findings") or "",
            })
    return rows


def parse_findings(raw: str) -> Optional[List[Dict]]:
    """Parse the baseline's stored findings cell.

    It is a Python repr, not JSON, and 42/145 cells are malformed. An
    unparseable cell is returned as None and excluded from the comparison
    rather than silently counted as 'found nothing', which would flatter the
    new architecture.
    """
    raw = (raw or "").strip()
    if not raw:
        return None
    for loader in (ast.literal_eval, json.loads):
        try:
            obj = loader(raw)
        except Exception:  # noqa: BLE001
            continue
        if isinstance(obj, dict):
            obj = obj.get("Findings") or obj.get("findings")
        if isinstance(obj, list):
            return [f for f in obj if isinstance(f, dict)]
    return None


# -------------------------------------------------------------- baselines ---
def rescore_baselines(dataset: List[Dict[str, Any]]) -> Dict[str, Dict[str, float]]:
    """Apply our matching rule to every published result file we can parse."""
    truth = {r["file"]: r for r in dataset}
    out: Dict[str, Dict[str, float]] = {}

    for fname in sorted(os.listdir(RESULTS_DIR)):
        if not fname.endswith(".csv"):
            continue
        label = fname.replace("_curated_datasets.csv", "")
        ranks: List[int] = []
        unparsed = 0
        with open(os.path.join(RESULTS_DIR, fname), encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                name = (row.get("File Name") or "").strip()
                if name not in truth:
                    continue
                findings = parse_findings(row.get("Findings") or "")
                if findings is None:
                    unparsed += 1
                    continue
                gt = truth[name]
                ranks.append(
                    match_ground_truth(findings, gt["vuln_type"], gt["issue"]).rank
                )
        if ranks:
            m = metrics(ranks)
            m["unparsed"] = unparsed
            out[f"{label} (re-scored)"] = m
    return out


# ------------------------------------------------------------- new engine ---
def audit_one(
    item: Dict[str, Any],
    model: Optional[str],
    max_rounds: int,
    panel_size: int = 1,
    allow_self_validation: bool = False,
    max_refine_rounds: int = 1,
    judge_top_n: int = 0,
    judge_models=None,
) -> Dict[str, Any]:
    from src.agents.graph import build_graph
    from src.agents.llm import LLMClient
    from src.agents.state import new_state

    started = time.time()
    # strict=True: a silent fallback to a different provider would invalidate
    # the measurement this whole script exists to produce.
    client = LLMClient(strict=True)
    # strict_independence: a published number must not come from a model
    # grading its own output. Override only with --allow-self-validation, and
    # the caveat is then carried in the output column.
    graph = build_graph(
        client=client,
        strict=True,
        model=model,
        panel_size=panel_size,
        strict_independence=not allow_self_validation,
        judge_top_n=judge_top_n,
        judge_models=judge_models,
    )

    try:
        final = graph.invoke(new_state(
            item["code"], session_id=item["file"],
            max_rounds=max_rounds, max_refine_rounds=max_refine_rounds,
        ))
    except Exception as exc:  # noqa: BLE001
        return {
            "file": item["file"], "vuln_type": item["vuln_type"], "rank": 0,
            "error": str(exc)[:300], "findings": 0,
            "seconds": round(time.time() - started, 1), "tokens": 0,
            "independence": "", "self_validated": 0, "unjudged": 0,
            "refined": 0, "withdrawn": 0, "refine_rounds": 0,
        }
    finally:
        # audit_one builds a client per contract, so its counters die with it.
        # Folding them into one process-wide total is what makes a cost figure
        # possible at the end of the run.
        _record_usage(locals().get("client"))

    findings = final.get("findings", []) or []
    match: Match = match_ground_truth(findings, item["vuln_type"], item["issue"])
    return {
        "file": item["file"],
        "vuln_type": item["vuln_type"],
        "rank": match.rank,
        "match_score": match.score,
        "matched": match.matched_text,
        "findings": len(findings),
        "confirmed": len([f for f in findings if f.get("verdict") == "confirmed"]),
        "rejected": len([f for f in findings if f.get("verdict") == "rejected"]),
        "merged": final.get("DuplicatesMerged", 0),
        "rounds": final.get("round", 0),
        "tokens": final.get("token_usage", 0),
        "seconds": round(time.time() - started, 1),
        "independence": final.get("IndependenceLevel", ""),
        "self_validated": final.get("SelfValidatedCount", 0),
        "unjudged": final.get("UnjudgedCount", 0),
        "refined": final.get("RefinedCount", 0),
        "withdrawn": final.get("WithdrawnCount", 0),
        "refine_rounds": final.get("refine_round", 0),
        "error": "",
    }


def run_new(
    dataset: List[Dict[str, Any]],
    model: Optional[str],
    workers: int,
    max_rounds: int,
    panel_size: int = 1,
    allow_self_validation: bool = False,
    max_refine_rounds: int = 1,
    judge_top_n: int = 0,
    judge_models=None,
) -> List[Dict[str, Any]]:
    results: List[Dict[str, Any]] = []
    done = 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(
                audit_one, item, model, max_rounds, panel_size,
                allow_self_validation, max_refine_rounds, judge_top_n,
                judge_models,
            ): item
            for item in dataset
        }
        for future in as_completed(futures):
            row = future.result()
            results.append(row)
            done += 1
            flag = "!" if row.get("error") else ("." if row["rank"] == 1 else "o")
            print(
                f"  [{done}/{len(dataset)}] {flag} {row['file'][:40]:<42}"
                f"rank={row['rank']:<3} findings={row['findings']:<3} "
                f"{row['seconds']}s",
                flush=True,
            )
    return results


# -------------------------------------------------------------------- cli ---

# USD per million tokens, checked against the providers' pricing pages on
# 2026-11-15. A provider absent here is treated as free, which is true of the
# tiers the course guide lists -- but it means the printed cost is a floor,
# not a guarantee. Override with SAF_PRICE_<provider>=<in>/<out>.
PRICES = {
    "openai": (0.15, 0.60),       # gpt-4o-mini
    "openai_judge": (0.40, 1.60),  # gpt-4.1-mini
    "anthropic": (3.00, 15.00),   # claude-3-5-sonnet
    # gemini-3.1-flash-lite, standard paid rate. Free below the daily cap,
    # but the judge needs ~2,400 calls for 143 contracts against a ~500/day
    # free allowance, so a full run bills.
    "gemini": (0.25, 1.50),
    "groq": (0.0, 0.0),           # free tier
    "mistral": (0.0, 0.0),
    "openrouter": (0.0, 0.0),     # :free catalogue
    "ollama": (0.0, 0.0),         # local
}


# Process-wide usage, summed across the per-contract clients.
_USAGE_LOCK = threading.Lock()
_USAGE: Dict[str, Dict[str, int]] = {}
_CACHE_HITS = [0]


def _record_usage(client) -> None:
    if client is None:
        return
    with _USAGE_LOCK:
        for name, u in (getattr(client, "usage", None) or {}).items():
            slot = _USAGE.setdefault(name, {"calls": 0, "input": 0, "output": 0})
            for k in ("calls", "input", "output"):
                slot[k] += u.get(k, 0)
        _CACHE_HITS[0] += getattr(client, "cache_hits", 0)


def _price_for(name):
    override = os.getenv(f"SAF_PRICE_{name.upper()}")
    if override and "/" in override:
        a, b = override.split("/", 1)
        return float(a), float(b)
    return PRICES.get(name, (0.0, 0.0))


def _print_usage(contracts):
    """Per-provider token usage and what it costs, measured not guessed."""
    with _USAGE_LOCK:
        usage = {k: dict(v) for k, v in _USAGE.items()}
        cache_hits = _CACHE_HITS[0]
    if not usage:
        return
    print("\n  provider      calls      input     output    USD")
    print("  " + "-" * 50)
    total = 0.0
    for name in sorted(usage):
        u = usage[name]
        pin, pout = _price_for(name)
        cost = u["input"] / 1e6 * pin + u["output"] / 1e6 * pout
        total += cost
        print(f"  {name:<12}{u['calls']:>7}{u['input']:>11,}{u['output']:>11,}"
              f"{cost:>8.3f}")
    print("  " + "-" * 50)
    print(f"  {'total':<12}{'':>7}{'':>11}{'':>11}{total:>8.3f}")
    # Contracts served from cache cost nothing, so dividing by the contract
    # count would understate the real per-contract price. Charged calls are
    # the honest denominator.
    charged = sum(u["calls"] for u in usage.values())
    billable = contracts
    if cache_hits and charged:
        billable = contracts * charged / (charged + cache_hits)
    if billable:
        per = total / billable
        print(f"\n  USD {per:.4f} per contract -> "
              f"USD {per * 143:.2f} for all 143, "
              f"USD {per * 143 * 2:.2f} for both ablation arms")
    if cache_hits:
        print(f"  ({cache_hits:,} of {charged + cache_hits:,} calls came from "
              f"cache and were not charged; the per-contract figure above is "
              f"scaled to {billable:.1f} fully-paid contracts)")

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--rescore-only", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--model", default=None)
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--max-rounds", type=int, default=1)
    ap.add_argument("--max-refine-rounds", type=int, default=1,
                    help="inner loop budget: a finding that fails verification "
                         "goes back to its specialist. 0 disables the loop, "
                         "which is the ablation arm.")
    ap.add_argument("--panel-size", type=int, default=1,
                    help="independent judges per finding (>1 convenes a panel "
                         "drawn from different providers)")
    ap.add_argument("--judge", action="append", default=None, metavar="PROVIDER",
                    help="pin the judge pool to these providers (repeatable). "
                         "Without it, judges are chosen automatically and a "
                         "different provider family is preferred. Use this to "
                         "force a specific judge, e.g. --judge openai_judge "
                         "when no cross-provider quota is left -- the result "
                         "is then cross-model, not cross-provider, and the "
                         "output says so.")
    ap.add_argument("--judge-top-n", type=int, default=0,
                    help="judge only the N most confident findings per "
                         "contract (0 = judge all). Top-1/Top-5/MRR depend "
                         "only on the head of the ranking, so a verdict on "
                         "the 18th finding cannot move a metric; this roughly "
                         "halves judge calls and fits a free-tier quota. "
                         "Deferred findings are flagged in the output.")
    ap.add_argument("--allow-self-validation", action="store_true",
                    help="permit a model to judge its own findings when no "
                         "independent judge is configured. Biases confidence "
                         "upward; the result is flagged.")
    ap.add_argument("--out", default="benchmark_results.csv")
    args = ap.parse_args()

    dataset = load_dataset()
    print(f"dataset: {len(dataset)} contracts with ground truth\n")

    table: Dict[str, Dict[str, float]] = {}

    print("re-scoring published baselines with our matching rule...")
    table.update(rescore_baselines(dataset))
    print(format_table(table))

    unparsed = {k: int(v.get("unparsed", 0)) for k, v in table.items()}
    if any(unparsed.values()):
        print(
            "\nnote: rows whose stored Findings cell could not be parsed were "
            "excluded, not counted as misses:"
        )
        for k, v in sorted(unparsed.items(), key=lambda kv: -kv[1]):
            if v:
                print(f"  {k}: {v} excluded")

    if args.rescore_only:
        return 0

    subset = dataset[: args.limit] if args.limit else dataset
    print(
        f"\nauditing {len(subset)} contracts with the multi-agent architecture "
        f"(model={args.model or 'per-role defaults'}, workers={args.workers}, "
        f"max_rounds={args.max_rounds})\n"
    )
    results = run_new(
        subset, args.model, args.workers, args.max_rounds,
        args.panel_size, args.allow_self_validation, args.max_refine_rounds,
        args.judge_top_n, args.judge,
    )

    ok = [r for r in results if not r.get("error")]
    failed = [r for r in results if r.get("error")]
    if ok:
        panel = f", panel={args.panel_size}" if args.panel_size > 1 else ""
        loop = "" if args.max_refine_rounds else ", sem refinamento"
        # How independent the judging actually was belongs in the label, not
        # in a footnote: a cross-model number read as cross-provider would
        # overstate the central claim of this work.
        seen_levels = {r.get("independence", "") for r in ok} - {""}
        weakest = ""
        for level in ("unjudged", "self", "cross-model"):
            if level in seen_levels:
                weakest = f", {level}"
                break
        label = f"multi-agent ({args.model or 'per-role'}{panel}{loop}{weakest})"
        table[label] = metrics([r["rank"] for r in ok])
        biased = sum(int(r.get("self_validated", 0) or 0) for r in ok)
        if biased:
            print(
                f"\nWARNING: {biased} finding(s) were judged by their own "
                "proposer. Confidence on those is inflated, which distorts "
                "ranking. Configure a second provider before reporting these."
            )
        blind = sum(int(r.get("unjudged", 0) or 0) for r in ok)
        contracts_blind = sum(1 for r in ok if int(r.get("unjudged", 0) or 0))
        if blind:
            # Loud and unmissable: the first full run of this benchmark
            # produced a complete-looking table while two thirds of its judge
            # calls had failed on quota. The numbers looked like an
            # architecture result and were an outage.
            table[f"{label} [INVALID]"] = table.pop(label, {})
            print(
                "\n" + "!" * 72 +
                f"\nINVALID RUN: {blind} finding(s) across {contracts_blind} "
                f"contract(s) were never judged -- every judge call failed.\n"
                "Those findings keep their proposer's own confidence, so the\n"
                "ranking metrics below measure an outage, not the architecture.\n"
                "Fix the judge provider's quota and re-run before reporting.\n"
                + "!" * 72
            )

    if results:
        with open(args.out, "w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(results[0].keys()))
            writer.writeheader()
            writer.writerows(results)
        print(f"\nper-contract results written to {args.out}")

    print()
    print(format_table(table))
    if failed:
        print(f"\n{len(failed)} contract(s) errored:")
        for r in failed[:5]:
            print(f"  {r['file']}: {r['error'][:110]}")
    if ok:
        print(
            f"\ncost: {sum(r['tokens'] for r in ok):,} tokens, "
            f"{sum(r['seconds'] for r in ok):.0f}s of agent time over {len(ok)} contracts"
        )
        _print_usage(len(ok))
    return 0


if __name__ == "__main__":
    sys.exit(main())
