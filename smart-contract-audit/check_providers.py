"""Check what the current `.env` actually gives you.

Answers the three questions that decide whether a run is worth starting:
what is configured, can a judge be independent of the proposer, and does each
key really work.

    .venv/Scripts/python.exe check_providers.py          # config only, free
    .venv/Scripts/python.exe check_providers.py --probe  # one tiny call each
    .venv/Scripts/python.exe check_providers.py --list-models
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.agents.llm import LLMClient, registry
from src.agents.providers import independence_report

GREEN, RED, YELLOW, DIM, RESET = (
    "\033[32m", "\033[31m", "\033[33m", "\033[2m", "\033[0m"
)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--probe", action="store_true",
                    help="send one minimal request per provider to prove the "
                         "key works (costs a few tokens)")
    ap.add_argument("--list-models", action="store_true",
                    help="ask each provider which models the key can serve; "
                         "names change often, so this beats guessing")
    args = ap.parse_args()

    providers = registry(refresh=True)
    if not providers:
        print(f"{RED}No provider is configured.{RESET}\n")
        print("Add at least two keys from different families to "
              "smart-contract-audit/.env. All of these have a free tier:\n")
        print("  GEMINI_API_KEY=...        aistudio.google.com/apikey")
        print("  GROQ_API_KEY=...          console.groq.com/keys")
        print("  MISTRAL_API_KEY=...       console.mistral.ai")
        print("  OPENROUTER_API_KEY=...    openrouter.ai  (+ OPENROUTER_MODEL)")
        print("  CF_ACCOUNT_ID=... CF_API_TOKEN=...   dash.cloudflare.com")
        print("  OLLAMA_BASE_URL=http://localhost:11434/v1/   (local)")
        return 1

    print(f"\n{len(providers)} provider(s) configured\n")
    width = max(len(n) for n in providers)
    for name, provider in sorted(providers.items()):
        limits = []
        if provider.rpm:
            limits.append(f"{provider.rpm}/min")
        if provider.daily:
            limits.append(f"{provider.daily}/day")
        print(
            f"  {name:<{width}}  {DIM}family={provider.family:<12}"
            f"model={provider.model:<28}{RESET}"
            + (f"{YELLOW}{' · '.join(limits)}{RESET}" if limits else
               f"{DIM}no limit{RESET}")
        )
        if provider.note:
            print(f"  {' ' * width}  {DIM}{provider.note}{RESET}")

    report = independence_report(providers)
    print(f"\nfamilies: {', '.join(sorted(report['families']))}")

    if report["can_judge_independently"]:
        print(f"{GREEN}Independent judging: available{RESET} "
              f"(panel of up to {report['max_panel_size']})")
    else:
        print(f"{RED}Independent judging: NOT available.{RESET}")
        print("  Every configured provider is in the same family, so a judge "
              "would be grading its own lineage.")
        print("  Benchmark runs refuse to start in this state; add a provider "
              "from another family.")

    if args.list_models:
        print("\nmodels each key can serve:")
        for name, provider in sorted(providers.items()):
            if provider.native:
                print(f"  {name}: {DIM}(native SDK; no listing endpoint){RESET}")
                continue
            try:
                client = LLMClient(providers={name: provider}, use_cache=False)
                sdk = client._sdk_client(provider)  # noqa: SLF001
                ids = sorted(m.id for m in sdk.models.list())
                shown = ", ".join(ids[:12])
                print(f"  {name}: {len(ids)} models — {shown}"
                      + (" ..." if len(ids) > 12 else ""))
                if provider.model not in ids:
                    print(f"    {YELLOW}warning: configured model "
                          f"{provider.model!r} is not in this list{RESET}")
            except Exception as exc:  # noqa: BLE001
                print(f"  {name}: {RED}listing failed — {str(exc)[:90]}{RESET}")

    if args.probe:
        print("\nprobing each provider with one minimal request:")
        failures = 0
        for name, provider in sorted(providers.items()):
            client = LLMClient(providers={name: provider}, use_cache=False,
                               strict=True, max_retries=1)
            try:
                response = client.complete(
                    "Responda apenas: ok", model=name,
                    max_tokens=16, timeout=45,
                )
                served = response.meta.get("served_model", provider.model)
                print(f"  {GREEN}ok{RESET}   {name:<{width}}  "
                      f"{DIM}{response.total_tokens} tokens, "
                      f"{response.duration:.1f}s, served={served}{RESET}")
            except Exception as exc:  # noqa: BLE001
                failures += 1
                print(f"  {RED}FAIL{RESET} {name:<{width}}  {str(exc)[:110]}")
        if failures:
            print(f"\n{RED}{failures} provider(s) failed.{RESET} A key that "
                  "lists models but fails here is usually out of quota.")
            return 1

    print()
    return 0 if report["can_judge_independently"] else 1


if __name__ == "__main__":
    sys.exit(main())
