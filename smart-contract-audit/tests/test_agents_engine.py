"""Integration: MultiAgentEngine against the real in-memory storage.

Checks the seams the graph tests cannot see -- progress reporting, the storage
round trip, token accounting, and that the ranking metadata survives
``storage._validate_findings_format``, which used to rebuild findings from the
five legacy fields and drop everything else.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.agents.engine import MultiAgentEngine
from src.storage.memory import storage
from tests.test_agents_graph import CONTRACT, FakeLLM


def _engine(session_id, llm, **kw):
    engine = MultiAgentEngine(session_id, model_choice="auto", **kw)
    # Swap in the scripted client: rebuild the graph around it so no network
    # call is made anywhere in the engine path.
    from src.agents.graph import build_graph

    engine.client = llm
    engine.graph = build_graph(
        client=llm, specialties=kw.get("specialties"), strict=False
    )
    return engine


SPECIALTIES = ["reentrancy", "unchecked_low_level_calls", "arithmetic", "gas_efficiency"]


def test_engine_completes_and_persists():
    session_id = storage.create_session(CONTRACT, "", "auto")
    llm = FakeLLM()
    engine = _engine(session_id, llm, specialties=SPECIALTIES)

    progress = []
    report = engine.execute_workflow_sync(
        CONTRACT, "", progress_callback=lambda **kw: progress.append(kw)
    )

    assert report["FindingNumber"] > 0
    assert storage.get_session(session_id)["status"] == "completed"

    saved = storage.get_result(session_id)
    assert saved is not None, "result was not persisted"
    assert len(saved["findings"]) == report["FindingNumber"]

    # Progress must advance monotonically and finish at 100.
    pcts = [p["percentage"] for p in progress]
    assert pcts == sorted(pcts), f"progress went backwards: {pcts}"
    assert pcts[-1] == 100, f"never reached 100: {pcts}"
    print(f"  phases: {[p['name'] for p in progress]}")


def test_ranking_metadata_survives_storage():
    """storage._validate_findings_format must not strip the new fields."""
    session_id = storage.create_session(CONTRACT, "", "auto")
    engine = _engine(session_id, FakeLLM(), specialties=SPECIALTIES)
    engine.execute_workflow_sync(CONTRACT, "")

    saved = storage.get_result(session_id)["findings"]
    first = saved[0]
    for field in ("Issue", "Severity", "Description", "Impact", "Location"):
        assert field in first, f"legacy field {field} lost"
    for field in ("Rank", "Confidence", "Verdict", "Agent"):
        assert field in first, f"multi-agent field {field} was stripped by storage"

    ranks = [f["Rank"] for f in saved]
    assert ranks == sorted(ranks), f"stored order does not match rank: {ranks}"
    print(f"  stored with metadata intact; ranks {ranks}")


def test_token_usage_is_summed_not_overwritten():
    """Accumulators must be reduced across nodes, not replaced by the last one."""
    session_id = storage.create_session(CONTRACT, "", "auto")
    llm = FakeLLM()
    engine = _engine(session_id, llm, specialties=SPECIALTIES)
    report = engine.execute_workflow_sync(CONTRACT, "")

    counted = report["ExecutionSummary"]["total_tokens"]
    expected = len(llm.calls) * 15  # FakeLLM bills 10 prompt + 5 completion
    assert counted == expected, (
        f"token accounting wrong: {counted} != {expected} "
        f"({len(llm.calls)} calls). Reducer likely overwritten."
    )
    print(f"  {counted} tokens over {len(llm.calls)} calls")


def test_failure_is_recorded_not_swallowed():
    class Broken(FakeLLM):
        def complete(self, *a, **kw):
            raise RuntimeError("provider down")

    session_id = storage.create_session(CONTRACT, "", "auto")
    engine = _engine(session_id, Broken(), specialties=["reentrancy"])

    # The context agent catches its own failure, so the run completes with an
    # empty brief rather than raising; what matters is that the session is not
    # silently reported as a clean success with real findings.
    report = engine.execute_workflow_sync(CONTRACT, "")
    assert report["FindingNumber"] == 0, "findings invented despite total LLM failure"
    assert report["ExecutionSummary"]["errors"], "errors were swallowed"
    print(f"  {len(report['ExecutionSummary']['errors'])} error(s) surfaced")


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
