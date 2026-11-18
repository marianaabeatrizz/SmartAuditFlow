"""The provider layer: discovery, rate limiting, retries and caching.

The free tiers the course guide documents are tight — OpenRouter 20 req/min
and 50/day, Groq 30/min — and this architecture fans out a dozen specialists
at once. Everything here exists so that a real run survives those limits
instead of collapsing into 429s, and so a provider is configuration rather
than a code change.
"""
from __future__ import annotations

import os
import sys
import tempfile
import threading
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.agents import providers as prov
from src.agents.llm import (
    INDEPENDENCE_CROSS_MODEL,
    INDEPENDENCE_CROSS_PROVIDER,
    LLMClient,
    LLMResponse,
    ModelUnavailable,
    RateLimiter,
    ResponseCache,
    independence_of,
    make_provider,
    role_model,
    set_registry,
)


def _clean_env(**overrides):
    """Environment with every provider variable stripped, plus overrides."""
    keep = {
        k: v for k, v in os.environ.items()
        if not any(
            k.startswith(p) or k.endswith(s)
            for p in ("SAF_PROVIDER_", "GEMINI_", "GROQ_", "OPENROUTER_",
                      "MISTRAL_", "DEEPSEEK_", "OPENAI_", "ANTHROPIC_",
                      "CF_", "OLLAMA_")
            for s in ("_API_KEY",)
        )
    }
    for name in ("GEMINI_API_KEY", "GROQ_API_KEY", "OPENROUTER_API_KEY",
                 "MISTRAL_API_KEY", "DEEPSEEK_API_KEY", "OPENAI_API_KEY",
                 "ANTHROPIC_API_KEY", "CF_ACCOUNT_ID", "CF_API_TOKEN",
                 "OLLAMA_BASE_URL", "OLLAMA_MODEL"):
        keep.pop(name, None)
    keep["SAF_SKIP_DOTENV"] = "1"
    keep.update(overrides)
    return keep


class _EnvPatch:
    def __init__(self, env):
        self.env = env

    def __enter__(self):
        self._saved = dict(os.environ)
        os.environ.clear()
        os.environ.update(self.env)

    def __exit__(self, *exc):
        os.environ.clear()
        os.environ.update(self._saved)


# ------------------------------------------------------------- discovery ---
def test_known_provider_is_discovered_from_its_key():
    with _EnvPatch(_clean_env(GROQ_API_KEY="gsk_test")):
        found = prov.discover()
    assert "groq" in found, sorted(found)
    assert found["groq"].family == "groq"
    assert found["groq"].base_url.startswith("https://api.groq.com")
    assert found["groq"].rpm == 30, "the guide's documented free-tier limit"


def test_placeholder_key_is_not_treated_as_configured():
    """The old code accepted 'your-openai-key' and only failed at the HTTP call."""
    for placeholder in ("your-openai-key", "sua_chave_gemini", "<KEY>", ""):
        with _EnvPatch(_clean_env(OPENAI_API_KEY=placeholder)):
            assert "openai" not in prov.discover(), placeholder


def test_model_can_be_overridden_because_names_change():
    with _EnvPatch(_clean_env(GEMINI_API_KEY="k", GEMINI_MODEL="gemini-9-ultra")):
        found = prov.discover()
    assert found["gemini"].model == "gemini-9-ultra"


def test_provider_without_a_model_is_skipped():
    """OpenRouter's :free catalogue changes, so there is no safe default."""
    with _EnvPatch(_clean_env(OPENROUTER_API_KEY="k")):
        assert "openrouter" not in prov.discover()
    with _EnvPatch(_clean_env(OPENROUTER_API_KEY="k",
                              OPENROUTER_MODEL="meta/llama:free")):
        assert "openrouter" in prov.discover()


def test_explicit_provider_declaration():
    env = _clean_env(
        SAF_PROVIDER_myjudge_BASE_URL="https://api.groq.com/openai/v1",
        SAF_PROVIDER_myjudge_API_KEY="k",
        SAF_PROVIDER_myjudge_MODEL="openai/gpt-oss-120b",
        SAF_PROVIDER_myjudge_FAMILY="groq",
        SAF_PROVIDER_myjudge_RPM="30",
    )
    with _EnvPatch(env):
        found = prov.discover()
    assert "myjudge" in found
    assert found["myjudge"].family == "groq"
    assert found["myjudge"].rpm == 30


def test_cloudflare_splices_the_account_id_into_the_url():
    with _EnvPatch(_clean_env(CF_ACCOUNT_ID="abc123", CF_API_TOKEN="t")):
        found = prov.discover()
    assert "cloudflare" in found
    assert "abc123" in found["cloudflare"].base_url


def test_independence_report_tells_you_if_judging_is_possible():
    one = {"groq": make_provider("groq", "groq")}
    assert prov.independence_report(one)["can_judge_independently"] is False

    two = {**one, "gemini": make_provider("gemini", "google")}
    report = prov.independence_report(two)
    assert report["can_judge_independently"] is True
    assert report["max_panel_size"] == 2


# ------------------------------------------------------- family / routing ---
def test_independence_uses_the_registry_families():
    set_registry({
        "a": make_provider("a", "groq"),
        "b": make_provider("b", "groq"),
        "c": make_provider("c", "google"),
    })
    try:
        assert independence_of("a", "b") == INDEPENDENCE_CROSS_MODEL
        assert independence_of("a", "c") == INDEPENDENCE_CROSS_PROVIDER
    finally:
        set_registry(None)


def test_role_model_falls_back_to_what_is_configured():
    set_registry({"mistral": make_provider("mistral", "mistral")})
    try:
        with _EnvPatch(_clean_env()):
            assert role_model("specialist") == "mistral"
    finally:
        set_registry(None)


# ----------------------------------------------------------- rate limiter ---
def test_rate_limiter_enforces_requests_per_minute():
    limiter = RateLimiter()
    provider = make_provider("slow", "x", rpm=3)
    started = time.time()
    for _ in range(3):
        limiter.acquire(provider, timeout=1)
    assert time.time() - started < 0.5, "the first burst must not be delayed"

    # The fourth must wait for the window, so a short timeout has to fail.
    try:
        limiter.acquire(provider, timeout=0.2)
    except RuntimeError as exc:
        assert "rate-limit slot" in str(exc)
        print("  4a requisicao bloqueada como esperado")
        return
    raise AssertionError("the limiter let a 4th request through an rpm=3 window")


def test_daily_quota_raises_rather_than_hanging():
    limiter = RateLimiter()
    provider = make_provider("capped", "x", rpm=0, daily=2)
    limiter.acquire(provider)
    limiter.acquire(provider)
    try:
        limiter.acquire(provider)
    except RuntimeError as exc:
        assert "daily quota" in str(exc)
        return
    raise AssertionError("the daily cap was not enforced")


def test_limiter_is_thread_safe():
    """A dozen specialists hit this concurrently."""
    limiter = RateLimiter()
    provider = make_provider("p", "x", rpm=20)
    errors = []

    def worker():
        try:
            limiter.acquire(provider, timeout=5)
        except Exception as exc:  # noqa: BLE001
            errors.append(exc)

    threads = [threading.Thread(target=worker) for _ in range(20)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not errors, errors


# ------------------------------------------------------------------ cache ---
def test_cache_avoids_a_second_call():
    with tempfile.TemporaryDirectory() as tmp:
        cache = ResponseCache(tmp)
        provider = make_provider("p", "x", model="m")
        key = ResponseCache.key(provider, "sys", "prompt", 0.0, 100)
        assert cache.get(key) is None
        cache.put(key, LLMResponse(text="hello", model="p",
                                   prompt_tokens=3, completion_tokens=4))
        hit = cache.get(key)
        assert hit is not None and hit.text == "hello"
        assert hit.cached is True
        assert hit.total_tokens == 7


def test_cache_key_changes_with_the_request():
    provider = make_provider("p", "x", model="m")
    base = ResponseCache.key(provider, "sys", "prompt", 0.0, 100)
    assert base != ResponseCache.key(provider, "sys", "OUTRO", 0.0, 100)
    assert base != ResponseCache.key(provider, "OUTRO", "prompt", 0.0, 100)
    assert base != ResponseCache.key(provider, "sys", "prompt", 0.7, 100)
    other = make_provider("p", "x", model="m2")
    assert base != ResponseCache.key(other, "sys", "prompt", 0.0, 100)


# ------------------------------------------------------------ retry/strict ---
class _Flaky:
    """Fails with a 429 a set number of times, then succeeds."""

    def __init__(self, failures):
        self.failures = failures
        self.calls = 0

        class _Completions:
            def __init__(self, outer):
                self.outer = outer

            def create(self, **kw):
                self.outer.calls += 1
                if self.outer.calls <= self.outer.failures:
                    raise RuntimeError("429 Too Many Requests: rate limit reached")

                class _Msg:
                    content = "ok"

                class _Choice:
                    message = _Msg()

                class _Usage:
                    prompt_tokens = 1
                    completion_tokens = 1

                class _Resp:
                    choices = [_Choice()]
                    usage = _Usage()
                    model = "m"

                return _Resp()

        class _Chat:
            def __init__(self, outer):
                self.completions = _Completions(outer)

        self.chat = _Chat(self)


def _client_with(flaky, **kw):
    provider = make_provider("p", "x", model="m")
    client = LLMClient(providers={"p": provider}, use_cache=False, **kw)
    client._clients["p"] = flaky  # noqa: SLF001 - injecting the SDK stub
    return client


def test_retries_a_429_then_succeeds():
    flaky = _Flaky(failures=2)
    client = _client_with(flaky, max_retries=5)
    started = time.time()
    response = client.complete("p", model="p", max_tokens=10)
    assert response.text == "ok"
    assert flaky.calls == 3, flaky.calls
    assert client.retries == 2
    # Backoff must actually back off, not spin.
    assert time.time() - started > 0.5
    print(f"  {flaky.calls} tentativas, {client.retries} retries")


def test_gives_up_after_max_retries():
    flaky = _Flaky(failures=99)
    client = _client_with(flaky, max_retries=2)
    try:
        client.complete("p", model="p", max_tokens=10)
    except Exception as exc:  # noqa: BLE001
        assert "429" in str(exc) or "rate limit" in str(exc).lower()
        assert flaky.calls == 3, f"expected 1 try + 2 retries, got {flaky.calls}"
        return
    raise AssertionError("it never gave up")


def test_token_budget_blocks_before_the_request_budget_does():
    """Regression: a 5-contract run stayed inside Groq's 30 req/min and still
    burned its 200k tokens/day. Counting requests measured the wrong thing."""
    limiter = RateLimiter()
    provider = make_provider("groq", "groq", rpm=30, tpm=8000, tpd=200000)

    # Four 2000-token calls fill the minute budget while using 4 of 30 requests.
    for _ in range(4):
        limiter.acquire(provider, timeout=1, est_tokens=2000)
    assert limiter.spent(provider)["minute"] == 8000

    try:
        limiter.acquire(provider, timeout=0.2, est_tokens=2000)
    except RuntimeError as exc:
        assert "rate-limit slot" in str(exc)
        return
    raise AssertionError("the limiter allowed a 5th call past an 8000 tpm cap")


def test_daily_token_quota_names_what_is_left():
    limiter = RateLimiter()
    provider = make_provider("groq", "groq", tpd=10000)
    limiter.acquire(provider, est_tokens=9000)
    try:
        limiter.acquire(provider, est_tokens=2000)
    except RuntimeError as exc:
        assert "daily token quota" in str(exc)
        assert "9,000" in str(exc), f"the message hides the usage: {exc}"
        return
    raise AssertionError("the daily token cap was not enforced")


def test_settle_replaces_the_estimate_with_real_usage():
    """The estimate is crude; the budget must track what was really charged."""
    limiter = RateLimiter()
    provider = make_provider("p", "x", tpm=100000)
    limiter.acquire(provider, est_tokens=5000)
    limiter.settle(provider, 5000, 1200)
    assert limiter.spent(provider)["minute"] == 1200

    # A failed call is not charged at all.
    limiter.acquire(provider, est_tokens=4000)
    limiter.settle(provider, 4000, 0)
    assert limiter.spent(provider)["minute"] == 1200


def test_a_dropped_connection_is_retried():
    """Regression: 'Connection error' was treated as permanent.

    Under fan-out the free tiers drop sockets, and each drop silently
    discarded a finding -- which reads as the model finding nothing, not as
    an error. Nothing reached the provider, so a retry costs no quota.
    """
    from src.agents.llm import _is_retryable

    for message in ("Connection error.",
                    "APIConnectionError: Connection error.",
                    "Connection reset by peer",
                    "Server disconnected without sending a response"):
        assert _is_retryable(Exception(message)), message


GEMINI_DAILY_429 = (
    "Error code: 429 - Quota exceeded for metric: "
    "generativelanguage.googleapis.com/generate_content_free_tier_requests, "
    "limit: 500, model: gemini-3.1-flash-lite, description: Generate content "
    "requests per day limit exceeded on Free Tier. "
    "Please retry in 12h37m58.050381965s. RESOURCE_EXHAUSTED"
)


def test_an_exhausted_daily_quota_is_not_retried():
    """Regression: the first full run spent most of 48 agent-hours backing off
    from 1,869 of these. The quota resets in 12h -- retrying cannot help."""
    from src.agents.llm import _is_retryable

    assert not _is_retryable(Exception(GEMINI_DAILY_429))
    assert not _is_retryable(Exception(
        "Rate limit reached ... on tokens per day (TPD): Limit 200000, "
        "Used 198112. Please try again in 32m33.936s"
    ))


def test_the_limiter_learns_the_real_rate_from_rejections():
    """Published limits have been wrong twice here: Gemini's daily cap was
    missing entirely, and NVIDIA's advertised 40/min produced 1,680
    rejections. The limiter must measure rather than trust."""
    limiter = RateLimiter()
    provider = make_provider("p", "x", rpm=40)

    limiter.penalize(provider)
    assert limiter._rate(provider) == 20.0  # noqa: SLF001
    limiter.penalize(provider)
    assert limiter._rate(provider) == 10.0  # noqa: SLF001

    # It must not collapse to zero and deadlock the run.
    for _ in range(20):
        limiter.penalize(provider)
    assert limiter._rate(provider) >= 2.0  # noqa: SLF001

    # And the throttle must be enforced, not merely recorded.
    fast = RateLimiter()
    quick = make_provider("q", "x", rpm=100)
    fast.penalize(quick)          # 100 -> 50
    for _ in range(50):
        fast.acquire(quick, timeout=1)
    try:
        fast.acquire(quick, timeout=0.2)
    except RuntimeError:
        return
    raise AssertionError("the reduced rate was recorded but not applied")


def test_sustained_success_walks_the_rate_back_up():
    """A one-off rejection must not throttle the rest of the run forever."""
    limiter = RateLimiter()
    provider = make_provider("p", "x", rpm=40)
    limiter.penalize(provider)
    assert limiter._rate(provider) == 20.0  # noqa: SLF001

    for _ in range(25):
        limiter.reward(provider)
    assert limiter._rate(provider) > 20.0  # noqa: SLF001

    # ...but never past what the provider was configured to allow.
    for _ in range(500):
        limiter.reward(provider)
    assert limiter._rate(provider) <= 40.0  # noqa: SLF001


def test_a_rate_rejection_is_told_apart_from_a_hard_stop():
    """Only 'slow down' should throttle; 'you are out of money' should not."""
    from src.agents.llm import _is_rate_rejection

    assert _is_rate_rejection(Exception("429 Too Many Requests"))
    assert not _is_rate_rejection(Exception(GEMINI_DAILY_429))
    assert not _is_rate_rejection(Exception(
        "429 - Your organization has reached its configured enforced spend "
        "limit. 'type': 'insufficient_quota'"
    ))


def test_a_billing_cap_is_not_retried():
    """A spend limit arrives as HTTP 429 but never clears on its own --
    someone has to raise the cap. Retrying it just burns wall-clock."""
    from src.agents.llm import _is_retryable

    assert not _is_retryable(Exception(
        "Error code: 429 - Your organization has reached its configured "
        "enforced spend limit. 'type': 'insufficient_quota', "
        "'code': 'organization_spend_limit_exceeded'"
    ))
    assert not _is_retryable(Exception(
        "Error code: 400 - Your credit balance is too low to access the API"
    ))


def test_a_per_minute_429_is_still_retried():
    """The other half: a burst limit clears in seconds and must be waited out."""
    from src.agents.llm import _is_retryable

    assert _is_retryable(Exception(
        "Error code: 429 - Rate limit reached for model `openai/gpt-oss-120b` "
        "on tokens per minute (TPM): Limit 8000, Used 5756. "
        "Please try again in 26.67s"
    ))
    assert _is_retryable(Exception("429 Too Many Requests: rate limit reached"))


def test_a_bad_key_is_not_retried():
    """The other half: retrying an auth or billing failure is pure waste."""
    from src.agents.llm import _is_retryable

    for message in ("invalid x-api-key",
                    "Your credit balance is too low to access the API",
                    "model not found"):
        assert not _is_retryable(Exception(message)), message


def test_strict_refuses_an_unconfigured_provider():
    client = LLMClient(providers={"p": make_provider("p", "x")},
                       strict=True, use_cache=False)
    try:
        client.complete("hi", model="nao-existe")
    except ModelUnavailable as exc:
        assert "not configured" in str(exc)
        return
    raise AssertionError("strict mode substituted a provider")


def test_report_states_whether_independent_judging_is_possible():
    one = LLMClient(providers={"groq": make_provider("groq", "groq")},
                    use_cache=False)
    assert one.report()["can_judge_independently"] is False

    two = LLMClient(providers={
        "groq": make_provider("groq", "groq"),
        "gemini": make_provider("gemini", "google"),
    }, use_cache=False)
    report = two.report()
    assert report["can_judge_independently"] is True
    assert report["max_panel_size"] == 2


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
