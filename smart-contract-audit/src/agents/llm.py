"""Calling models: provider routing, rate limiting, retries and caching.

What this layer is for
----------------------
1. **Any OpenAI-compatible endpoint.** Providers are declared in `.env` and
   discovered at runtime (see ``providers.py``), not hardcoded. A "model" here
   is a *provider entry name* -- ``gemini``, ``groq``, ``myjudge`` -- which
   carries its own base_url, key, model id and family.

2. **Judge independence.** Classified by provider *family*, not model id:
   ``gpt-4o`` judging ``gpt-4o-mini`` shares training lineage and alignment, so
   the self-preference effect does not disappear just because the checkpoint
   differs.

3. **No shared mutable state.** The legacy engine mutated a process-wide
   singleton to pick a model, so two overlapping sessions clobbered each
   other. Nothing here mutates: the provider is an argument on every call.

4. **Experimental validity.** ``strict=True`` refuses to silently answer with
   a different provider. A run labelled ``groq`` that was in fact served by
   Gemini is not a measurement.

5. **Free tiers survive contact with the architecture.** A dozen specialists
   fan out at once against limits as low as 20 requests/minute. A token-bucket
   limiter per provider, exponential backoff on 429, and an optional on-disk
   cache keyed by prompt hash keep that workable -- the cache in particular
   stops debugging from burning a daily quota.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import random
import re
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from src.agents.providers import Provider, discover, families

logger = logging.getLogger(__name__)

# How independent a judgement actually was. Recorded on every finding so a
# result can be audited rather than assumed clean.
INDEPENDENCE_CROSS_PROVIDER = "cross-provider"  # different family: strongest
INDEPENDENCE_CROSS_MODEL = "cross-model"        # same family, other entry
INDEPENDENCE_SELF = "self"                      # same entry: biased
# No judge answered at all -- every call failed. Strictly worse than "self":
# the finding carries only its proposer's own confidence, unchecked.
INDEPENDENCE_UNJUDGED = "unjudged"

# Preferred provider per role when nothing is configured. Planning and judging
# deserve the strongest available; the bulk specialist work does not. Entries
# that are not configured are skipped, so this degrades rather than breaks.
ROLE_PREFERENCES: Dict[str, List[str]] = {
    "orchestrator": ["openai", "gemini", "groq", "mistral", "deepseek", "ollama"],
    "specialist": ["groq", "gemini", "cloudflare", "mistral", "deepseek", "openai"],
    "validator": ["gemini", "anthropic", "mistral", "groq", "openai", "deepseek"],
    "aggregator": ["openai", "gemini", "groq", "mistral", "deepseek"],
}

_registry_lock = threading.Lock()
_registry: Optional[Dict[str, Provider]] = None


def registry(refresh: bool = False) -> Dict[str, Provider]:
    """The discovered providers, cached per process."""
    global _registry
    with _registry_lock:
        if _registry is None or refresh:
            _registry = discover()
            if _registry:
                logger.info(
                    "providers configured: %s",
                    ", ".join(
                        f"{p.name}({p.family}/{p.model})" for p in _registry.values()
                    ),
                )
            else:
                logger.warning("no providers configured; set a key in .env")
        return _registry


def set_registry(providers: Optional[Dict[str, Provider]]) -> None:
    """Install an explicit provider set, bypassing `.env` discovery.

    Used by tests and by any caller that wants to pin exactly which providers
    exist. ``None`` restores discovery on the next access.
    """
    global _registry
    with _registry_lock:
        _registry = providers


def make_provider(
    name: str, family: str, model: str = "", rpm: int = 0, **kw
) -> Provider:
    """Convenience constructor for an in-memory provider entry."""
    return Provider(
        name=name,
        base_url=kw.pop("base_url", ""),
        api_key=kw.pop("api_key", "test"),
        model=model or name,
        family=family,
        rpm=rpm,
        **kw,
    )


def family(name: str) -> str:
    provider = registry().get(name)
    return provider.family if provider else name


def independence_of(proposer: str, judge: str) -> str:
    """Classify how independent ``judge`` is from ``proposer``."""
    if proposer == judge:
        return INDEPENDENCE_SELF
    if family(proposer) == family(judge):
        return INDEPENDENCE_CROSS_MODEL
    return INDEPENDENCE_CROSS_PROVIDER


class NoIndependentJudge(RuntimeError):
    """No judge independent of the proposer is configured."""


class ModelUnavailable(RuntimeError):
    """A specifically requested provider is not configured."""


def select_judges(
    proposer: str,
    candidates: List[str],
    panel_size: int = 1,
    strict: bool = False,
    seed: str = "",
) -> List[str]:
    """Pick judges for a finding proposed by ``proposer``, most independent first.

    ``seed`` spreads the work across equally independent judges. With one
    judge per finding and three free providers configured, always choosing
    the first one drives that provider into its rate limit while the other
    two sit idle -- which is what reduced a run to 20 minutes per contract.

    The spread is keyed on the finding's identity rather than a rotating
    counter, so the same finding goes to the same judge on every run. A
    counter would make the assignment depend on thread scheduling, and a
    benchmark whose judge allocation changes between runs is not reproducible.

    Preference: cross-provider, then cross-model, then -- only if nothing else
    exists -- the proposer itself. Strict mode raises instead, because a
    benchmark number produced by a model grading its own output does not
    measure what it claims to.

    A panel is drawn from distinct families where possible, so three judges
    means three lineages rather than three entries of the same one.
    """
    cross_provider = [
        m for m in candidates
        if independence_of(proposer, m) == INDEPENDENCE_CROSS_PROVIDER
    ]
    cross_model = [
        m for m in candidates
        if independence_of(proposer, m) == INDEPENDENCE_CROSS_MODEL
    ]

    if seed and len(cross_provider) > 1:
        offset = int(hashlib.sha256(seed.encode()).hexdigest()[:8], 16)
        offset %= len(cross_provider)
        cross_provider = cross_provider[offset:] + cross_provider[:offset]

    chosen: List[str] = []
    seen_families: set = set()
    for model in cross_provider:
        if family(model) not in seen_families:
            chosen.append(model)
            seen_families.add(family(model))
        if len(chosen) >= panel_size:
            return chosen

    for pool in (cross_provider, cross_model):
        for model in pool:
            if model not in chosen:
                chosen.append(model)
            if len(chosen) >= panel_size:
                return chosen

    if chosen:
        return chosen

    if strict:
        raise NoIndependentJudge(
            f"no provider independent of '{proposer}' is configured "
            f"(available: {candidates or 'none'}). Add a second provider from "
            "a different family -- Gemini, Groq, Mistral, Cloudflare and "
            "Ollama all have free tiers -- or pass strict=False to accept "
            "self-validation, which will be flagged on every finding."
        )
    return [proposer] if proposer in candidates else list(candidates[:1])


def role_model(role: str, override: Optional[str] = None) -> str:
    """Resolve which provider a role uses.

    Precedence: explicit override > ``SAF_MODEL_<ROLE>`` > the first configured
    provider in ``ROLE_PREFERENCES`` > any configured provider.
    """
    if override:
        return override
    env = (os.getenv(f"SAF_MODEL_{role.upper()}") or "").strip()
    if env:
        return env

    available = registry()
    for candidate in ROLE_PREFERENCES.get(role, []):
        if candidate in available:
            return candidate
    return next(iter(available), "openai")


@dataclass
class LLMResponse:
    text: str
    model: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    duration: float = 0.0
    cached: bool = False
    meta: Dict[str, Any] = field(default_factory=dict)

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


class RateLimiter:
    """Token bucket per provider, plus a daily cap.

    The free tiers the course guide lists are tight (OpenRouter 20/min and
    50/day, Groq 30/min and 1000/day) and this architecture is deliberately
    parallel, so without this the first real run is a wall of 429s.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._minute: Dict[str, deque] = {}
        self._day: Dict[str, deque] = {}
        # (timestamp, tokens) pairs, so a window's cost can be summed.
        self._minute_tok: Dict[str, deque] = {}
        self._day_tok: Dict[str, deque] = {}
        # Measured rate, which starts at the configured one and moves with
        # what the provider actually accepts.
        self._effective: Dict[str, float] = {}
        self._streak: Dict[str, int] = {}

    @staticmethod
    def _trim(window: deque, now: float, span: float) -> None:
        while window and now - window[0][0] > span:
            window.popleft()

    def acquire(self, provider: Provider, timeout: float = 300.0,
                est_tokens: int = 0) -> None:
        """Block until this request fits both the request and token budgets.

        ``est_tokens`` is the caller's estimate of what the request will cost.
        It is reserved up front and reconciled by :meth:`settle` once the real
        usage is known, because a provider rejects the request on the budget it
        sees, not on the one we hoped for.
        """
        if not any((provider.rpm, provider.daily, provider.tpm, provider.tpd)):
            return
        deadline = time.time() + timeout
        while True:
            with self._lock:
                now = time.time()
                minute = self._minute.setdefault(provider.name, deque())
                day = self._day.setdefault(provider.name, deque())
                min_tok = self._minute_tok.setdefault(provider.name, deque())
                day_tok = self._day_tok.setdefault(provider.name, deque())
                while minute and now - minute[0] > 60:
                    minute.popleft()
                while day and now - day[0] > 86400:
                    day.popleft()
                self._trim(min_tok, now, 60)
                self._trim(day_tok, now, 86400)

                if provider.daily and len(day) >= provider.daily:
                    raise RuntimeError(
                        f"{provider.name}: daily quota of {provider.daily} "
                        "requests is exhausted. Use another provider, or wait "
                        "for the reset."
                    )
                spent_today = sum(t for _, t in day_tok)
                if provider.tpd and spent_today + est_tokens > provider.tpd:
                    raise RuntimeError(
                        f"{provider.name}: daily token quota exhausted "
                        f"({spent_today:,}/{provider.tpd:,} used, this call "
                        f"needs ~{est_tokens:,}). Use another provider, or "
                        "wait for the reset."
                    )

                spent_minute = sum(t for _, t in min_tok)
                # The measured rate, not the advertised one.
                rpm = self._effective.get(provider.name, float(provider.rpm or 0))
                requests_ok = not rpm or len(minute) < rpm
                tokens_ok = (not provider.tpm
                             or spent_minute + est_tokens <= provider.tpm)
                if requests_ok and tokens_ok:
                    minute.append(now)
                    if provider.daily:
                        day.append(now)
                    if est_tokens and (provider.tpm or provider.tpd):
                        min_tok.append((now, est_tokens))
                        day_tok.append((now, est_tokens))
                    return

                waits = [0.05]
                if not requests_ok and minute:
                    waits.append(60 - (now - minute[0]))
                if not tokens_ok and min_tok:
                    waits.append(60 - (now - min_tok[0][0]))
                wait = max(waits)

            if time.time() + wait > deadline:
                raise RuntimeError(
                    f"{provider.name}: waited {timeout:.0f}s for a rate-limit slot"
                )
            logger.debug("%s: rate limit, waiting %.1fs", provider.name, wait)
            time.sleep(wait)

    # ------------------------------------------------------- adaptive rate ---
    # Published limits have been wrong twice here: Gemini's daily cap was not
    # in the registry at all, and NVIDIA's "40/min" came from a third-party
    # list and produced 1,680 rejections under sustained load. So the limiter
    # measures instead of trusting: every rejection halves the provider's
    # effective rate, and a run of successes walks it back up (AIMD, the same
    # principle TCP uses on an unknown link).
    def penalize(self, provider: Provider) -> float:
        """A rate rejection: back off hard and immediately."""
        with self._lock:
            current = self._effective.get(provider.name, float(provider.rpm or 60))
            reduced = max(2.0, current / 2.0)
            self._effective[provider.name] = reduced
            self._streak[provider.name] = 0
        if reduced < current:
            logger.warning("%s: rate rejected, throttling %.0f -> %.0f per minute",
                           provider.name, current, reduced)
        return reduced

    def reward(self, provider: Provider) -> None:
        """Sustained success: creep back up, never past what was configured."""
        ceiling = float(provider.rpm or 60)
        with self._lock:
            streak = self._streak.get(provider.name, 0) + 1
            self._streak[provider.name] = streak
            if streak < 25:
                return
            self._streak[provider.name] = 0
            current = self._effective.get(provider.name, ceiling)
            if current < ceiling:
                self._effective[provider.name] = min(ceiling, current * 1.25)

    def _rate(self, provider: Provider) -> float:
        with self._lock:
            return self._effective.get(provider.name, float(provider.rpm or 0))

    def settle(self, provider: Provider, est_tokens: int, actual: int) -> None:
        """Replace the reserved estimate with what the call really cost."""
        if not est_tokens or not (provider.tpm or provider.tpd):
            return
        delta = int(actual) - int(est_tokens)
        if not delta:
            return
        with self._lock:
            now = time.time()
            for store in (self._minute_tok, self._day_tok):
                window = store.get(provider.name)
                if window:
                    window.append((now, delta))

    def spent(self, provider: Provider) -> Dict[str, int]:
        """Tokens charged to this provider in the last minute and day."""
        with self._lock:
            now = time.time()
            minute = self._minute_tok.get(provider.name, deque())
            day = self._day_tok.get(provider.name, deque())
            self._trim(minute, now, 60)
            self._trim(day, now, 86400)
            return {"minute": sum(t for _, t in minute),
                    "day": sum(t for _, t in day)}


class ResponseCache:
    """On-disk cache keyed by the exact request.

    The guide recommends this explicitly: debugging a prompt should not burn a
    daily quota. Keyed on everything that changes the answer, so a changed
    prompt is a miss rather than a stale hit.
    """

    def __init__(self, path: Optional[str] = None) -> None:
        self.dir = path or os.getenv("SAF_CACHE_DIR") or os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(
                os.path.abspath(__file__)))), ".llm_cache"
        )
        self._lock = threading.Lock()
        os.makedirs(self.dir, exist_ok=True)

    @staticmethod
    def key(provider: Provider, system: str, prompt: str,
            temperature: float, max_tokens: int) -> str:
        blob = json.dumps(
            [provider.name, provider.model, system, prompt, temperature, max_tokens],
            ensure_ascii=False,
        )
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()

    def get(self, key: str) -> Optional[LLMResponse]:
        path = os.path.join(self.dir, f"{key}.json")
        try:
            with open(path, encoding="utf-8") as fh:
                data = json.load(fh)
        except (OSError, json.JSONDecodeError):
            return None
        return LLMResponse(
            text=data["text"], model=data["model"],
            prompt_tokens=data.get("prompt_tokens", 0),
            completion_tokens=data.get("completion_tokens", 0),
            duration=0.0, cached=True,
        )

    def put(self, key: str, response: LLMResponse) -> None:
        path = os.path.join(self.dir, f"{key}.json")
        payload = {
            "text": response.text, "model": response.model,
            "prompt_tokens": response.prompt_tokens,
            "completion_tokens": response.completion_tokens,
            "stored_at": time.time(),
        }
        try:
            with self._lock:
                tmp = f"{path}.tmp{os.getpid()}"
                with open(tmp, "w", encoding="utf-8") as fh:
                    json.dump(payload, fh, ensure_ascii=False)
                os.replace(tmp, path)
        except OSError as exc:  # pragma: no cover
            logger.debug("cache write failed: %s", exc)


# A dropped socket is the most retryable failure there is: nothing reached the
# provider, so retrying costs no quota and risks no double billing. Leaving
# these out meant a transient TCP failure under fan-out silently discarded a
# finding, which looks like the model finding nothing rather than an error.
_RETRYABLE = ("rate limit", "429", "overloaded", "503", "502", "timeout",
              "temporarily unavailable", "too many requests",
              "connection error", "connection reset", "connection aborted",
              "connection refused", "remote end closed", "server disconnected",
              "eof occurred", "broken pipe", "read operation timed out",
              "temporary failure in name resolution", "apiconnectionerror")


# A 429 usually means "slow down" and clears in seconds. These mean the daily
# allowance is gone and the next window is hours away, so backing off is not
# waiting -- it is burning wall-clock to fail anyway. The first full benchmark
# run spent most of 48 agent-hours retrying 1,869 of these.
# Money, not rate: a billing cap arrives as 429 but never clears on its own.
# Someone has to raise the limit or add funds, so every retry is wasted.
_BILLING = ("spend limit", "insufficient_quota", "credit balance",
            "billing details", "organization_spend_limit")

# A daily allowance. It does clear, but hours from now, so backing off inside
# a run is not waiting -- it is burning wall-clock to fail anyway.
_DAILY = ("free_tier_requests", "requests per day", "tokens per day",
          "resource_exhausted", "per day limit exceeded", "tpd:", "rpd:",
          "daily quota", "quota exceeded")


def _is_exhausted(exc: Exception) -> bool:
    """True when retrying cannot succeed within this run."""
    text = str(exc).lower()
    if any(token in text for token in _BILLING):
        return True
    if not any(token in text for token in _DAILY):
        return False
    # "retry in 12h37m" is the giveaway; a per-minute limit says seconds.
    match = re.search(r"retry in (?:(\d+)h)?(?:(\d+)m)?", text)
    if match and match.group(1):
        return True
    return any(token in text for token in
               ("per day", "free_tier_requests", "resource_exhausted"))


def _is_rate_rejection(exc: Exception) -> bool:
    """Told to slow down, as opposed to told to stop (billing or daily cap)."""
    if _is_exhausted(exc):
        return False
    if getattr(exc, "status_code", None) == 429:
        return True
    text = str(exc).lower()
    return "429" in text or "too many requests" in text or "rate limit" in text


def _is_retryable(exc: Exception) -> bool:
    if _is_exhausted(exc):
        return False
    status = getattr(exc, "status_code", None)
    if status in (429, 500, 502, 503, 504):
        return True
    return any(token in str(exc).lower() for token in _RETRYABLE)


class LLMClient:
    """Thread-safe facade over the configured providers."""

    _pool = None  # bounded httpx pool, shared across clients and threads

    def __init__(
        self,
        strict: bool = True,
        use_cache: Optional[bool] = None,
        max_retries: int = 5,
        providers: Optional[Dict[str, Provider]] = None,
    ):
        self._providers = providers if providers is not None else registry()
        self._strict = strict
        self._limiter = RateLimiter()
        self._clients: Dict[str, Any] = {}
        self._client_lock = threading.Lock()
        self.max_retries = max_retries
        if use_cache is None:
            use_cache = (os.getenv("SAF_CACHE", "1") or "1").lower() not in (
                "0", "false", "no"
            )
        self._cache = ResponseCache() if use_cache else None
        self._lock = threading.Lock()
        self.calls = 0
        self.tokens = 0
        self.cache_hits = 0
        self.retries = 0
        # Input and output are billed at very different rates, so a single
        # total cannot be turned into a cost. Tracked per provider because a
        # run mixes a paid proposer with free-tier judges.
        self.usage: Dict[str, Dict[str, int]] = {}

    # ------------------------------------------------------------------
    def available(self, model: str) -> bool:
        return model in self._providers

    def get_available_models(self) -> Dict[str, bool]:
        return {name: True for name in self._providers}

    def provider(self, name: str) -> Provider:
        try:
            return self._providers[name]
        except KeyError:
            raise ModelUnavailable(
                f"provider '{name}' is not configured. Configured: "
                f"{sorted(self._providers) or 'none'}. "
                "Add its key to .env."
            ) from None

    def report(self) -> Dict[str, Any]:
        by_family = families(self._providers)
        return {
            "providers": {n: p.describe() for n, p in self._providers.items()},
            "families": {k: sorted(v) for k, v in by_family.items()},
            "can_judge_independently": len(by_family) >= 2,
            "max_panel_size": len(by_family),
            "calls": self.calls, "tokens": self.tokens,
            "cache_hits": self.cache_hits, "retries": self.retries,
            "usage": {k: dict(v) for k, v in self.usage.items()},
        }

    # ------------------------------------------------------------------
    def complete(
        self,
        prompt: str,
        model: str,
        system: str = "",
        timeout: int = 90,
        max_tokens: int = 4000,
        temperature: float = 0.0,
    ) -> LLMResponse:
        """Call exactly the requested provider.

        Non-strict mode falls back to any other configured provider, which
        keeps an interactive session usable when one endpoint is down. Strict
        mode raises, because a silently substituted provider invalidates a
        measurement.
        """
        order = [model]
        if not self._strict:
            order += [m for m in self._providers if m != model]

        last_error: Optional[str] = None
        for name in order:
            if name not in self._providers:
                last_error = f"provider '{name}' is not configured"
                if self._strict:
                    raise ModelUnavailable(
                        f"{last_error}. Configured: {sorted(self._providers) or 'none'}."
                    )
                continue
            try:
                return self._call_with_retry(
                    self._providers[name], prompt, system,
                    timeout, max_tokens, temperature,
                )
            except Exception as exc:  # noqa: BLE001
                last_error = f"{name}: {exc}"
                logger.warning("LLM call failed (%s)", last_error)
                if self._strict:
                    raise
        raise RuntimeError(f"no provider could serve the request. Last: {last_error}")

    # ------------------------------------------------------------------
    def _call_with_retry(
        self, provider: Provider, prompt: str, system: str,
        timeout: int, max_tokens: int, temperature: float,
    ) -> LLMResponse:
        cache_key = None
        if self._cache:
            cache_key = ResponseCache.key(
                provider, system, prompt, temperature, max_tokens
            )
            hit = self._cache.get(cache_key)
            if hit is not None:
                with self._lock:
                    self.cache_hits += 1
                return hit

        # Roughly 4 characters per token, plus the completion we are asking
        # for. An estimate is enough: settle() corrects it from real usage.
        est = (len(prompt) + len(system)) // 4 + int(max_tokens)

        attempt = 0
        while True:
            self._limiter.acquire(provider, est_tokens=est)
            try:
                response = self._dispatch(
                    provider, prompt, system, timeout, max_tokens, temperature
                )
                self._limiter.settle(provider, est, response.total_tokens)
                self._limiter.reward(provider)
                with self._lock:
                    self.calls += 1
                    self.tokens += response.total_tokens
                    slot = self.usage.setdefault(
                        provider.name,
                        {"calls": 0, "input": 0, "output": 0},
                    )
                    slot["calls"] += 1
                    slot["input"] += response.prompt_tokens
                    slot["output"] += response.completion_tokens
                if self._cache and cache_key:
                    self._cache.put(cache_key, response)
                return response
            except Exception as exc:  # noqa: BLE001
                # A rejected call is not charged, so holding its reservation
                # would shrink the budget for no reason.
                self._limiter.settle(provider, est, 0)
                # Being told to slow down is the only reliable measurement of
                # a provider's real rate, so feed it back into the limiter.
                if _is_rate_rejection(exc):
                    self._limiter.penalize(provider)
                attempt += 1
                if attempt > self.max_retries or not _is_retryable(exc):
                    raise
                # Full jitter: a dozen specialists retrying in lockstep would
                # just rebuild the burst that triggered the limit.
                delay = min(60.0, 2 ** attempt) * (0.5 + random.random() / 2)
                with self._lock:
                    self.retries += 1
                logger.info(
                    "%s: retryable error (%s), attempt %s/%s in %.1fs",
                    provider.name, str(exc)[:90], attempt, self.max_retries, delay,
                )
                time.sleep(delay)

    # ------------------------------------------------------------------
    @staticmethod
    def _http_pool():
        """One bounded connection pool, shared by every OpenAI-compatible SDK.

        ``SAF_MAX_CONNECTIONS`` tunes it; the default is deliberately modest
        because the free tiers drop connections long before they return a
        clean 429.
        """
        try:
            import httpx
        except ImportError:  # the SDK's own default is then used
            return None
        if LLMClient._pool is None:
            limit = int(os.getenv("SAF_MAX_CONNECTIONS", "12") or 12)
            LLMClient._pool = httpx.Client(
                limits=httpx.Limits(
                    max_connections=limit,
                    max_keepalive_connections=max(2, limit // 2),
                ),
                timeout=httpx.Timeout(180.0, connect=15.0),
            )
        return LLMClient._pool

    # ------------------------------------------------------------------
    def _sdk_client(self, provider: Provider):
        with self._client_lock:
            if provider.name in self._clients:
                return self._clients[provider.name]
            if provider.native == "anthropic":
                import anthropic

                client = anthropic.Anthropic(api_key=provider.api_key)
            else:
                import openai

                kwargs: Dict[str, Any] = {"api_key": provider.api_key or "none"}
                if provider.base_url:
                    kwargs["base_url"] = provider.base_url
                # A dozen specialists times a panel of judges times the
                # contract workers opens far more sockets than a free tier
                # tolerates, and the drops come back as connection errors.
                # Bounding the pool queues the excess instead of failing it.
                pool = self._http_pool()
                if pool is not None:
                    kwargs["http_client"] = pool
                client = openai.OpenAI(**kwargs)
            self._clients[provider.name] = client
            return client

    def _dispatch(
        self, provider: Provider, prompt: str, system: str,
        timeout: int, max_tokens: int, temperature: float,
    ) -> LLMResponse:
        client = self._sdk_client(provider)
        started = time.time()

        if provider.native == "anthropic":
            msg = client.messages.create(
                model=provider.model,
                max_tokens=max_tokens,
                system=system or "You are a smart contract security auditor.",
                messages=[{"role": "user", "content": prompt}],
                timeout=timeout,
            )
            usage = getattr(msg, "usage", None)
            return LLMResponse(
                text=msg.content[0].text if msg.content else "",
                model=provider.name,
                prompt_tokens=getattr(usage, "input_tokens", 0) or 0,
                completion_tokens=getattr(usage, "output_tokens", 0) or 0,
                duration=time.time() - started,
            )

        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        kwargs: Dict[str, Any] = {
            "model": provider.model,
            "messages": messages,
            "timeout": timeout,
            "max_tokens": max_tokens,
        }
        # Reasoning models reject temperature; the guide flags this for both
        # the OpenAI o-series and some Gemini/Groq entries.
        if not _rejects_temperature(provider.model):
            kwargs["temperature"] = temperature

        try:
            completion = client.chat.completions.create(**kwargs)
        except Exception as exc:  # noqa: BLE001
            # Some endpoints reject max_tokens or temperature with a 400.
            # Retrying once without them is the difference between a provider
            # working and being unusable, and it is visible in the log.
            message = str(exc).lower()
            if "max_tokens" in message or "temperature" in message or "unsupported" in message:
                kwargs.pop("temperature", None)
                kwargs.pop("max_tokens", None)
                logger.info(
                    "%s rejected a sampling parameter; retrying without it",
                    provider.name,
                )
                completion = client.chat.completions.create(**kwargs)
            else:
                raise

        usage = getattr(completion, "usage", None)
        choice = completion.choices[0] if completion.choices else None
        return LLMResponse(
            text=(choice.message.content if choice and choice.message else "") or "",
            model=provider.name,
            prompt_tokens=getattr(usage, "prompt_tokens", 0) or 0,
            completion_tokens=getattr(usage, "completion_tokens", 0) or 0,
            duration=time.time() - started,
            meta={"served_model": getattr(completion, "model", provider.model)},
        )


def _rejects_temperature(model_id: str) -> bool:
    low = model_id.lower()
    return low.startswith(("o1", "o3", "o4")) or "/o1" in low or "/o3" in low
