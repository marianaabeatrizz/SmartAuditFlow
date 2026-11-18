"""Provider registry: any OpenAI-compatible endpoint, declared in `.env`.

Every provider the course guide lists -- Gemini, Groq, OpenRouter, Mistral,
Cloudflare Workers AI, Ollama, DeepSeek, OpenAI -- speaks the OpenAI wire
format. Only ``base_url``, the key and the model name change. The previous
design hardcoded four providers and their model ids in Python, which meant a
new provider needed a code change and a renamed model silently broke a run.

Here a provider is data. Two ways to declare one:

**Known providers** are auto-detected from the key names the guide uses::

    GEMINI_API_KEY=...            # + optional GEMINI_MODEL
    GROQ_API_KEY=...
    OPENROUTER_API_KEY=...
    MISTRAL_API_KEY=...
    DEEPSEEK_API_KEY=...
    OPENAI_API_KEY=...
    ANTHROPIC_API_KEY=...
    CF_ACCOUNT_ID=... + CF_API_TOKEN=...
    OLLAMA_BASE_URL=http://localhost:11434/v1/

**Anything else** is declared explicitly, which also lets one provider appear
twice under different models::

    SAF_PROVIDER_myjudge_BASE_URL=https://api.groq.com/openai/v1
    SAF_PROVIDER_myjudge_API_KEY=gsk_...
    SAF_PROVIDER_myjudge_MODEL=openai/gpt-oss-120b
    SAF_PROVIDER_myjudge_FAMILY=groq
    SAF_PROVIDER_myjudge_RPM=30

``FAMILY`` is what judge independence is computed from, so two entries pointing
at the same vendor must share a family even when their model ids differ.

``RPM`` matters more than it looks: the free tiers are tight (Groq 30/min,
OpenRouter 20/min and 50/day) and this architecture fans out a dozen
specialists at once. Without a limiter the first real run is a wall of 429s.
"""
from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

# Defaults for the providers the course guide documents. `model` is only a
# starting point: the guide warns that names change often, so every entry can
# be overridden with <NAME>_MODEL in .env, and `saf-models` lists what a key
# actually serves.
KNOWN_PROVIDERS: Dict[str, Dict[str, object]] = {
    "gemini": {
        "key_env": "GEMINI_API_KEY",
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai/",
        "model": "gemini-3.8-flash",
        "family": "google",
        "rpm": 15,
        # The free tier allows 500 generate_content calls per project per day.
        # Leaving this at 0 let a 143-contract run spend hours retrying 1,869
        # calls against an exhausted quota. Set GEMINI_DAILY=0 once billing is
        # enabled on the project.
        "daily": 500,
        "note": "free tier: 500 requests/day is the binding limit for a judge "
                "role; set GEMINI_DAILY=0 after enabling billing. The guide "
                "warns prompts may be used by Google outside UK/CH/EEA -- "
                "fine for public benchmark contracts",
    },
    "groq": {
        "key_env": "GROQ_API_KEY",
        "base_url": "https://api.groq.com/openai/v1",
        "model": "openai/gpt-oss-120b",
        "family": "groq",
        "rpm": 30,
        # Measured against the live API on 2026-11-14, not guessed.
        "tpm": 8000,
        "tpd": 200000,
        "note": "free; 8k tokens/min and 200k/day are the binding limits, "
                "not the request count; no logprobs, n must be 1",
    },
    "openrouter": {
        "key_env": "OPENROUTER_API_KEY",
        "base_url": "https://openrouter.ai/api/v1",
        "model": "",  # must be set: the :free catalogue changes
        "family": "openrouter",
        "rpm": 20,
        "note": "set OPENROUTER_MODEL to an exact id; :free models come and go",
    },
    "mistral": {
        "key_env": "MISTRAL_API_KEY",
        "base_url": "https://api.mistral.ai/v1",
        "model": "mistral-small-latest",
        "family": "mistral",
        "rpm": 30,
    },
    "deepseek": {
        "key_env": "DEEPSEEK_API_KEY",
        "base_url": "https://api.deepseek.com",
        "model": "deepseek-chat",
        "family": "deepseek",
        "rpm": 60,
    },
    "openai": {
        "key_env": "OPENAI_API_KEY",
        "base_url": "",  # SDK default
        "model": "gpt-4o-mini",
        "family": "openai",
        "rpm": 60,
    },
    "nvidia": {
        "key_env": "NVIDIA_API_KEY",
        "base_url": "https://integrate.api.nvidia.com/v1",
        # The catalogue rotates and retires models with HTTP 410, so override
        # with NVIDIA_MODEL when this one goes. Deliberately a Nemotron:
        # NVIDIA also serves Gemma and Mistral, and judging with those would
        # make `family` a lie -- the provider would be nvidia while the
        # model's lineage was Google's or Mistral's.
        "model": "nvidia/nemotron-3-ultra-550b-a55b",
        "family": "nvidia",
        # A third-party list said 40/min; sustained load produced 1,680
        # rejections. Measured here: 20 concurrent and ~77/min in short
        # bursts both pass, so the real constraint is sustained volume, not
        # instantaneous rate. Start conservative and let the limiter's AIMD
        # find the ceiling instead of guessing a third number.
        "rpm": 15,
        "note": "free prototyping trial; a genuinely different family, so it "
                "gives cross-provider judging. Published limits are "
                "inconsistent -- measure with --probe before planning a run",
    },
    # A second entry on the same OpenAI key, running a different model. It is
    # deliberately the SAME family, so independence_of() reports cross-model,
    # not cross-provider: a different checkpoint from the same provider shares
    # training data and RLHF lineage, so family-level self-preference survives.
    # Weaker than a genuine second provider, stronger than literal
    # self-judgement -- and it is what a free-tier-only setup can afford.
    # Set OPENAI_JUDGE_MODEL to change it; it must differ from OPENAI_MODEL.
    "openai_judge": {
        "key_env": "OPENAI_API_KEY",
        "base_url": "",
        "model": "gpt-4.1-mini",
        "family": "openai",
        "rpm": 60,
        "note": "same key as 'openai', different model: CROSS-MODEL only, "
                "not cross-provider. Label results accordingly.",
    },
    "anthropic": {
        "key_env": "ANTHROPIC_API_KEY",
        "base_url": "",  # native SDK, not OpenAI-compatible
        "model": "claude-3-5-sonnet-20241022",
        "family": "anthropic",
        "rpm": 50,
        "native": "anthropic",
    },
    "ollama": {
        "key_env": "",  # no key; presence of OLLAMA_BASE_URL enables it
        "base_url": "http://localhost:11434/v1/",
        "model": "qwen3:8b",
        "family": "ollama",
        "rpm": 0,  # local, unlimited
        "note": "fully reproducible and free; needs the daemon running",
    },
}

_EXPLICIT = re.compile(r"^SAF_PROVIDER_([A-Za-z0-9]+)_BASE_URL$")


@dataclass
class Provider:
    """One callable endpoint."""

    name: str
    base_url: str
    api_key: str
    model: str
    family: str
    rpm: int = 0            # requests per minute; 0 = unlimited
    daily: int = 0          # requests per day; 0 = unlimited
    # Tokens, not requests, are what the free tiers actually meter: a 5-contract
    # run stayed well inside Groq's 30 req/min and still exhausted its 200k
    # tokens/day. Counting requests alone measures the wrong quantity.
    tpm: int = 0            # tokens per minute; 0 = unlimited
    tpd: int = 0            # tokens per day; 0 = unlimited
    native: str = ""        # "anthropic" for the non-OpenAI-shaped SDK
    note: str = ""

    @property
    def openai_compatible(self) -> bool:
        return not self.native

    def describe(self) -> Dict[str, object]:
        return {
            "name": self.name,
            "family": self.family,
            "model": self.model,
            "base_url": self.base_url or "(SDK default)",
            "rpm": self.rpm or "unlimited",
            "daily": self.daily or "unlimited",
            "tpm": self.tpm or "unlimited",
            "tpd": self.tpd or "unlimited",
            "note": self.note,
        }


def _env(name: str, default: str = "") -> str:
    return (os.getenv(name) or default).strip()


def _looks_unset(value: str) -> bool:
    """Placeholder keys are worse than missing ones: the old code treated
    'your-openai-key' as configured and only failed at the HTTP call."""
    if not value:
        return True
    low = value.lower()
    return low.startswith(("your-", "your_", "sua_", "seu_", "<", "changeme", "xxx"))


def discover(env_file: Optional[str] = None) -> Dict[str, Provider]:
    """Build the registry from the environment.

    Loads `.env` first so a plain script behaves like the Flask app.
    """
    if env_file or not os.getenv("SAF_SKIP_DOTENV"):
        try:
            from dotenv import load_dotenv

            load_dotenv(env_file) if env_file else load_dotenv()
        except ImportError:  # pragma: no cover
            pass

    providers: Dict[str, Provider] = {}

    for name, spec in KNOWN_PROVIDERS.items():
        upper = name.upper()
        key_env = str(spec.get("key_env") or "")
        api_key = _env(key_env) if key_env else ""

        if name == "ollama":
            base = _env("OLLAMA_BASE_URL", str(spec["base_url"]))
            # Only enable when explicitly pointed at, so a stray default does
            # not advertise a daemon that is not running.
            if not _env("OLLAMA_BASE_URL") and not _env("OLLAMA_MODEL"):
                continue
            api_key = "ollama"
        elif name == "cloudflare":  # handled below
            continue
        else:
            if _looks_unset(api_key):
                continue
            base = _env(f"{upper}_BASE_URL", str(spec["base_url"]))

        model = _env(f"{upper}_MODEL", str(spec["model"]))
        if not model:
            logger.warning(
                "provider %r has a key but no model; set %s_MODEL "
                "(its catalogue changes, so there is no safe default)",
                name, upper,
            )
            continue

        providers[name] = Provider(
            name=name,
            base_url=base,
            api_key=api_key,
            model=model,
            family=_env(f"{upper}_FAMILY", str(spec["family"])),
            rpm=int(_env(f"{upper}_RPM", str(spec.get("rpm", 0))) or 0),
            daily=int(_env(f"{upper}_DAILY", str(spec.get("daily", 0))) or 0),
            tpm=int(_env(f"{upper}_TPM", str(spec.get("tpm", 0))) or 0),
            tpd=int(_env(f"{upper}_TPD", str(spec.get("tpd", 0))) or 0),
            native=str(spec.get("native", "")),
            note=str(spec.get("note", "")),
        )

    # Cloudflare needs the account id spliced into the URL.
    account = _env("CF_ACCOUNT_ID")
    token = _env("CF_API_TOKEN")
    if not _looks_unset(account) and not _looks_unset(token):
        providers["cloudflare"] = Provider(
            name="cloudflare",
            base_url=f"https://api.cloudflare.com/client/v4/accounts/{account}/ai/v1",
            api_key=token,
            model=_env("CF_MODEL", "@cf/openai/gpt-oss-20b"),
            family=_env("CF_FAMILY", "cloudflare"),
            rpm=int(_env("CF_RPM", "0") or 0),
            note="model ids are fixed and versioned -- good for replication",
        )

    # Explicit declarations, which may also shadow a known provider.
    for var in list(os.environ):
        match = _EXPLICIT.match(var)
        if not match:
            continue
        name = match.group(1).lower()
        base = _env(var)
        key = _env(f"SAF_PROVIDER_{match.group(1)}_API_KEY")
        model = _env(f"SAF_PROVIDER_{match.group(1)}_MODEL")
        if not base or not model:
            logger.warning(
                "explicit provider %r is incomplete; needs BASE_URL and MODEL", name
            )
            continue
        providers[name] = Provider(
            name=name,
            base_url=base,
            api_key=key or "none",
            model=model,
            family=_env(f"SAF_PROVIDER_{match.group(1)}_FAMILY", name),
            rpm=int(_env(f"SAF_PROVIDER_{match.group(1)}_RPM", "0") or 0),
            daily=int(_env(f"SAF_PROVIDER_{match.group(1)}_DAILY", "0") or 0),
            tpm=int(_env(f"SAF_PROVIDER_{match.group(1)}_TPM", "0") or 0),
            tpd=int(_env(f"SAF_PROVIDER_{match.group(1)}_TPD", "0") or 0),
        )

    return providers


def families(providers: Dict[str, Provider]) -> Dict[str, List[str]]:
    """Provider names grouped by family -- the unit judge independence uses."""
    out: Dict[str, List[str]] = {}
    for provider in providers.values():
        out.setdefault(provider.family, []).append(provider.name)
    return out


def independence_report(providers: Dict[str, Provider]) -> Dict[str, object]:
    """Can this configuration support an independent judge, and a panel?"""
    by_family = families(providers)
    return {
        "providers": sorted(providers),
        "families": {k: sorted(v) for k, v in sorted(by_family.items())},
        "family_count": len(by_family),
        "can_judge_independently": len(by_family) >= 2,
        "max_panel_size": len(by_family),
    }
