"""Agent base class and response-parsing helpers."""
from __future__ import annotations

import json
import logging
import re
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from src.agents.llm import LLMClient, LLMResponse, role_model

logger = logging.getLogger(__name__)

_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)


def extract_json(text: str, expect: str = "array") -> Optional[Any]:
    """Pull the first well-formed JSON value out of a model response.

    The legacy implementation did ``text[text.find('['):text.rfind(']')+1]``,
    which silently produced garbage whenever the model wrote prose containing a
    bracket, or wrapped the payload in a markdown fence. We try, in order:
    the whole string, any fenced block, then a bracket-matched scan that
    respects strings and escapes.
    """
    if not text:
        return None

    candidates: List[str] = [text.strip()]
    candidates.extend(m.strip() for m in _FENCE.findall(text))

    opener, closer = ("[", "]") if expect == "array" else ("{", "}")
    scanned = _scan_balanced(text, opener, closer)
    if scanned:
        candidates.append(scanned)

    for candidate in candidates:
        if not candidate:
            continue
        try:
            value = json.loads(candidate)
        except (json.JSONDecodeError, ValueError):
            continue
        if expect == "array" and isinstance(value, list):
            return value
        if expect == "object" and isinstance(value, dict):
            return value
        # A model asked for an array sometimes returns {"findings": [...]}.
        if expect == "array" and isinstance(value, dict):
            for key in ("findings", "results", "items", "data", "issues"):
                if isinstance(value.get(key), list):
                    return value[key]
    return None


def _scan_balanced(text: str, opener: str, closer: str) -> Optional[str]:
    """Return the first balanced ``opener``/``closer`` span, ignoring punctuation
    that appears inside JSON string literals."""
    start = text.find(opener)
    if start == -1:
        return None
    depth = 0
    in_string = False
    escaped = False
    for i in range(start, len(text)):
        ch = text[i]
        if escaped:
            escaped = False
            continue
        if ch == "\\":
            escaped = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch == opener:
            depth += 1
        elif ch == closer:
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    return None


class Agent(ABC):
    """One role in the audit. Owns its own prompt, model and failure handling."""

    role: str = "specialist"
    name: str = "agent"

    def __init__(self, client: LLMClient, model: Optional[str] = None):
        self.client = client
        self.model = role_model(self.role, model)

    @property
    def system_prompt(self) -> str:
        return "You are a meticulous smart contract security auditor."

    def ask(
        self,
        prompt: str,
        timeout: int = 90,
        max_tokens: int = 4000,
        temperature: float = 0.0,
    ) -> LLMResponse:
        return self.client.complete(
            prompt,
            model=self.model,
            system=self.system_prompt,
            timeout=timeout,
            max_tokens=max_tokens,
            temperature=temperature,
        )

    @abstractmethod
    def run(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Execute this agent and return a partial state update."""

    def log_entry(self, **fields: Any) -> Dict[str, Any]:
        entry = {"agent": self.name, "role": self.role, "model": self.model}
        entry.update(fields)
        return entry
