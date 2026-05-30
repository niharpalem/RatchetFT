"""Agent brain — drives the `claude` CLI as a subprocess.

Same pattern as SAGE-X's llm/claude.py: no Anthropic API key needed, the local
`claude` CLI is already authenticated. We shell out with `-p` (one-shot, no REPL)
and `--output-format text`, then parse JSON out of the reply.

The whole AutoFinetune loop uses two calls from here:
  - ask_json(): force a structured decision (LoRA config, next-action, etc.)
  - ask_text(): free-form reasoning / explanation for the UI narrative.
"""
from __future__ import annotations

import json
import re
import subprocess
import uuid
from pathlib import Path

# Where claude runs (cwd). Repo root, so any files it writes land here.
REPO_DIR = Path(__file__).resolve().parent.parent.parent


class BrainError(RuntimeError):
    """The claude CLI failed or returned something unparseable."""


def _run(system: str, user: str, model: str = "sonnet",
         effort: str | None = None, timeout: int = 300) -> str:
    """One-shot claude CLI call. Returns stdout text or raises BrainError."""
    call_id = str(uuid.uuid4())
    cmd = [
        "claude", "-p",
        "--output-format", "text",
        "--model", model,
    ]
    if effort:
        cmd += ["--effort", effort]
    cmd += [
        "--dangerously-skip-permissions",
        "--session-id", call_id,
        "--system-prompt", system,
        user,
    ]
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True,
            timeout=timeout, cwd=str(REPO_DIR),
        )
    except FileNotFoundError as e:
        raise BrainError("`claude` CLI not found on PATH") from e
    except subprocess.TimeoutExpired as e:
        raise BrainError(f"claude call timed out after {timeout}s") from e

    out = (result.stdout or "").strip()
    if not out:
        raise BrainError(f"empty reply (stderr: {result.stderr.strip()[:300]})")
    return out


def ask_text(system: str, user: str, *, model: str = "sonnet",
             effort: str | None = None) -> str:
    """Free-form reasoning. Returns the assistant's text reply."""
    return _run(system, user, model=model, effort=effort)


def ask_json(system: str, user: str, *, model: str = "sonnet",
             effort: str | None = None) -> dict:
    """Force a structured decision. Returns a parsed dict or raises BrainError.

    Tolerant parsing: fenced ```json block first, then the widest {...} span.
    """
    # Nudge the model toward pure JSON.
    system = system + "\n\nRespond with ONLY a single valid JSON object. No prose, no markdown fences."
    reply = _run(system, user, model=model, effort=effort)
    return _extract_json(reply)


def _extract_json(text: str) -> dict:
    # 1) fenced code block
    m = re.search(r"```(?:json)?\s*\n(.*?)\n```", text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(1))
        except json.JSONDecodeError:
            pass
    # 2) widest {...} span
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end > start:
        try:
            return json.loads(text[start:end + 1])
        except json.JSONDecodeError as e:
            raise BrainError(f"reply had a JSON-ish span but it didn't parse: {e}\n{text[:400]}")
    raise BrainError(f"no JSON object found in reply:\n{text[:400]}")
