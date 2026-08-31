"""Prompt template versioning, so an old report stays readable.

A run's decisions only mean something next to the prompt that produced
them. Edit the system prompt and yesterday's report silently describes a
different experiment. Two identifiers guard against that:

``PROMPT_TEMPLATE_VERSION``
    Bumped by hand when the prompt changes in a way that should invalidate
    comparison with earlier runs. It carries intent -- "this is a different
    prompt generation" -- which a hash cannot.

``prompt_fingerprint()``
    A content hash of the prompt actually sent. It catches the edit someone
    forgot to bump the version for, and distinguishes a custom or persona
    prompt from the shipped default.

Both are recorded per agent in the run metadata; ``describe_prompt()``
builds that record.
"""

import hashlib
from typing import Any, Dict, Optional, Tuple

#: Bump on any deliberate change to the shipped system prompt. MAJOR.MINOR:
#: MAJOR for a change that makes runs incomparable (a different task
#: framing, a changed output contract), MINOR for wording that should not
#: move behaviour much.
#:
#: History:
#:   1.0 - first versioned prompt (v0.3.0): JSON decision contract, simple
#:         and deep reasoning variants, deep-mode optional extras.
PROMPT_TEMPLATE_VERSION = "1.0"

_FINGERPRINT_CHARS = 12


def prompt_fingerprint(text: str) -> str:
    """Short stable content hash of a prompt.

    Truncated SHA-256: this identifies a template, it does not defend
    against anyone deliberately constructing a collision.
    """
    digest = hashlib.sha256((text or "").encode("utf-8")).hexdigest()
    return digest[:_FINGERPRINT_CHARS]


def describe_prompt(agent: Any) -> Optional[Dict[str, Any]]:
    """Prompt provenance for one agent, or None if it has no prompt.

    ``source`` distinguishes three cases:

    ``"default"``
        The shipped template, verbatim.
    ``"persona+default"``
        The persona layer's disposition prepended to the shipped template --
        the normal shape for a trader with a personality. The rules half is
        still this version's, so the template version remains meaningful.
    ``"custom"``
        A caller-supplied prompt. The template version says nothing about
        the text; only the fingerprint identifies it.
    """
    base = getattr(agent, "base_agent", agent)
    system_prompt = getattr(base, "system_prompt", None)
    if not isinstance(system_prompt, str) or not system_prompt:
        return None

    builder = getattr(base, "_default_system_prompt", None)
    deep = bool(getattr(agent, "deep", False))
    source = "custom"
    if callable(builder):
        variants: Tuple[str, ...]
        try:
            variants = (builder(deep=False), builder(deep=True))
        except TypeError:
            # A subclass with a different signature: treat the prompt as
            # custom rather than guessing.
            variants = ()
        if system_prompt in variants:
            source = "default"
        elif any(system_prompt.endswith(v) for v in variants):
            source = "persona+default"

    return {
        "template_version": PROMPT_TEMPLATE_VERSION,
        "fingerprint": prompt_fingerprint(system_prompt),
        "source": source,
        "deep": deep,
        "chars": len(system_prompt),
    }


def shipped_fingerprints() -> Dict[str, str]:
    """Fingerprints of both shipped prompt variants.

    Recorded once per run so a report can be checked against the exact
    template text this version shipped, even for agents that overrode it.
    """
    from .agents.external_ai_agent import ExternalAIAgent

    return {
        "simple": prompt_fingerprint(ExternalAIAgent._default_system_prompt(deep=False)),
        "deep": prompt_fingerprint(ExternalAIAgent._default_system_prompt(deep=True)),
    }
