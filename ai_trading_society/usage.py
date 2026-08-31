"""Per-agent token and cost accounting for LLM calls.

Every provider call an agent makes is recorded here: which round it belonged
to, why it was made, how many tokens it moved, and what that cost. Running
eight traders for fifty rounds is real money, so the numbers are surfaced
live in the dashboard rather than discovered on a billing page later.

Prices live in ``model_prices.json`` and are user-maintained -- see the
``_meta`` block in that file. A model with no price row is still counted in
full; only its cost is reported as ``None`` ("unknown"), never as zero.
"""

import json
import os
import threading
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Tuple

# Where one call came from. Kept coarse on purpose: the useful question is
# "how much did the simulation cost", with repairs broken out because they
# are the one line item a prompt change can actually shrink.
CALL_KINDS = ("decision", "repair", "chat", "test")

_PRICES_FILENAME = "model_prices.json"
# Points at an alternative price file; use it to price providers this repo
# cannot verify rates for without editing the shipped table.
_PRICES_ENV_VAR = "ATS_MODEL_PRICES"

# Rough characters-per-token, used only when a provider returns no usage
# block at all. Flagged as ``estimated`` everywhere it surfaces so an
# estimate is never mistaken for a billed number.
_CHARS_PER_TOKEN = 4.0

_price_cache: Optional[Dict[str, Dict[str, float]]] = None
_price_cache_key: Optional[str] = None


def _prices_path() -> str:
    override = os.environ.get(_PRICES_ENV_VAR, "").strip()
    if override:
        return override
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), _PRICES_FILENAME)


def load_prices(force: bool = False) -> Dict[str, Dict[str, float]]:
    """Load the price table, cached per resolved path.

    A missing or malformed file is not an error: costs simply become
    unknown. Token accounting must never depend on the price table being
    present or correct.
    """
    global _price_cache, _price_cache_key
    path = _prices_path()
    if not force and _price_cache is not None and _price_cache_key == path:
        return _price_cache

    table: Dict[str, Dict[str, float]] = {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        for key, value in (raw.get("models") or {}).items():
            if not isinstance(value, dict):
                continue
            try:
                table[str(key).lower()] = {
                    "input": float(value["input"]),
                    "output": float(value["output"]),
                }
            except (KeyError, TypeError, ValueError):
                continue
    except (OSError, ValueError):
        table = {}

    _price_cache, _price_cache_key = table, path
    return table


def normalize_model(model: str) -> str:
    """Reduce a model id to the form the price table is keyed by.

    OpenRouter routes the same model as ``anthropic/claude-opus-5`` and
    ``...:free``; both should price like ``claude-opus-5``.
    """
    name = str(model or "").strip().lower()
    if ":" in name:
        name = name.split(":", 1)[0]
    if "/" in name:
        name = name.rsplit("/", 1)[-1]
    return name


def model_price(model: str) -> Optional[Dict[str, float]]:
    """USD per 1M input/output tokens for ``model``, or None if unpriced.

    Exact match first, then the longest key that prefixes the model id --
    so a price row also covers any dated snapshot of the same model.
    """
    name = normalize_model(model)
    if not name:
        return None
    table = load_prices()
    exact = table.get(name)
    if exact is not None:
        return exact
    best: Optional[Tuple[str, Dict[str, float]]] = None
    for key, price in table.items():
        if name.startswith(key) and (best is None or len(key) > len(best[0])):
            best = (key, price)
    return best[1] if best else None


def estimate_tokens(text: str) -> int:
    """Crude token count for providers that report no usage at all."""
    if not text:
        return 0
    return max(1, int(len(text) / _CHARS_PER_TOKEN))


def compute_cost(
    model: str, prompt_tokens: int, completion_tokens: int
) -> Optional[float]:
    """Cost of one call in USD, or None when the model has no price row."""
    price = model_price(model)
    if price is None:
        return None
    return (
        prompt_tokens * price["input"] + completion_tokens * price["output"]
    ) / 1_000_000.0


@dataclass
class UsageRecord:
    """One provider call."""

    step: Optional[int]
    kind: str
    provider: str
    model: str
    prompt_tokens: int
    completion_tokens: int
    cost_usd: Optional[float]
    #: True when the token counts came from ``estimate_tokens`` rather than
    #: from the provider's own usage block.
    estimated: bool = False

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


@dataclass
class UsageTotals:
    """Aggregated counters. ``cost_usd`` covers priced calls only."""

    calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost_usd: float = 0.0
    #: Calls whose model had no price row, so are missing from ``cost_usd``.
    unpriced_calls: int = 0
    #: Calls whose token counts are estimates, not provider-reported.
    estimated_calls: int = 0

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    def add(self, record: UsageRecord) -> None:
        self.calls += 1
        self.prompt_tokens += record.prompt_tokens
        self.completion_tokens += record.completion_tokens
        if record.cost_usd is None:
            self.unpriced_calls += 1
        else:
            self.cost_usd += record.cost_usd
        if record.estimated:
            self.estimated_calls += 1

    def to_dict(self) -> Dict[str, Any]:
        out = asdict(self)
        out["total_tokens"] = self.total_tokens
        out["cost_usd"] = round(self.cost_usd, 6)
        # A run in which nothing could be priced must not read as "$0.00".
        out["cost_complete"] = self.unpriced_calls == 0
        return out


# Cap on retained per-call records. Totals stay exact; only the itemized
# tail is bounded, so a long run cannot grow memory without limit.
_MAX_RECORDS = 5000


class UsageTracker:
    """Token and cost accounting for one agent.

    Agents act concurrently (``parallel_agents``), each with its own
    tracker, but the dashboard reads a tracker from the request thread while
    the agent thread writes to it, so the counters are guarded by a lock.
    """

    def __init__(self, agent_id: str = "", provider: str = "", model: str = ""):
        self.agent_id = agent_id
        self.provider = provider
        self.model = model
        self.total = UsageTotals()
        self.by_kind: Dict[str, UsageTotals] = {}
        self.by_step: Dict[int, UsageTotals] = {}
        self.records: List[UsageRecord] = []
        self._step: Optional[int] = None
        self._lock = threading.Lock()

    # -- round tagging -------------------------------------------------

    def begin_step(self, step: Optional[int]) -> None:
        """Tag subsequent calls with the round they belong to."""
        with self._lock:
            try:
                self._step = int(step) if step is not None else None
            except (TypeError, ValueError):
                self._step = None

    def end_step(self) -> None:
        with self._lock:
            self._step = None

    # -- recording -----------------------------------------------------

    def record(
        self,
        *,
        prompt_tokens: int,
        completion_tokens: int,
        kind: str = "decision",
        model: Optional[str] = None,
        provider: Optional[str] = None,
        estimated: bool = False,
        step: Optional[int] = None,
    ) -> UsageRecord:
        """File one provider call and return the record."""
        used_model = model or self.model
        prompt_tokens = max(0, int(prompt_tokens or 0))
        completion_tokens = max(0, int(completion_tokens or 0))
        record = UsageRecord(
            step=step if step is not None else self._step,
            kind=kind if kind in CALL_KINDS else "decision",
            provider=provider or self.provider,
            model=used_model,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            cost_usd=compute_cost(used_model, prompt_tokens, completion_tokens),
            estimated=bool(estimated),
        )
        with self._lock:
            self.total.add(record)
            self.by_kind.setdefault(record.kind, UsageTotals()).add(record)
            if record.step is not None:
                self.by_step.setdefault(record.step, UsageTotals()).add(record)
            self.records.append(record)
            if len(self.records) > _MAX_RECORDS:
                del self.records[: len(self.records) - _MAX_RECORDS]
        return record

    def record_estimated(
        self,
        prompt_text: str,
        response_text: str,
        *,
        kind: str = "decision",
        model: Optional[str] = None,
        provider: Optional[str] = None,
    ) -> UsageRecord:
        """Record a call whose provider returned no usage block."""
        return self.record(
            prompt_tokens=estimate_tokens(prompt_text),
            completion_tokens=estimate_tokens(response_text),
            kind=kind,
            model=model,
            provider=provider,
            estimated=True,
        )

    # -- reading -------------------------------------------------------

    def step_totals(self, step: int) -> UsageTotals:
        with self._lock:
            return self.by_step.get(step, UsageTotals())

    def to_dict(self, step: Optional[int] = None) -> Dict[str, Any]:
        """Serializable snapshot; ``step`` adds that round's slice."""
        with self._lock:
            out: Dict[str, Any] = {
                "agent_id": self.agent_id,
                "provider": self.provider,
                "model": self.model,
                "total": self.total.to_dict(),
                "by_kind": {k: v.to_dict() for k, v in self.by_kind.items()},
                "priced": model_price(self.model) is not None,
            }
            if step is not None:
                out["step"] = self.by_step.get(step, UsageTotals()).to_dict()
            return out


def extract_usage(response: Any) -> Optional[Tuple[int, int]]:
    """Pull (prompt_tokens, completion_tokens) out of a provider response.

    Covers the three shapes this project talks to: OpenAI-compatible
    (``usage.prompt_tokens`` / ``completion_tokens``), Anthropic
    (``usage.input_tokens`` / ``output_tokens``) and Google
    (``usage_metadata.prompt_token_count`` / ``candidates_token_count``).
    Returns None when the response carries no usage block, so the caller can
    fall back to an estimate.
    """
    if response is None:
        return None

    if isinstance(response, dict):
        usage = response.get("usage") or response.get("usage_metadata")
    else:
        usage = getattr(response, "usage", None)
        if usage is None:
            usage = getattr(response, "usage_metadata", None)
    if usage is None:
        return None

    def _read(*names: str) -> Optional[int]:
        for name in names:
            value = (
                usage.get(name) if isinstance(usage, dict)
                else getattr(usage, name, None)
            )
            if value is None:
                continue
            try:
                return max(0, int(value))
            except (TypeError, ValueError):
                continue
        return None

    prompt = _read("prompt_tokens", "input_tokens", "prompt_token_count")
    completion = _read("completion_tokens", "output_tokens", "candidates_token_count")
    if prompt is None and completion is None:
        return None
    return prompt or 0, completion or 0


def agent_usage(agent: Any) -> Optional[UsageTracker]:
    """The tracker for an agent, looking through a persona wrapper."""
    for candidate in (agent, getattr(agent, "base_agent", None)):
        tracker = getattr(candidate, "usage", None)
        if isinstance(tracker, UsageTracker):
            return tracker
    return None


def collect_usage(agents: Any) -> Dict[str, Any]:
    """Aggregate usage across a roster, for run metadata and reports.

    Accepts either wrapped (``TraitAgent``) or bare agents; an agent with no
    tracker (the human player, an offline agent) is skipped.
    """
    per_agent: List[Dict[str, Any]] = []
    grand = UsageTotals()
    for agent in agents or []:
        tracker = agent_usage(agent)
        if tracker is None:
            continue
        snapshot = tracker.to_dict()
        per_agent.append(snapshot)
        totals = snapshot["total"]
        grand.calls += totals["calls"]
        grand.prompt_tokens += totals["prompt_tokens"]
        grand.completion_tokens += totals["completion_tokens"]
        grand.cost_usd += totals["cost_usd"]
        grand.unpriced_calls += totals["unpriced_calls"]
        grand.estimated_calls += totals["estimated_calls"]
    return {"total": grand.to_dict(), "agents": per_agent}


def format_cost(cost: Optional[float], complete: bool = True) -> str:
    """Render a cost for the CLI / dashboard.

    An unknown price shows as such rather than as zero, and a partially
    priced total is marked so it is not read as the whole bill.
    """
    if cost is None:
        return "n/a"
    if 0 < cost < 0.01:
        text = f"${cost:.4f}"
    else:
        text = f"${cost:,.2f}"
    return text if complete else text + "+"
