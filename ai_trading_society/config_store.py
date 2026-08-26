"""
Shared user configuration store.

The web UI (homepage "Configure Traders" panel) and the CLI both read and
write the same JSON file so that a simulation configured in the browser runs
identically from the terminal.

This replaces the old `.env`-based API key loading: every API key comes from
the user's configuration (entered on the homepage), never from environment
variables.
"""

import json
import math
import os
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DEFAULT_CONFIG: Dict[str, Any] = {
    "steps": 30,
    "price": 100.0,
    "cash": 10000.0,
    "hold": 20,
    "fee": 0.001,
    "slip": 0.001,
    "provider": "openai",
    "model": "gpt-4o",
    "social_influence": 0.0,
    "player_participates": True,
    "traders": [],
    "stocks": [],
}

# Default location of the shared config file. Overridable for tests.
CONFIG_PATH = PROJECT_ROOT / "user_config.json"

# Personality presets the roster builder understands. An empty value means
# "use the default personality for this trader slot".
_VALID_PERSONALITIES = frozenset({
    "balanced", "aggressive", "conservative", "panicky",
    "greedy", "fomo_driven", "stubborn", "emotional",
})


def get_config_path() -> Path:
    """Return the path of the shared user configuration file."""
    return CONFIG_PATH


def _normalize_traders(traders: Any) -> list:
    """Coerce a raw traders payload into a clean list of dicts."""
    if not isinstance(traders, list):
        return []
    out = []
    for t in traders:
        if not isinstance(t, dict):
            continue
        personality = str(t.get("personality") or "")
        out.append({
            "name": str(t.get("name") or ""),
            "provider": str(t.get("provider") or "openai"),
            "model": str(t.get("model") or ""),
            "api_key": str(t.get("api_key") or ""),
            "base_url": str(t.get("base_url") or ""),
            "personality": personality if personality in _VALID_PERSONALITIES else "",
        })
    return out


def _normalize_stocks(stocks: Any) -> list:
    """Coerce a raw stocks payload into a clean list of dicts.

    Each stock entry carries: name, price (initial price), and
    hold (initial per-agent holdings). Duplicate names are dropped.
    """
    if not isinstance(stocks, list):
        return []
    out = []
    seen_names = set()
    for s in stocks:
        if not isinstance(s, dict):
            continue
        name = str(s.get("name") or s.get("symbol") or "").strip()
        if not name or name in seen_names:
            continue
        seen_names.add(name)
        p = _coerce_number(s.get("price", s.get("initial_price", 100.0)))
        price = max(0.01, min(p, 1_000_000.0)) if p is not None else 100.0
        h = _coerce_number(s.get("hold", s.get("initial_holdings", 0)))
        hold = int(max(0, min(h, 1_000_000))) if h is not None else 0
        out.append({
            "name": name,
            "price": price,
            "hold": hold,
            "sector": str(s.get("sector") or "").strip(),
            "blurb": str(s.get("blurb") or s.get("description") or "").strip(),
        })
    return out


_RANGES: Dict[str, Tuple[float, float]] = {
    "steps": (1, 10_000),
    "hold": (0, 1_000_000),
    "price": (0.01, 1_000_000),
    "cash": (0.0, 1e12),
    "fee": (0.0, 0.5),
    "slip": (0.0, 0.5),
    "social_influence": (0.0, 1.0),
}
_STR_KEYS = ("provider", "model")
_BOOL_KEYS = ("player_participates",)


def _coerce_number(val: Any) -> Optional[float]:
    """Return a finite float from a JSON number or numeric string, else ``None``.

    Homepage inputs arrive as JSON numbers or numeric strings (DOM input
    values are always text); both are accepted. Booleans, lists, and
    non-numeric junk are rejected so callers can fall back to defaults.
    Non-finite values (``nan``/``inf``/``infinity``, accepted by ``float()``
    and by bare JSON literals) are also rejected: they would either crash
    the later ``int()`` conversion in :func:`save_config` or poison market
    math if they ever reached a config consumer.
    """
    if isinstance(val, bool):
        return None
    if isinstance(val, (int, float)):
        num = float(val)
    elif isinstance(val, str):
        try:
            num = float(val.strip())
        except ValueError:
            return None
    else:
        return None
    return num if math.isfinite(num) else None


def _apply_scalar_fields(cfg: Dict[str, Any], data: Dict[str, Any]) -> None:
    """Copy scalar config fields into ``cfg``, type-checking and clamping numeric values.

    Numeric fields keep their default whenever the stored value is not
    interpretable as a number (e.g. a hand-edited ``"steps": "abc"``), so the
    CLI cannot crash on ``range("abc")``. Valid numeric inputs are coerced and
    clamped to safe ranges.
    """
    for key in _STR_KEYS:
        if key in data and data[key] is not None:
            cfg[key] = str(data[key])
    for key, (lo, hi) in _RANGES.items():
        if key in data:
            num = _coerce_number(data.get(key))
            if num is not None:
                kind = int if key in ("steps", "hold") else float
                cfg[key] = kind(max(lo, min(num, hi)))
    for key in _BOOL_KEYS:
        val = data.get(key)
        if isinstance(val, bool):
            cfg[key] = val


def _migrate_legacy_stocks(cfg: Dict[str, Any], data: Dict[str, Any]) -> None:
    """Populate ``cfg["stocks"]`` from the payload.

    If the payload has an explicit non-empty ``stocks`` list, it is normalized
    and used. Otherwise, a legacy single-stock config (top-level
    ``price``/``hold``) is migrated into a one-element stocks list so old
    saved configs keep working.
    """
    stocks = data.get("stocks")
    if isinstance(stocks, list) and stocks:
        cfg["stocks"] = _normalize_stocks(stocks)
        return
    # Legacy single-stock migration.
    price = cfg.get("price", 100.0)
    hold = cfg.get("hold", 0)
    cfg["stocks"] = [{
        "name": "Stock 1",
        "price": price,
        "hold": hold,
    }]


def load_config(path: Optional[str] = None) -> Dict[str, Any]:
    """
    Load the saved user configuration, falling back to defaults on any error.

    Parameters
    ----------
    path : str, optional
        Override the config file path (used by tests).

    Returns
    -------
    dict
        Full configuration with all expected keys populated.
    """
    cfg = dict(DEFAULT_CONFIG)
    cfg_path = Path(path) if path else Path(CONFIG_PATH)
    try:
        with open(cfg_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            return cfg
        _apply_scalar_fields(cfg, data)
        cfg["traders"] = _normalize_traders(data.get("traders"))
        _migrate_legacy_stocks(cfg, data)
    except (OSError, ValueError):
        pass
    return cfg


def save_config(data: Dict[str, Any], path: Optional[str] = None) -> Dict[str, Any]:
    """
    Persist user configuration to disk (defaults to user_config.json).

    Parameters
    ----------
    data : dict
        Partial or full configuration; missing keys fall back to defaults.
    path : str, optional
        Override the config file path (used by tests).

    Returns
    -------
    dict
        The normalized configuration that was written.
    """
    cfg = dict(DEFAULT_CONFIG)
    _apply_scalar_fields(cfg, data)
    cfg["traders"] = _normalize_traders(data.get("traders", cfg["traders"]))
    _migrate_legacy_stocks(cfg, data)

    cfg_path = Path(path) if path else Path(CONFIG_PATH)
    cfg_path.parent.mkdir(parents=True, exist_ok=True)
    # Keep a one-generation backup of the previous config so an accidental
    # overwrite (bad script, bad request) is always recoverable.
    if cfg_path.exists():
        try:
            (cfg_path.parent / (cfg_path.name + ".bak")).write_bytes(
                cfg_path.read_bytes()
            )
        except OSError:
            pass  # best-effort backup only

    tmp_file = cfg_path.parent / f".{cfg_path.name}.{os.getpid()}.tmp"
    try:
        with open(tmp_file, "w", encoding="utf-8") as f:
            json.dump(cfg, f, indent=2, ensure_ascii=False)
        os.replace(tmp_file, cfg_path)
    finally:
        if tmp_file.exists():
            try:
                tmp_file.unlink()
            except OSError:
                pass
    return cfg
