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
from pathlib import Path
from typing import Any, Dict, Optional

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
        try:
            price = float(s.get("price", s.get("initial_price", 100.0)))
        except (TypeError, ValueError):
            price = 100.0
        try:
            hold = int(s.get("hold", s.get("initial_holdings", 0)))
        except (TypeError, ValueError):
            hold = 0
        out.append({
            "name": name,
            "price": price,
            "hold": hold,
            "sector": str(s.get("sector") or "").strip(),
            "blurb": str(s.get("blurb") or s.get("description") or "").strip(),
        })
    return out


_INT_KEYS = ("steps", "hold")
_FLOAT_KEYS = ("price", "cash", "fee", "slip")
_STR_KEYS = ("provider", "model")


def _apply_scalar_fields(cfg: Dict[str, Any], data: Dict[str, Any]) -> None:
    """Copy scalar config fields into ``cfg``, type-checking numeric values.

    Numeric fields keep their default whenever the stored value is not a real
    number (e.g. a hand-edited ``"steps": "abc"``), so the CLI cannot crash on
    ``range("abc")``. Bool is rejected too, matching the web API's behavior.
    """
    for key in _STR_KEYS:
        if key in data:
            cfg[key] = data[key]
    for key in _INT_KEYS:
        val = data.get(key)
        if isinstance(val, (int, float)) and not isinstance(val, bool):
            cfg[key] = int(val)
    for key in _FLOAT_KEYS:
        val = data.get(key)
        if isinstance(val, (int, float)) and not isinstance(val, bool):
            cfg[key] = float(val)


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
    with open(cfg_path, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2, ensure_ascii=False)
    return cfg
