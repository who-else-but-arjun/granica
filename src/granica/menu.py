"""Static menu config loader.

The full 7-day x 3-meal Barak mess menu lives in ``data/menu_config.json``.
Everything downstream reads from here so the pipeline can never "discover" an
item that is not on the printed menu.
"""
from __future__ import annotations

import json
import re
from collections import OrderedDict
from pathlib import Path

from .config import MENU_CONFIG_JSON
from .units import UNIT_TO_GRAMS

DAYS = ["MONDAY", "TUESDAY", "WEDNESDAY", "THURSDAY", "FRIDAY", "SATURDAY", "SUNDAY"]
MEALS = ["BREAKFAST", "LUNCH", "DINNER"]

# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------
def load_menu(path: Path = MENU_CONFIG_JSON) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def menu_for(menu: dict, day: str, meal: str) -> list[dict]:
    return menu["days"][day.upper()][meal.upper()]


def menu_items(menu: dict, day: str, meal: str) -> list[str]:
    """Whitelist of dish names the pipeline may ever output."""
    return [e["name"] for e in menu_for(menu, day, meal)]


def all_items(menu: dict) -> list[str]:
    return [e["name"] for d in menu["days"].values() for m in d.values() for e in m]


# ---------------------------------------------------------------------------
# Closed-set resolution
# ---------------------------------------------------------------------------
def resolve_to_menu_item(label: str, whitelist: list[str]) -> str | None:
    """Map a free-form VLM/LLM label back to the menu whitelist.

    Returns the best whitelist match, or None if nothing is close enough - in
    which case the caller must DROP the item. This is the mechanism that keeps
    grounding and recommendations inside the printed menu.
    """
    if not label:
        return None
    cand = label.strip().lower()
    cand = re.sub(r"\s+", " ", cand).strip(" .-")

    # 1. exact / substring match on the whitelist
    for w in whitelist:
        wl = w.lower()
        if cand == wl or wl in cand or cand in wl:
            return w
    # 2. token-overlap fallback (handles 'chana dal' vs 'chana daal')
    best, best_score = None, 0
    for w in whitelist:
        wt = set(w.lower().split())
        ct = set(cand.split())
        score = len(wt & ct) / max(1, len(wt))
        if score > best_score:
            best, best_score = w, score
    return best if best_score >= 0.75 else None


def on_menu(label: str, whitelist: list[str]) -> bool:
    return resolve_to_menu_item(label, whitelist) is not None


def _find_entry(menu: dict, dish_name: str):
    for day in DAYS:
        for meal in MEALS:
            for e in menu["days"][day][meal]:
                if e["name"].lower() == dish_name.lower():
                    return e
    return None


def category_of(menu: dict, dish_name: str) -> str:
    e = _find_entry(menu, dish_name)
    return e["category"] if e else "curry"


def grams_per_unit(menu: dict, dish_name: str, unit: str) -> float | None:
    e = _find_entry(menu, dish_name)
    return e["grams_per_unit"].get(unit) if e else None


def allowed_units(menu: dict, dish_name: str) -> list[str]:
    e = _find_entry(menu, dish_name)
    return list(e["units"]) if e else []


def categories() -> list[str]:
    return list(UNIT_TO_GRAMS.keys())
