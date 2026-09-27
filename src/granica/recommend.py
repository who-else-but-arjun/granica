"""Closed-set, unit-constrained meal recommendations with OpenAI.

The LLM is given the student's past consumption, today's menu and the strict
unit->grams table and is asked for an integer count of a *countable* unit per
item. A deterministic validation/repair layer then forces every prediction back
inside the closed menu + unit set, so an off-model answer can never surface."""
from __future__ import annotations

import json
import os
import re
from pathlib import Path

import numpy as np
import openai

from .config import RUNTIME
from .io_utils import dump_json, load_json
from .menu import allowed_units, all_items, grams_per_unit, menu_for
from .units import MAX_PORTIONS

TARGET_BAND = (450.0, 750.0)


def _history_block(history: dict) -> str:
    if not history:
        return "The student has no recorded meal history yet."
    lines = ["Yesterday / past portions (grams actually eaten):"]
    for meal in history.get("consumption", []):
        for e in meal.get("items", []):
            lines.append(f"  - {e['item']}: {e['grams']}g")
        lines.append(f"  (meal total {meal.get('total_grams', 0):.0f}g)")
    return "\n".join(lines)


def _menu_block(menu: dict, day: str, meal: str) -> str:
    entries = menu_for(menu, day, meal)
    lines = ["Today's menu (only these items may be recommended):"]
    for e in entries:
        lines.append(f"  - {e['name']}  [{e['category']}]")
    return "\n".join(lines)


def _unit_table_block(menu: dict) -> str:
    lines = ["For each recommendation emit a countable unit and an integer count. "
             "The unit must be one of the allowed units for the item's category and "
             "each unit maps to a fixed gram weight:"]
    seen = set()
    entries = [e for day_entries in menu["days"].values() for meal_entries in day_entries.values() for e in meal_entries]
    for e in entries:
        if e["name"] in seen:
            continue
        seen.add(e["name"])
        u = ", ".join(f"{u}={e['grams_per_unit'][u]}g" for u in e["units"])
        lines.append(f"  - {e['name']} ({e['category']}): {u}")
    return "\n".join(lines)


SYSTEM_PROMPT = (Path(__file__).resolve().parents[2] / "prompts" / "recommendation_system.md").read_text(encoding="utf-8")


def build_recommendation_prompt(menu: dict, day: str, meal: str,
                                history: dict | None) -> list[dict]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": (
            f"{_menu_block(menu, day, meal)}\n\n"
            f"{_unit_table_block(menu)}\n\n"
            f"{_history_block(history or {})}\n\n"
            "Today is a "
            f"{day.title()} {meal.lower()} service.\n\n"
            "Return JSON exactly like:\n"
            '{"recommendations":['
            '{"item":"Chicken Hyderabadi Biryani","unit":"ladle","count":2,"grams":260}'
            '], "reasoning": "string"}'
        )},
    ]


def recommend(menu: dict, day: str, meal: str, history: dict | None = None,
              out_path: Path | None = None) -> dict:
    api_key = os.environ.get("OPENAI_API_KEY")
    openai.api_key = api_key
    prompt = build_recommendation_prompt(menu, day, meal, history)
    system_msg = prompt[0]["content"]
    user_msg = prompt[1]["content"]

    response = openai.ChatCompletion.create(
        model=RUNTIME.llm_model,
        messages=[{"role": "system", "content": system_msg}, {"role": "user", "content": user_msg}],
        max_tokens=RUNTIME.llm_max_new_tokens,
    )
    text = response["choices"][0]["message"]["content"].strip()
    parsed = _extract_json(text)
    validated = validate_and_repair(parsed, menu, day, meal)
    if out_path:
        dump_json(validated, out_path)
    return validated


def _extract_json(text: str) -> dict:
    text = text.strip()
    for opener in ("```json", "```"):
        if text.startswith(opener):
            text = text[len(opener):]
    if text.endswith("```"):
        text = text[:-3]
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1:
            try:
                return json.loads(text[start:end + 1])
            except json.JSONDecodeError:
                pass
    return {"recommendations": []}


def validate_and_repair(obj: dict, menu: dict, day: str, meal: str) -> dict:
    whitelist = {e["name"]: e for e in menu_for(menu, day, meal)}
    valid_units = {e["name"]: set(e["units"]) for e in menu_for(menu, day, meal)}

    recs = []
    seen = set()
    for r in obj.get("recommendations", []):
        name = r.get("item", "")
        resolved = None
        for wn in whitelist:
            if wn.lower() == name.lower() or name.lower() in wn.lower() or wn.lower() in name.lower():
                resolved = wn
                break
        if resolved is None or resolved in seen:
            continue
        seen.add(resolved)

        unit = r.get("unit", "")
        if unit not in valid_units[resolved]:
            unit = whitelist[resolved]["units"][0]
        count = int(r.get("count", 1) or 1)
        count = max(1, min(MAX_PORTIONS, count))
        grams = round(count * float(grams_per_unit(menu, resolved, unit)), 1)
        recs.append({
            "item": resolved,
            "category": whitelist[resolved]["category"],
            "unit": unit,
            "count": count,
            "grams": grams,
        })

    if not recs:
        for e in menu_for(menu, day, meal):
            recs.append({
                "item": e["name"], "category": e["category"],
                "unit": e["units"][0], "count": 1,
                "grams": round(float(e["grams_per_unit"][e["units"][0]]), 1),
            })

    total = round(sum(r["grams"] for r in recs), 1)
    return {
        "recommendations": recs,
        "total_grams": total,
        "daily_target_band": list(TARGET_BAND),
        "within_band": TARGET_BAND[0] <= total <= TARGET_BAND[1],
        "reasoning": obj.get("reasoning", ""),
        "repaired": True,
    }
