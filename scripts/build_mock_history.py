"""Create a deterministic menu-constrained history for every DB student."""
from __future__ import annotations
import csv, random, re, sys
from datetime import date, timedelta
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from granica.config import STUDENT_HISTORY_JSON, DATASET_DIR
from granica.io_utils import dump_json
from granica.menu import load_menu, menu_for

menu = load_menu(); entries = menu_for(menu, "FRIDAY", "DINNER")
pattern = re.compile(r"^(.+)-\d{6,}-(?:train|test|dev)$", re.I)
names = sorted({pattern.match(p.stem).group(1) for p in (ROOT / "dataset" / "db").glob("*") if pattern.match(p.stem)})
rng = random.Random(20260927); people = {}; rows = []
for name in names:
    meals = []
    for days_ago in range(1, 6):
        chosen = rng.sample(entries, k=min(3, len(entries))); items = []
        meal_date = str(date(2026, 9, 25) - timedelta(days=days_ago))
        for e in chosen:
            unit = rng.choice(e["units"]); count = rng.randint(1, 2)
            served = round(count * float(e["grams_per_unit"][unit]), 1)
            taken = round(served * rng.uniform(0.65, 1.0), 1)
            wasted = round(served - taken, 1)
            items.append({
                "item": e["name"], "category": e["category"], "unit": unit,
                "count": count, "grams": taken, "served_grams": served,
                "taken_grams": taken, "wasted_grams": wasted,
            })
            rows.append({
                "student": name, "date": meal_date, "meal": "DINNER",
                "item": e["name"], "unit": unit, "count": count,
                "served_grams": served, "taken_grams": taken,
                "wasted_grams": wasted,
            })
        meals.append({
            "date": meal_date, "meal": "DINNER", "items": items,
            "served_grams": round(sum(i["served_grams"] for i in items), 1),
            "taken_grams": round(sum(i["taken_grams"] for i in items), 1),
            "wasted_grams": round(sum(i["wasted_grams"] for i in items), 1),
            "total_grams": round(sum(i["taken_grams"] for i in items), 1),
        })
    people[name] = {"name": name, "consumption": meals}
DATASET_DIR.mkdir(parents=True, exist_ok=True)
dump_json({"source": "deterministic menu-constrained mock history", "people": people}, STUDENT_HISTORY_JSON)
with STUDENT_HISTORY_JSON.with_suffix(".csv").open("w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
print(f"wrote mock history for {len(people)} students")
