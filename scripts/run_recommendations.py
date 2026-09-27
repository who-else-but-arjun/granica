"""Generate one validated OpenAI recommendation per person from cached history."""
from __future__ import annotations
import os, sys, time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from granica.config import DATA_DIR, RECOMMENDATIONS_JSON, STUDENT_HISTORY_JSON, RUNTIME
from granica.io_utils import dump_json, load_json
from granica.menu import load_menu
from granica.recommend import recommend

menu = load_menu()
people = load_json(STUDENT_HISTORY_JSON).get("people", {}) if STUDENT_HISTORY_JSON.exists() else load_json(DATA_DIR / "people.json").get("people", {})
out = {}
delay = float(os.environ.get("GRANICA_OPENAI_DELAY", "12"))
for person, history in sorted(people.items()):
    try:
        out[person] = recommend(menu, RUNTIME.day_of_week, RUNTIME.meal, history)
        print(person, out[person].get("total_grams"), flush=True)
    except Exception as exc:
        out[person] = {"error": f"{type(exc).__name__}: {exc}"}
        print(person, "error", out[person]["error"], flush=True)
    time.sleep(delay)
dump_json({"service_date": RUNTIME.service_date, "meal": RUNTIME.meal, "model": RUNTIME.llm_model, "recommendations": out}, RECOMMENDATIONS_JSON)
