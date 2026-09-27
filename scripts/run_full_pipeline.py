"""Run the complete cached Granica training, evaluation, and recommendation pipeline."""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from granica.config import ARTIFACTS_DIR, DATA_DIR, RECOMMENDATIONS_JSON, STUDENT_HISTORY_JSON, RUNTIME
from granica.faceid import FaceIdentifier
from granica.grounding import OpenAIGrounding, load_groundings
from granica.io_utils import dump_json, load_json
from granica.menu import load_menu, menu_items
from granica.recommend import recommend
from granica.segmentation import SAMSegmenter
from granica.weights import (apply_to_test, build_consumption, evaluate_test,
                             fit_models)


def all_images():
    return [p for split in ("train", "test") for stage in ("pre", "post") for p in RUNTIME.plate_images(split, stage)]


def main():
    os.environ.setdefault("YOLO_CONFIG_DIR", str(ROOT / ".ultralytics"))
    os.environ.setdefault("INSIGHTFACE_HOME", str(ROOT / ".insightface"))
    menu = load_menu(); whitelist = menu_items(menu, RUNTIME.day_of_week, RUNTIME.meal); images = all_images()
    delay = float(os.environ.get("GRANICA_OPENAI_DELAY", "12"))
    print(f"[1/7] OpenAI grounding {len(images)} images (cached calls are free; live buffer={delay}s)", flush=True)
    gm = OpenAIGrounding()
    sm = SAMSegmenter()
    for p in images:
        rec = gm.ground(p, whitelist)
        print(f"  {p.name}: {len(rec.get('items', []))} items", flush=True)
    print("[2/7] MobileSAM segmentation", flush=True)
    groundings = load_groundings()
    for p in images:
        rec = groundings.get(str(p), {})
        boxes = [x["bbox"] for x in rec.get("items", [])]
        rec = sm.segment(p, boxes, labels=[x["item"] for x in rec.get("items", [])])
        print(f"  {p.name}: {rec['num_boxes']} boxes", flush=True)
    print("[3/7] ground-truth weights", flush=True)
    from analyze_images import main as analyze_main
    analyze_main()
    gt = load_json(DATA_DIR / "ground_truth_weights.json")
    print("[4/7] fitted weight models and test inference", flush=True)
    models = fit_models(gt)
    from granica.segmentation import load_segmentation
    seg = load_segmentation()
    preds = apply_to_test(gt, seg, menu, models, out_path=DATA_DIR / "test_predictions.json")
    print(evaluate_test(gt, preds))
    build_consumption(gt, preds, out_path=DATA_DIR / "people.json")
    print("[5/7] face gallery and metrics", flush=True)
    ident = FaceIdentifier(); ident.build_gallery(RUNTIME.db_dir); ident.evaluate(RUNTIME.db_dir)
    print("[6/7] recommendations", flush=True)
    import runpy
    runpy.run_path(str(ROOT / "scripts" / "build_mock_history.py"))
    people = load_json(STUDENT_HISTORY_JSON).get("people", {}); recs = {}
    for person, history in sorted(people.items()):
        for attempt in range(4):
            try:
                recs[person] = recommend(menu, RUNTIME.day_of_week, RUNTIME.meal, history)
                break
            except Exception as exc:
                if attempt == 3: recs[person] = {"error": str(exc)}
                else: time.sleep(15 * (attempt + 1))
        time.sleep(delay)
        print(f"  {person}: {recs[person].get('total_grams', 'error')}", flush=True)
    dump_json({"service_date": RUNTIME.service_date, "meal": RUNTIME.meal, "recommendations": recs}, RECOMMENDATIONS_JSON)
    print("[7/7] visualizations and comprehensive summary", flush=True)
    from visualize_outputs import build_summary
    build_summary()


if __name__ == "__main__": main()
