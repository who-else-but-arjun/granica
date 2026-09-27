"""One-off analysis: bake data/ground_truth_weights.json from the real images.

Pipeline (every step cached, so this is fast on the 2nd run):

  OpenAI vision     -> groundings.json   (closed-set food boxes)
  MobileSAM         -> segmentation.json (per-box pixel counts)
  scale anchors     -> real plate totals (curated from the digital scale LCD)
  density           -> split each plate's total grams across its items by
                       (pixels * category density) so the pixel->weight relation is
                       learnable and category-specific.

The output is the single source of ground truth the notebooks load - there is
no stochastic generator at notebook time.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from granica.config import (DATA_DIR, DINNER_TEST_DIR, DINNER_TRAIN_DIR,
                            DATASET_CONFIG_JSON, TRAIN_WEIGHTS_JSON,
                            GROUNDINGS_JSON, RUNTIME, SEGMENTATION_JSON)
from granica.grounding import load_groundings, ground_image
from granica.io_utils import dump_json, load_json, load_image_rgb
from granica.menu import load_menu, menu_for
from granica.segmentation import SAMSegmenter, load_segmentation

DENSITY_PRIOR = {
    "rice": 0.55, "bread": 0.45, "dal": 0.70, "curry": 0.72,
    "raita_curd": 0.65, "chutney_pickle": 0.40, "egg": 0.55, "snack": 0.48,
    "dessert": 0.42, "drink": 1.00, "salad": 0.30, "fruit": 0.70,
    "boilerplate": 0.45,
}
TRAY_TARE_GRAMS = 410

SCALE_ANCHORS: dict[str, dict[str, dict[str, int | None]]] = {
    "train": {
        "pre": {
            "kunj-230103056": 1073, "mithil-230103077": 1250,
            "nitish-230103022": 1177, "takshay-230102111": None,
            "vaibhav-230103066": 1208,
        },
        "post": {
            "kunj-230103056": 825, "mithil-230103077": None,
            "nitish-230103022": None, "takshay-230102111": None,
            "vaibhav-230103066": 1451,
        },
    },
    "test": {
        "pre": {
            "aakarsh-230102122": None, "archit-230101010": 1287,
            "arjun-230102125": None, "garv-230104044": 1161,
        },
        "post": {
            "aakarsh-230102122": None, "archit-230101010": None,
            "arjun-230102125": None, "garv-230104044": None,
        },
    },
}


def _plate_images():
    out = []
    selected = [s.strip() for s in os.environ.get("GRANICA_SPLITS", "train,test").split(",") if s.strip()]
    for split in selected:
        for stage in ("pre", "post"):
            out += [(split, stage, p) for p in RUNTIME.plate_images(split, stage)]
    return out


def _read_scale(path: Path) -> int | None:
    """Read the digital scale LCD value from the curated SCALE_ANCHORS table."""
    stem = path.stem
    split = "train" if "train" in str(path) else "test"
    stage = "pre" if "/pre/" in str(path).replace("\\", "/") else "post"
    if stage not in ("pre", "post"):
        stage = "pre"
    return SCALE_ANCHORS.get(split, {}).get(stage, {}).get(stem)


def _global_calibration(plates_pixels_total: list[float], plates_grams: list[float]) -> dict:
    if len(plates_pixels_total) < 2:
        return {"slope": 0.6, "intercept": TRAY_TARE_GRAMS, "n": 0}
    x = np.array(plates_pixels_total, float)
    y = np.array(plates_grams, float)
    slope, intercept = np.polyfit(x, y, 1)
    return {"slope": float(slope), "intercept": float(intercept),
            "n": len(plates_pixels_total)}


def main() -> None:
    menu = load_menu()
    whitelist = [e["name"] for e in menu_for(menu, RUNTIME.day_of_week, RUNTIME.meal)]
    cat_of = {e["name"]: e["category"] for e in menu["days"][RUNTIME.day_of_week][RUNTIME.meal]}

    seg_model = SAMSegmenter()

    # 1) grounding
    print("[1/4] OpenAI closed-set grounding for selected split images...")
    for split, stage, p in _plate_images():
        ground_image(p, menu, day=RUNTIME.day_of_week, meal=RUNTIME.meal)
        g = load_groundings().get(str(p))
        items = [i["item"] for i in g.get("items", [])] if g else []
        print(f"  {split}/{stage}/{p.name}: {items}")

    # 2) segmentation within the grounded boxes
    print("[2/4] MobileSAM segmentation inside each box...")
    for split, stage, p in _plate_images():
        g = load_groundings().get(str(p))
        if g and g.get("items"):
            boxes = [it["bbox"] for it in g["items"]]
            seg_model.segment(p, boxes, labels=[it.get("item", "food") for it in g["items"]])

    groundings = load_groundings()
    segmentation = load_segmentation()

    # 3) read the scale for every plate
    print("[3/4] reading digital scale totals...")
    totals = {}
    px_totals = []
    grams_anchored = []
    for split, stage, p in _plate_images():
        grams = _read_scale(p)
        totals[str(p)] = grams
        seg = segmentation.get(str(p))
        if seg and grams is not None:
            px_totals.append(sum(b["pixel_count"] for b in seg["boxes"]))
            grams_anchored.append(grams - TRAY_TARE_GRAMS)
    cal = _global_calibration(px_totals, grams_anchored) if grams_anchored else \
        _global_calibration([], [])

    # 4) allocate per-item weights
    print("[4/4] allocating per-item ground-truth weights...")
    plates = {}
    for split, stage, p in _plate_images():
        g = groundings.get(str(p), {})
        seg = segmentation.get(str(p), {})
        items = []
        for it, box in zip(g.get("items", []), seg.get("boxes", [])):
            item_name = it.get("menu_item") or it.get("item", "")
            px = box.get("pixel_count", 0)
            cat = cat_of.get(item_name, "curry")
            items.append({"item": item_name, "category": cat,
                          "pixels": int(px), "grams": 0.0})
        density = {i["item"]: DENSITY_PRIOR.get(i["category"], 0.6) for i in items}
        total_density = sum(i["pixels"] * density[i["item"]] for i in items) or 1.0
        grams = totals.get(str(p))
        if grams is None:
            est = total_density * cal["slope"] + cal["intercept"]
            grams = max(TRAY_TARE_GRAMS + 50, int(est))
        weight_source = "scale-anchored allocation" if totals.get(str(p)) is not None else "estimated from scale calibration"
        food_total = grams - TRAY_TARE_GRAMS
        for i in items:
            share = (i["pixels"] * density[i["item"]]) / total_density if total_density else 0
            # Keep the explicit item allocation deterministic. Adding random
            # per-item noise only makes the regression targets less consistent.
            i["grams"] = round(max(0.0, food_total * share), 1)
        delta = round(food_total - sum(i["grams"] for i in items), 1)
        if items and delta != 0:
            big = max(range(len(items)),
                      key=lambda k: items[k]["grams"]) if items else 0
            items[big]["grams"] = round(items[big]["grams"] + delta, 1)
        for item in items:
            item["weight_source"] = weight_source

        plates[f"{split}/{stage}/{p.stem}"] = {
            "split": split, "stage": stage, "person": p.stem, "name": p.stem,
            "date": RUNTIME.service_date, "meal": RUNTIME.meal,
            "image": str(p), "anchored": totals.get(str(p)),
            "tare": TRAY_TARE_GRAMS,
            "items": items,
            "total_grams": round(sum(i["grams"] for i in items), 1),
        }

    out = {"calibration": cal, "density_prior": DENSITY_PRIOR,
           "tare": TRAY_TARE_GRAMS, "plates": plates}
    out_path = DATA_DIR / "ground_truth_weights.json"
    dump_json(out, out_path)
    train_items = [
        {"plate": key, "person": plate["person"], "stage": plate["stage"],
         "image": plate["image"], "item": item["item"],
         "category": item["category"], "pixels": item["pixels"],
         "grams": item["grams"], "weight_source": item["weight_source"]}
        for key, plate in plates.items() if plate["split"] == "train"
        for item in plate["items"]
    ]
    dump_json({"source": "scale-anchored per-item allocation; not individually weighed",
               "items": train_items}, TRAIN_WEIGHTS_JSON)
    dump_json({"project": "Savor", "split_directories": {
        "train": RUNTIME.dinner_roots()["train"].name,
        "test": RUNTIME.dinner_roots()["test"].name},
        "menu_config": "menu_config.json", "student_history": "student_history.json",
        "training_weights": TRAIN_WEIGHTS_JSON.name,
        "weight_label_note": "Per-item weights allocate plate scale totals by pixel area and density priors; they are not direct item measurements."},
        DATASET_CONFIG_JSON)
    print(f"wrote {out_path} with {len(plates)} plates")
    for k in list(plates)[:3]:
        pl = plates[k]
        print(f"  {k}: total={pl['total_grams']}g anchored={pl['anchored']} items={len(pl['items'])}")


if __name__ == "__main__":
    main()
