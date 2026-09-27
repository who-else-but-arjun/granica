"""Per-category pixel->weight linear model + consumption.

Training/eval data lives in the *static* file ``data/ground_truth_weights.json``
(baked once by running ``scripts/analyze_images.py`` over the real plate
images). It holds, per plate, the menu items actually present, the pixel area
SAM found for each, and the ground-truth gram weight (total anchored to the real
scale reading where the digital display is readable).

The notebook loads this file, fits ``grams = a * pixels + b`` per food category
on the TRAIN split, then predicts the held-out TEST split and compares to the
baked test GT.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .config import DATA_DIR, ROOT, CONSUMPTION_JSON, SEGMENTATION_JSON, WEIGHT_MODELS_JSON, RUNTIME
from .io_utils import dump_json, load_json
from .menu import category_of

GT_JSON = DATA_DIR / "ground_truth_weights.json"

def load_ground_truth(path: Path = None) -> dict:
    return load_json(path or GT_JSON)


def _train_points(gt: dict):
    """Flatten the GT file into (category, [(pixels, grams)]) for all train plates.

    Both the ``pre`` (full portions) and ``post`` (leftover portions) images are
    valid pixel<->gram observations of the same foods, so they are pooled to give
    the tiny dataset enough points per category to fit.
    """
    by_cat: dict[str, list[tuple[float, float]]] = {}
    for key, plate in gt["plates"].items():
        if plate["split"] != "train" or plate.get("anchored") is None:
            continue
        for it in plate["items"]:
            by_cat.setdefault(it["category"], []).append((it["pixels"], it["grams"]))
    return by_cat


def fit_models(gt: dict, out_path: Path = WEIGHT_MODELS_JSON) -> dict:
    by_cat = _train_points(gt)
    models, pool_x, pool_y = {}, [], []
    for cat, pts in sorted(by_cat.items()):
        xs = np.array([p[0] for p in pts], float)
        ys = np.array([p[1] for p in pts], float)
        if len(pts) >= RUNTIME.min_points_per_category:
            slope, intercept = np.polyfit(xs, ys, 1)
        else:
            pool_x.extend(xs.tolist())
            pool_y.extend(ys.tolist())
            slope, intercept = 0.0, float(ys.mean()) if len(ys) else 0.0
        preds = slope * xs + intercept
        ss_res = float(np.sum((ys - preds) ** 2))
        ss_tot = float(np.sum((ys - ys.mean()) ** 2)) if len(ys) > 1 else 0.0
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
        models[cat] = {
            "slope": float(slope), "intercept": float(intercept), "n": len(pts),
            "r2": float(r2), "mae": float(np.mean(np.abs(ys - preds))) if len(ys) else 0.0,
            "mse": float(np.mean((ys - preds) ** 2)) if len(ys) else 0.0,
            "used": len(pts) >= RUNTIME.min_points_per_category,
        }

    if pool_x:
        gxs = np.array(pool_x, float)
        gys = np.array(pool_y, float)
        a, b = np.polyfit(gxs, gys, 1)
        models["__global__"] = {
            "slope": float(a), "intercept": float(b), "n": len(pool_x),
            "r2": 0.0, "mae": float(np.mean(np.abs(gys - (a * gxs + b)))),
            "mse": float(np.mean((gys - (a * gxs + b)) ** 2)),
            "used": True,
        }
    dump_json(models, out_path)
    return models


def predict_plate_items(items: list[dict], menu: dict, models: dict) -> dict:
    preds = []
    for it in items:
        cat = category_of(menu, it.get("item", ""))
        m = models.get(cat)
        if m is None or not m.get("used"):
            m = models.get("__global__", {"slope": 0.6, "intercept": 5.0})
        px = float(it.get("pixels", 0))
        grams = m["slope"] * px + m["intercept"]
        preds.append({"item": it["item"], "category": cat, "pixels": px,
                      "grams": round(max(0.0, grams), 1)})
    return {"predicted_items": preds,
            "total_grams": round(sum(p["grams"] for p in preds), 1)}


def apply_to_test(gt: dict, segmentation: dict, menu: dict, models: dict,
                  out_path: Path = None) -> dict:
    """Predict every TEST plate (pre and post) using the fitted models.

    Uses closed-set image groundings and SAM pixel counts for each test plate.
    """
    from .grounding import load_groundings

    gtbl = load_groundings()
    preds = {}
    for key, plate in gt["plates"].items():
        if plate["split"] != "test":
            continue
        seg = segmentation.get(plate["image"])
        if not seg:
            continue
        g = gtbl.get(plate["image"])
        items = []
        if g:
            items = [{"item": it["item"], "pixels": b.get("pixel_count", 0)}
                     for it, b in zip(g["items"], seg["boxes"])]
        elif plate.get("items"):
            items = [{"item": it.get("item", ""), "pixels": 0}
                     for it in plate["items"]]
        prediction = predict_plate_items(items, menu, models)
        preds[key] = prediction
    if out_path:
        dump_json(preds, out_path)
    return preds


def evaluate_test(gt: dict, test_preds: dict) -> dict:
    """Per-plate & per-category error of the TEST predictions vs baked GT."""
    plate_errs, abs_errs = [], []
    cat_errs: dict[str, list[float]] = {}
    cat_abs: dict[str, list[float]] = {}
    for key, plate in gt["plates"].items():
        if plate["split"] != "test" or plate.get("anchored") is None:
            continue
        pred = test_preds.get(key, {})
        pg = pred.get("total_grams", 0.0)
        plate_errs.append(pg - plate["total_grams"])
        abs_errs.append(abs(pg - plate["total_grams"]))
        pg_items = {i["item"]: i["grams"] for i in pred.get("predicted_items", [])}
        for it in plate["items"]:
            pe = pg_items.get(it["item"], 0.0) - it["grams"]
            cat_errs.setdefault(it["category"], []).append(pe)
            cat_abs.setdefault(it["category"], []).append(abs(pe))
    plate_errs = np.array(plate_errs, float)
    abs_errs = np.array(abs_errs, float)
    by_cat = {}
    for c in cat_errs:
        e = np.array(cat_errs[c], float)
        a = np.array(cat_abs[c], float)
        by_cat[c] = {"mae": float(a.mean()), "rmse": float(np.sqrt((e**2).mean())), "n": len(e)}
    return {
        "n_plates": int(len(plate_errs)),
        "mae_total": float(abs_errs.mean()) if len(abs_errs) else 0.0,
        "mse_total": float(np.mean(plate_errs ** 2)) if len(plate_errs) else 0.0,
        "rmse_total": float(np.sqrt((plate_errs ** 2).mean())) if len(plate_errs) else 0.0,
        "mean_signed_error": float(plate_errs.mean()) if len(plate_errs) else 0.0,
        "per_category": by_cat,
    }


def build_consumption(gt: dict, test_preds: dict, out_path: Path = None) -> dict:
    """Per-person food intake = pre_total - post_total.

    Train intake uses the baked ground-truth weights. Test intake uses the model
    prediction (pre - post), i.e. exactly the inference the model supports.
    """
    hist = {"people": {}}

    for key, plate in gt["plates"].items():
        pid = plate["person"]
        rec = hist["people"].setdefault(pid, {"name": pid, "roll": plate.get("roll", ""),
                                              "meals": []})
        rec["meals"].append({
            "split": plate["split"], "date": plate["date"], "meal": "DINNER",
            "stage": plate["stage"], "source": "ground_truth_scale",
            "items": plate["items"], "total_grams": plate["total_grams"],
            "consumed_grams": plate.get("consumed_grams"),
        })

    for key, pred in test_preds.items():
        if "pre" not in key:
            continue
        post_key = key.replace("pre", "post")
        post = test_preds.get(post_key, {})
        pid = gt["plates"].get(key, {}).get("person", key)
        consumed = pred["total_grams"] - post.get("total_grams", 0.0)
        hist["people"].setdefault(pid, {"name": pid, "meals": []})["meals"].append({
            "split": "test", "date": RUNTIME.service_date, "meal": "DINNER",
            "stage": "inference", "source": "model_prediction",
            "pre_grams": pred["total_grams"], "post_grams": post.get("total_grams"),
            "consumed_grams": round(consumed, 1), "items": pred.get("predicted_items", []),
        })

    if out_path:
        dump_json(hist, out_path)
    return hist
