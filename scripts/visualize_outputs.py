"""Create the requested image overlays, regression figures, and final summary."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from granica.config import ARTIFACTS_DIR, DATA_DIR, FACE_METRICS_JSON, RECOMMENDATIONS_JSON, STUDENT_HISTORY_JSON, WEIGHT_MODELS_JSON
from granica.grounding import load_groundings
from granica.io_utils import dump_json, load_json
from granica.segmentation import load_segmentation
from granica.weights import evaluate_test


def _mkdir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


def _grounding_images(groundings: dict, segmentation: dict) -> list[str]:
    base = ARTIFACTS_DIR / "visualizations" / "grounding"
    _mkdir(base)
    for image, rec in groundings.items():
        source = Path(image)
        split = source.parent.parent.name
        stage = source.parent.name
        seg_path = ARTIFACTS_DIR / "visualizations" / "segmentation" / split / stage / f"{source.stem}_segmentation.png"
        img = cv2.imread(str(seg_path)) if seg_path.exists() else cv2.imread(image)
        if img is None:
            continue
        seg_boxes = segmentation.get(image, {}).get("boxes", [])
        for idx, item in enumerate(rec.get("items", [])):
            # Use SAM's food-region refinement when available so the review
            # overlay stays tight while still showing the original item label.
            box = (seg_boxes[idx].get("refined_bbox", item["bbox"])
                   if idx < len(seg_boxes) else item["bbox"])
            x1, y1, x2, y2 = box
            cv2.rectangle(img, (x1, y1), (x2, y2), (40, 210, 80), 3)
            label = item.get("item", "food")[:36]
            font = cv2.FONT_HERSHEY_SIMPLEX
            (tw, th), baseline = cv2.getTextSize(label, font, 0.65, 2)
            tx = max(0, min(x1, img.shape[1] - tw - 8))
            ty = y1 - 6 if y1 - th - baseline - 8 >= 0 else min(img.shape[0] - 1, y1 + th + baseline + 8)
            top = max(0, ty - th - baseline - 6)
            bottom = min(img.shape[0] - 1, ty + baseline + 4)
            cv2.rectangle(img, (tx, top), (min(img.shape[1] - 1, tx + tw + 8), bottom), (40, 125, 45), -1)
            cv2.putText(img, label, (tx + 4, ty), font, 0.65, (255, 255, 255), 2, cv2.LINE_AA)
        dest = _mkdir(base / split / stage) / f"{source.stem}_grounding.png"
        cv2.imwrite(str(dest), img)
    return [str(p) for p in sorted(base.rglob("*.png"))]


def _regressions(gt: dict, models: dict) -> list[str]:
    out = _mkdir(ARTIFACTS_DIR / "visualizations" / "regression")
    by_cat = {}
    for plate in gt.get("plates", {}).values():
        if plate.get("split") != "train" or plate.get("anchored") is None:
            continue
        for it in plate.get("items", []):
            by_cat.setdefault(it["category"], []).append((it["pixels"], it["grams"]))
    paths = []
    for cat, pts in sorted(by_cat.items()):
        x = np.array([p[0] for p in pts], float); y = np.array([p[1] for p in pts], float)
        m = models.get(cat, models.get("__global__", {"slope": 0, "intercept": 0}))
        xx = np.linspace(0, max(1, x.max() * 1.08), 100)
        plt.figure(figsize=(7, 5)); plt.scatter(x, y, label="train observations")
        plt.plot(xx, m["slope"] * xx + m["intercept"], color="crimson", label=f"g={m['slope']:.4f}px+{m['intercept']:.1f}")
        plt.xlabel("Segmented pixels"); plt.ylabel("Weight (g)"); plt.title(f"Pixel to weight: {cat}"); plt.legend(); plt.tight_layout()
        p = out / f"{cat}_pixel_vs_weight.png"; plt.savefig(p, dpi=150); plt.close(); paths.append(str(p))
    return paths


def _inference(gt: dict, preds: dict) -> dict:
    out = _mkdir(ARTIFACTS_DIR / "visualizations" / "inference")
    keys = sorted(k for k, plate in gt.get("plates", {}).items()
                  if k.startswith("test/") and plate.get("anchored") is not None)
    labels = [k.split("/")[-1] for k in keys]
    actual = [gt["plates"][k]["total_grams"] for k in keys]
    predicted = [preds.get(k, {}).get("total_grams", 0) for k in keys]
    plt.figure(figsize=(10, 5)); x = np.arange(len(labels)); plt.bar(x - .18, actual, .36, label="ground truth"); plt.bar(x + .18, predicted, .36, label="inference")
    plt.xticks(x, labels, rotation=45, ha="right"); plt.ylabel("Total food weight (g)"); plt.title("Test inference: predicted vs ground truth"); plt.legend(); plt.tight_layout()
    comparison = out / "test_inference_comparison.png"; plt.savefig(comparison, dpi=150); plt.close()
    measured = [(k, p) for k, p in gt["plates"].items()
                if p["split"] == "test" and p.get("anchored") is not None]
    errors = [preds.get(k, {}).get("total_grams", 0) - p["total_grams"] for k, p in measured]
    labels = [k.split("/")[-1] for k, _ in measured]
    plt.figure(figsize=(10, 5)); colours = ["#2f855a" if e >= 0 else "#c53030" for e in errors]
    plt.bar(np.arange(len(errors)), errors, color=colours); plt.axhline(0, color="black", linewidth=1)
    plt.xticks(np.arange(len(labels)), labels, rotation=45, ha="right"); plt.ylabel("Signed error (g)")
    plt.title("Savor test inference errors: positive overestimate / negative underestimate")
    plt.tight_layout(); error_path = out / "test_inference_signed_errors.png"; plt.savefig(error_path, dpi=150); plt.close()
    return {"comparison": str(comparison), "signed_errors": str(error_path)}


def _face_similarity() -> list[str]:
    from granica.faceid import FaceIdentifier
    from granica.io_utils import load_image_rgb
    out = _mkdir(ARTIFACTS_DIR / "visualizations" / "face_similarity")
    db = ROOT / "dataset" / "db"
    gallery_paths = sorted(p for p in db.glob("*.*")
                           if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
                           and p.stem.lower().endswith("-train"))
    target_paths = sorted(p for p in db.glob("*.*")
                          if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
                          and p.stem.lower().endswith("-test"))
    cached = sorted((out / "test_vs_train").glob("*_vs_train_gallery.png"))
    if cached and len(cached) >= max(0, len(target_paths) - 1):
        return [str(path) for path in cached]
    ident = FaceIdentifier()
    gallery = [(p, ident._embed(load_image_rgb(p))) for p in gallery_paths]
    gallery = [(p, e) for p, e in gallery if e is not None]
    targets = [(p, ident._embed(load_image_rgb(p))) for p in target_paths]
    targets = [(p, e) for p, e in targets if e is not None]
    plots = []
    for target, target_emb in targets:
        scores = [(p.stem, float(np.dot(target_emb, emb))) for p, emb in gallery]
        scores.sort(key=lambda x: x[1], reverse=True)
        labels = [x[0] for x in scores]; values = [x[1] for x in scores]
        target_person = target.stem.rsplit("-", 2)[0]
        plt.figure(figsize=(12, 6)); colours = ["tab:green" if name.rsplit("-", 1)[0] == target_person else "tab:blue" for name in labels]
        plt.bar(np.arange(len(labels)), values, color=colours); plt.xticks(np.arange(len(labels)), labels, rotation=70, ha="right", fontsize=7)
        plt.ylim(0, 1.05); plt.ylabel("Cosine similarity"); plt.title(f"Test face {target.name} vs TRAIN-only gallery"); plt.tight_layout()
        path = out / "test_vs_train" / f"{target.stem}_vs_train_gallery.png"
        path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(path, dpi=150); plt.close(); plots.append(str(path))
    return plots


def _recommendation_history(recs: dict, history: dict) -> list[str]:
    out = _mkdir(ARTIFACTS_DIR / "visualizations" / "recommendation_history")
    paths = []
    for person, result in sorted(recs.items()):
        if not isinstance(result, dict) or result.get("error"):
            continue
        meals = history.get("people", {}).get(person, {}).get("consumption", [])
        prior = {}
        for meal in meals:
            for item in meal.get("items", []):
                key = item.get("item", "")
                prior.setdefault(key, []).append(float(item.get("taken_grams", item.get("grams", 0))))
        suggested = {x["item"]: float(x.get("grams", 0)) for x in result.get("recommendations", [])}
        names = sorted(set(prior) | set(suggested))
        if not names:
            continue
        avg_taken = [float(np.mean(prior.get(name, [0]))) for name in names]
        today = [suggested.get(name, 0.0) for name in names]
        y = np.arange(len(names)); h = 0.38
        plt.figure(figsize=(10, max(4.5, len(names) * .42)))
        plt.barh(y - h / 2, avg_taken, h, label="recent average taken", color="#4c78a8")
        plt.barh(y + h / 2, today, h, label="recommended today", color="#f28e2b")
        plt.yticks(y, names); plt.xlabel("Food amount (g)")
        plt.title(f"{person}: recommendation compared with recent consumption")
        plt.legend(); plt.tight_layout()
        path = out / f"{person}_recommendation_vs_history.png"
        plt.savefig(path, dpi=150); plt.close(); paths.append(str(path))
    return paths


def build_summary() -> dict:
    groundings = load_groundings(); segmentation = load_segmentation()
    gt = load_json(DATA_DIR / "ground_truth_weights.json")
    models = load_json(WEIGHT_MODELS_JSON) if WEIGHT_MODELS_JSON.exists() else {}
    preds = load_json(DATA_DIR / "test_predictions.json") if (DATA_DIR / "test_predictions.json").exists() else {}
    people = load_json(DATA_DIR / "people.json") if (DATA_DIR / "people.json").exists() else {}
    recs = load_json(RECOMMENDATIONS_JSON) if RECOMMENDATIONS_JSON.exists() else {}
    waste = []
    for person, stages in sorted({p["person"]: p for p in gt.get("plates", {}).values()}.items()):
        before = next((p for p in gt["plates"].values() if p["person"] == person and p["stage"] == "pre"), None)
        after = next((p for p in gt["plates"].values() if p["person"] == person and p["stage"] == "post"), None)
        if not before or not after: continue
        waste.append({"person": person, "before_grams": before["total_grams"], "after_grams": after["total_grams"], "food_wasted_grams": round(after["total_grams"], 1), "food_consumed_grams": round(before["total_grams"] - after["total_grams"], 1), "items_left": after["items"]})
    mock_history = load_json(STUDENT_HISTORY_JSON) if STUDENT_HISTORY_JSON.exists() else {}
    summary = {
        "grounding": {k: {"items": v.get("items", []), "dropped": v.get("dropped", 0), "model": v.get("model")} for k, v in sorted(groundings.items())},
        "segmentation": {k: {"num_boxes": v.get("num_boxes", 0), "boxes": v.get("boxes", []), "image_size": v.get("image_size")} for k, v in sorted(segmentation.items())},
        "weights": {"models": models, "test_predictions": preds,
                    "test_metrics": evaluate_test(gt, preds),
                    "evaluation": gt.get("evaluation", {}),
                    "people_history": people},
        "face_matching": load_json(FACE_METRICS_JSON) if FACE_METRICS_JSON.exists() else {},
        "recommendations": recs,
        "food_waste_by_before_after_pair": waste,
        "visualizations": {"grounding_and_masks": _grounding_images(groundings, segmentation), "regression": _regressions(gt, models), "inference": _inference(gt, preds), "face_similarity": _face_similarity(), "recommendation_vs_history": _recommendation_history(recs.get("recommendations", {}), mock_history), "segmentation_dir": str(ARTIFACTS_DIR / "visualizations" / "segmentation")},
    }
    path = ROOT / "artifacts" / "reports" / "pipeline_summary.json"; path.parent.mkdir(parents=True, exist_ok=True)
    dump_json(summary, path); return summary


if __name__ == "__main__":
    s = build_summary(); print(f"wrote {ROOT / 'artifacts' / 'reports' / 'pipeline_summary.json'}; groundings={len(s['grounding'])} segmentations={len(s['segmentation'])} waste_pairs={len(s['food_waste_by_before_after_pair'])}")
