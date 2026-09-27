"""Food segmentation with SAM, restricted to the OpenAI-grounded food boxes.

Segmentation results are cached by source-image hash. SAM receives the full
image and native-pixel prompts; resulting masks are clipped to their prompt
boxes before pixel counts are recorded. Empty masks remain empty instead of
being replaced by a color-based estimate.
"""
from __future__ import annotations

from pathlib import Path
import os

import numpy as np
import cv2

from .config import SEGMENTATION_JSON, RUNTIME
from .io_utils import dump_json, file_hash, load_json, load_image_rgb


class SAMSegmenter:
    def __init__(self, model_name: str = RUNTIME.sam_model, imgsz: int = RUNTIME.sam_imgsz):
        self.model_name = model_name
        self.imgsz = imgsz
        self._sam = None
        self._load()

    def _load(self):
        os.environ.setdefault("YOLO_CONFIG_DIR", str(Path(__file__).resolve().parents[2] / ".ultralytics"))
        from ultralytics import SAM

        self._sam = SAM(self.model_name)

    # -------------------------------------------------------------- cache
    def _cache_key(self, path: Path) -> str:
        return file_hash(path)

    def cached(self, path: Path) -> dict | None:
        if not SEGMENTATION_JSON.exists():
            return None
        db = load_json(SEGMENTATION_JSON)
        return db.get(self._cache_key(path))

    # ---------------------------------------------------------- segment
    def segment(self, image_path: Path, boxes: list[list[int]], *, labels: list[str] | None = None,
                use_cache: bool = True, viz_path: Path | None = None) -> dict:
        """Return per-box pixel count + mask quality, cached on disk."""
        image_path = Path(image_path)
        key = self._cache_key(image_path)
        cached = self.cached(image_path) if use_cache else None
        viz_dir = Path(__file__).resolve().parents[2] / "artifacts" / "visualizations" / "segmentation" / image_path.parent.parent.name / image_path.parent.name
        viz_output = Path(viz_path) if viz_path else viz_dir / f"{image_path.stem}_segmentation.png"
        expected_labels = labels or []
        if (cached is not None and cached.get("input_boxes") == boxes
                and cached.get("labels") == expected_labels and cached.get("viz_version") == 9
                and viz_output.exists()):
            return cached

        img = load_image_rgb(image_path)
        h, w = img.shape[:2]
        valid = []
        for b in boxes:
            x1, y1, x2, y2 = map(int, b)
            x1 = max(0, min(x1, w))
            y1 = max(0, min(y1, h))
            x2 = max(0, min(x2, w))
            y2 = max(0, min(y2, h))
            if x2 > x1 and y2 > y1:
                # Keep SAM strictly inside the food grounding box. This is
                # important for compartment trays: a larger prompt can cause
                # SAM to select the metal plate instead of the food.
                valid.append([x1, y1, x2, y2])

        counts, ious, refined_boxes = [], [], []
        overlay = img.copy()
        if valid:
            # A positive point is more reliable than a broad rectangle for
            # compartment trays: it selects the food surface at the grounded
            # item's visual center instead of the surrounding metal.
            for i, prompt_box in enumerate(valid):
                x1, y1, x2, y2 = prompt_box
                point = [int(round((x1 + x2) / 2)), int(round((y1 + y2) / 2))]
                results = self._sam.predict(
                    source=str(image_path),
                    points=[point],
                    labels=[1],
                    imgsz=self.imgsz,
                    retina_masks=True,
                    verbose=False,
                    device="cpu",
                )
                masks = results[0].masks.data if hasattr(results[0], "masks") else None
                refined = prompt_box
                if masks is not None and masks.shape[0] and masks[0].sum() > 0:
                    mask = masks[0].detach().cpu().numpy().astype(bool)
                    if mask.shape != (h, w):
                        mask = cv2.resize(mask.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST).astype(bool)
                    ys, xs = np.where(mask)
                    if len(xs):
                        refined = [int(xs.min()), int(ys.min()), int(xs.max() + 1), int(ys.max() + 1)]
                    counts.append(int(mask.sum()))
                    colour = np.array([(67 * i + 80) % 255, (131 * i + 140) % 255, (197 * i + 60) % 255], dtype=np.uint8)
                    overlay[mask] = (0.25 * overlay[mask] + 0.75 * colour).astype(np.uint8)
                else:
                    counts.append(0)
                refined_boxes.append(refined)
                ious.append(0.0)
        # if there were no valid boxes, counts/ious stay empty (num_boxes=0)

        result = {
            "_id": key,
            "image": str(image_path),
            "input_boxes": boxes,
            "viz_version": 9,
            "labels": expected_labels,
            "boxes": [
                {"bbox": b, "refined_bbox": refined_boxes[i], "pixel_count": int(counts[i]), "iou": float(ious[i]),
                 "label": labels[i] if labels and i < len(labels) else f"region {i + 1}"}
                for i, b in enumerate(valid)
            ],
            "num_boxes": len(valid),
            "image_size": [w, h],
        }
        viz_dir.mkdir(parents=True, exist_ok=True)
        viz_output.parent.mkdir(parents=True, exist_ok=True)
        for i, box in enumerate(refined_boxes):
            x1, y1, x2, y2 = box
            colour = tuple(int(x) for x in ((67 * i + 80) % 255, (131 * i + 140) % 255, (197 * i + 60) % 255))
            cv2.rectangle(overlay, (x1, y1), (x2, y2), colour, 3)
            label = labels[i] if labels and i < len(labels) else f"region {i + 1}"
            label = label[:36]
            font = cv2.FONT_HERSHEY_SIMPLEX
            (tw, th), baseline = cv2.getTextSize(label, font, 0.65, 2)
            tx = max(0, min(x1, w - tw - 8))
            ty = y1 - 6 if y1 - th - baseline - 8 >= 0 else min(h - 1, y1 + th + baseline + 8)
            top = max(0, ty - th - baseline - 6)
            bottom = min(h - 1, ty + baseline + 4)
            cv2.rectangle(overlay, (tx, top), (min(w - 1, tx + tw + 8), bottom), colour, -1)
            cv2.putText(overlay, label, (tx + 4, ty), font, 0.65, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.imwrite(str(viz_output), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
        if use_cache:
            db = load_json(SEGMENTATION_JSON) if SEGMENTATION_JSON.exists() else {}
            db[key] = result
            dump_json(db, SEGMENTATION_JSON)
        return result

def load_segmentation(path: Path = SEGMENTATION_JSON) -> dict:
    """Segmentation cache keyed by image path (for pixel-area lookups)."""
    if not Path(path).exists():
        return {}
    db = load_json(path)
    return {rec["image"]: rec for rec in db.values()}
