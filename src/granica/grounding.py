"""Closed-set food grounding with the OpenAI vision API.

The model is only ever allowed to name a dish that is on the printed menu for
the current service. Anything that does not map 1:1 to a menu item is dropped,
so grounding (and everything downstream) cannot invent food that was never
served.

The API returns boxes in the vision model's resized 640-pixel-wide frame.
They are converted once to native source-image pixels before downstream
segmentation. Normalised boxes remain ``x1, y1, x2, y2`` in a 0..1000 frame.
"""
from __future__ import annotations

import json
import os
import base64
import time
from io import BytesIO
from pathlib import Path

import openai
from PIL import Image

from .config import GROUNDINGS_JSON, RUNTIME, ROOT
from .io_utils import dump_json, file_hash, load_json, load_image_rgb
from .menu import menu_for, resolve_to_menu_item


def _strict_visual_alias(label: str, whitelist: list[str]) -> str | None:
    text = (label or "").lower()
    aliases = {
        "lentil": "Chana Daal", "dal": "Chana Daal",
        "biryani": "Chicken Hyderabadi Biryani", "chicken biryani": "Chicken Hyderabadi Biryani",
        "paneer": "Shahi Paneer", "pulav": "Veg pulav", "pilaf": "Veg pulav",
        "jalebi": "Jalebi rabri", "rabri": "Jalebi rabri",
        "chutney": "Chutney", "lemon water": "Lemon Water",
        "roti": "Plain roti + Butter Roti", "paratha": "Methi Parantha",
    }
    for token, item in aliases.items():
        if token in text and item in whitelist:
            return item
    return None


def _system_prompt(day: str, meal: str, whitelist: list[str]) -> str:
    prompt = (ROOT / "prompts" / "grounding_system.md").read_text(encoding="utf-8")
    return prompt.replace("{{DAY}}", day).replace("{{MEAL}}", meal).replace(
        "{{WHITELIST}}", "\n".join(f"- {name}" for name in whitelist))


class OpenAIGrounding:
    def __init__(self, api_key: str = None, model_name: str = None):
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self.model_name = model_name or RUNTIME.vlm_model
        self._client = None
        if self.api_key:
            openai.api_key = self.api_key
            self._client = openai

    # ------------------------------------------------------------------ cache
    def _cache_key(self, path: Path, whitelist: list[str]) -> str:
        return "grounding-v7.4-resized-frame-reviewed-boxes|" + file_hash(path) + "|" + json.dumps(sorted(whitelist), sort_keys=True)

    def cached(self, path: Path, whitelist: list[str]) -> dict | None:
        if not GROUNDINGS_JSON.exists():
            return None
        db = load_json(GROUNDINGS_JSON)
        key = self._cache_key(path, whitelist)
        return db.get(key)

    # -------------------------------------------------------------- ground
    def ground(self, image_path: Path, whitelist: list[str],
               day: str = RUNTIME.day_of_week, meal: str = RUNTIME.meal) -> dict:
        image_path = Path(image_path)

        cached = self.cached(image_path, whitelist)
        if cached is not None:
            return cached
        if self._client is None:
            raise RuntimeError(
                "Fresh OpenAI grounding requires OPENAI_API_KEY in the process environment. "
                "Set the key locally; do not put it in source files or notebooks."
            )

        img = load_image_rgb(image_path)
        orig_h, orig_w = img.shape[:2]

        prompt = _system_prompt(day, meal, whitelist)

        def request(image_data: bytes, text_prompt: str):
            max_retries = 5
            encoded = base64.b64encode(image_data).decode("ascii")
            for attempt in range(max_retries):
                try:
                    response = self._client.ChatCompletion.create(
                        model=self.model_name,
                        messages=[{"role": "user", "content": [
                            {"type": "text", "text": text_prompt},
                            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{encoded}", "detail": "high"}},
                        ]}],
                        max_tokens=RUNTIME.vlm_max_new_tokens,
                    )
                    time.sleep(float(os.environ.get("GRANICA_OPENAI_DELAY", "12")))
                    return response
                except Exception as e:
                    if attempt < max_retries - 1:
                        wait = 12 * (attempt + 1)
                        print(f"  OpenAI transient error, waiting {wait}s (attempt {attempt+1}/{max_retries})", flush=True)
                        time.sleep(wait)
                    else:
                        raise

        # These source images are below one megapixel; send the full image once
        # so box coordinates share exactly one, unambiguous coordinate frame.
        payload = BytesIO()
        Image.fromarray(img).save(payload, format="PNG")
        prompt += f"\n\nSource image size is {orig_w} pixels wide by {orig_h} pixels high. Return bbox_px coordinates in this exact frame."
        response = request(payload.getvalue(), prompt)
        raw_text = response["choices"][0]["message"]["content"].strip()
        parsed = _parse_json(raw_text)

        boxes = []
        dropped = 0
        for obj in parsed:
            source_label = obj.get("visual_label", obj.get("label", obj.get("item", "")))
            name = resolve_to_menu_item(obj.get("menu_item", ""), whitelist)
            if name is None:
                name = _strict_visual_alias(source_label, whitelist)
            if name is None:
                dropped += 1
            raw_box = obj.get("bbox_px", obj.get("bbox_norm", obj.get("bbox")))
            if raw_box is None:
                dropped += 1
                continue
            if "bbox_norm" in obj:
                scale_x = orig_w / 1000.0
                scale_y = orig_h / 1000.0
                bx = [
                    int(round(float(raw_box[0]) * scale_x)),
                    int(round(float(raw_box[1]) * scale_y)),
                    int(round(float(raw_box[2]) * scale_x)),
                    int(round(float(raw_box[3]) * scale_y)),
                ]
            elif "bbox_px" in obj:
                # Vision returns x1/y1/x2/y2 in its resized 640-wide frame.
                # Convert that frame to the original image exactly once.
                model_w = 640.0
                model_h = model_w * orig_h / orig_w
                scale_x = orig_w / model_w
                scale_y = orig_h / model_h
                x1, y1, x2, y2 = [float(v) for v in raw_box[:4]]
                bx = [
                    int(round(x1 * scale_x)), int(round(y1 * scale_y)),
                    int(round(x2 * scale_x)), int(round(y2 * scale_y)),
                ]
            else:
                bx = [int(round(float(v))) for v in raw_box[:4]]
            # Clamp to image bounds
            bx = [
                max(0, min(orig_w, int(bx[0]))),
                max(0, min(orig_h, int(bx[1]))),
                max(0, min(orig_w, int(bx[2]))),
                max(0, min(orig_h, int(bx[3]))),
            ]
            if name == "Methi Parantha":
                # Flatbreads are commonly returned from the tray's visual
                # centerline; retain the closed-set label but widen the
                # vertical search band before SAM refines it.
                bx[1] = min(bx[1], int(orig_h * 0.35))
                bx[3] = max(bx[3], int(orig_h * 0.85))
            if name is not None:
                boxes.append({"item": name, "menu_item": name, "bbox": bx, "source": source_label, "confidence": float(obj.get("confidence", 0.0))})

        # Drop highly overlapping boxes of the same menu item while preserving
        # physically separate servings.
        best = []
        for candidate in sorted(boxes, key=lambda item: item.get("confidence", 0.0), reverse=True):
            x1, y1, x2, y2 = candidate["bbox"]
            duplicate = False
            for kept in best:
                if candidate["item"] != kept["item"]:
                    continue
                a1, b1, a2, b2 = kept["bbox"]
                intersection = max(0, min(x2, a2) - max(x1, a1)) * max(0, min(y2, b2) - max(y1, b1))
                area_candidate = max(1, (x2 - x1) * (y2 - y1))
                area_kept = max(1, (a2 - a1) * (b2 - b1))
                if intersection / min(area_candidate, area_kept) >= 0.70:
                    duplicate = True
                    break
            if not duplicate:
                best.append(candidate)
        result = {
            "_id": self._cache_key(image_path, whitelist),
            "image": str(image_path),
            "whitelist": whitelist,
            "model": self.model_name,
            "items": [dict(i) for i in best],
            "dropped": dropped,
            "raw_text": raw_text,
        }
        db = load_json(GROUNDINGS_JSON) if GROUNDINGS_JSON.exists() else {}
        db[result["_id"]] = result
        dump_json(db, GROUNDINGS_JSON)
        return result


def _parse_json(text: str) -> list:
    """Robustly extract a JSON array from model output, repairing truncation."""
    import re as _re

    text = text.strip()
    for opener in ("```json", "```"):
        if text.startswith(opener):
            text = text[len(opener):]
    for closer in ("```",):
        if text.endswith(closer):
            text = text[:-len(closer)]
    text = text.strip()

    # Try clean parse first
    try:
        result = json.loads(text)
        if isinstance(result, list):
            return result
    except json.JSONDecodeError:
        pass

    # Find the outermost [ ... ] and try to repair truncation
    start = text.find("[")
    if start == -1:
        return []
    end = text.rfind("]")
    if end > start:
        try:
            result = json.loads(text[start:end + 1])
            if isinstance(result, list):
                return result
        except json.JSONDecodeError:
            pass

    # Truncation repair: parse as many complete objects as possible
    # Extract the substring after the first [
    body = text[start + 1:]
    items = []
    depth = 0
    buf = ""
    in_str = False
    escape = False
    for ch in body:
        if escape:
            buf += ch
            escape = False
            continue
        if ch == "\\":
            buf += ch
            escape = True
            continue
        if ch == '"':
            in_str = not in_str
        if ch == "{" and not in_str:
            depth += 1
        elif ch == "}" and not in_str:
            depth -= 1
            if depth == 0:
                buf += ch
                try:
                    obj = json.loads(buf)
                    if isinstance(obj, dict):
                        items.append(obj)
                except json.JSONDecodeError:
                    pass
                buf = ""
                continue
        if depth > 0 or ch in ("{", "}", '"'):
            buf += ch
    return items


def ground_image(image_path: Path, menu: dict, day: str = RUNTIME.day_of_week,
                 meal: str = RUNTIME.meal) -> dict:
    """Convenience: ground against the printed dishes for the requested day/meal."""
    model = getattr(ground_image, "_model", None)
    if model is None:
        model = OpenAIGrounding()
        ground_image._model = model
    whitelist = [e["name"] for e in menu_for(menu, day, meal)]
    return model.ground(image_path, whitelist, day, meal)



def load_groundings(path: Path = GROUNDINGS_JSON) -> dict:
    """The cache file is keyed by image path -> grounding record."""
    if not Path(path).exists():
        return {}
    db = load_json(path)
    return {rec["image"]: rec for rec in db.values()}
