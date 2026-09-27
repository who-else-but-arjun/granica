"""Small I/O helpers shared by the notebooks and pipeline modules."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np


def load_json(path: str | Path) -> dict | list:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def dump_json(obj, path: str | Path, indent: int = 2) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, indent=indent, ensure_ascii=False, default=_default), encoding="utf-8")
    return p


def _default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.ndarray,)):
        return o.tolist()
    if isinstance(o, (np.bool_,)):
        return bool(o)
    raise TypeError(f"not serialisable: {type(o)}")


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def file_hash(path: str | Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def load_image_rgb(path: str | Path) -> np.ndarray:
    """Load an image as an HxWx3 contiguous uint8 RGB array (OpenCV backend)."""
    import cv2

    img = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"could not read image: {path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
