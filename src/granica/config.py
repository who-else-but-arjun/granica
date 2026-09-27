"""Global configuration, paths and runtime switches for the Granica pipeline."""
from __future__ import annotations

import os
import json
import random
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PKG_DIR = Path(__file__).resolve().parent
ROOT = PKG_DIR.parent.parent  # .../granica

DATASET_DIR = ROOT / "dataset"
MENU_PDF = DATASET_DIR / "Barak_september_Menu.pdf"
DB_DIR = DATASET_DIR / "db"
DATASET_CONFIG_JSON = DATASET_DIR / "config.json"
_dataset_config = json.loads(DATASET_CONFIG_JSON.read_text(encoding="utf-8")) if DATASET_CONFIG_JSON.exists() else {}
_split_directories = _dataset_config.get("split_directories", {})
DINNER_TRAIN_DIR = DATASET_DIR / _split_directories.get("train", "dinner-2026-09-25-train")
DINNER_TEST_DIR = DATASET_DIR / _split_directories.get("test", "dinner-2026-09-25-test")

ARTIFACTS_DIR = ROOT / "artifacts"
DATA_DIR = ARTIFACTS_DIR / "data"

MENU_CONFIG_JSON = DATASET_DIR / "menu_config.json"
STUDENT_HISTORY_JSON = DATASET_DIR / "student_history.json"
TRAIN_WEIGHTS_JSON = DATASET_DIR / "train_item_weights.json"
GROUNDINGS_JSON = DATA_DIR / "groundings.json"
SYNTHETIC_WEIGHTS_JSON = DATA_DIR / "synthetic_weights.json"
SEGMENTATION_JSON = DATA_DIR / "segmentation.json"
CONSUMPTION_JSON = DATA_DIR / "consumption.json"
PEOPLE_JSON = DATA_DIR / "people.json"

FACE_ARTIFACTS_DIR = ARTIFACTS_DIR / "faces"
WEIGHT_ARTIFACTS_DIR = ARTIFACTS_DIR / "weights"
RECOMMENDATION_ARTIFACTS_DIR = ARTIFACTS_DIR / "recommendations"
FACE_DB_NPZ = FACE_ARTIFACTS_DIR / "face_gallery.npz"
FACE_METRICS_JSON = FACE_ARTIFACTS_DIR / "face_metrics.json"
WEIGHT_MODELS_JSON = WEIGHT_ARTIFACTS_DIR / "weight_models.json"
FIT_REPORT_JSON = WEIGHT_ARTIFACTS_DIR / "fit_report.json"
RECOMMENDATIONS_JSON = RECOMMENDATION_ARTIFACTS_DIR / "recommendations.json"

for _d in (DATA_DIR, ARTIFACTS_DIR, FACE_ARTIFACTS_DIR, WEIGHT_ARTIFACTS_DIR, RECOMMENDATION_ARTIFACTS_DIR):
    _d.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
def set_seed(seed: int = 20260925) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.set_num_threads(max(1, torch.get_num_threads()))
        torch.use_deterministic_algorithms(False)
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Runtime
# ---------------------------------------------------------------------------
@dataclass
class Runtime:
    """Everything the two notebooks need to agree on for one run."""

    # The dataset folder is dated 2026-09-25 (a Friday).
    service_date: str = "2026-09-25"
    day_of_week: str = "FRIDAY"
    meal: str = "DINNER"

    # VLM grounding (OpenAI GPT-4o Vision)
    vlm_model: str = "gpt-4o"
    vlm_max_pixels: int = 1000 * 1000
    vlm_min_pixels: int = 28 * 28 * 128
    vlm_max_new_tokens: int = 512
    vlm_temperature: float = 0.0

    # LLM recommendation (OpenAI GPT-4o)
    llm_model: str = "gpt-4o"
    llm_max_new_tokens: int = 1400
    llm_temperature: float = 0.3

    # Face identification
    face_model: str = "buffalo_l"
    face_det_size: tuple = (640, 640)
    face_threshold: float | None = None  # None -> use the calibrated EER threshold

    # Segmentation
    sam_model: str = str(ROOT / "notebooks" / "mobile_sam.pt")
    sam_imgsz: int = 1024

    # Synthetic ground-truth generation
    synthetic_seed: int = 20260925
    synthetic_noise_sigma: float = 0.18  # per-portion multiplicative lognormal noise

    # Minimum portions before a per-category weight model is trusted.
    min_points_per_category: int = 4

    device: str = field(default_factory=lambda: "cpu")

    def torch_dtype(self):
        import torch

        if self.device == "cuda":
            return torch.bfloat16
        return torch.float32

    def dinner_roots(self) -> dict:
        """Resolve only the existing, explicitly configured image splits."""
        return {"train": DINNER_TRAIN_DIR, "test": DINNER_TEST_DIR}

    @property
    def db_dir(self) -> Path:
        return DB_DIR

    def plate_images(self, split: str, stage: str) -> list:
        """All plate photos for one split and one stage ('pre' | 'post' | 'second-serve')."""
        root = self.dinner_roots()[split]
        folder = root / stage
        if not folder.is_dir():
            return []
        exts = {".png", ".jpg", ".jpeg", ".JPG", ".PNG"}
        return sorted(p for p in folder.iterdir() if p.suffix in exts)

    def people_ids(self) -> list:
        ids = set()
        for split, root in self.dinner_roots().items():
            for stage in ("pre", "post", "second-serve"):
                for p in self.plate_images(split, stage):
                    ids.add(p.stem)
        return sorted(ids)


RUNTIME = Runtime()
RUNTIME.openai_api_key = os.environ.get("OPENAI_API_KEY", "")


def offline_mode() -> bool:
    """Set GRANICA_OFFLINE=1 to forbid any network call (models must be cached)."""
    return os.environ.get("GRANICA_OFFLINE", "0") == "1"
