"""Identity resolution via InsightFace (face recognition from the db gallery).

* db images are named ``<name>-<roll>-<split>.jpeg`` (train/test enrollment pairs).
* A person's gallery = mean embedding over their db images.
* Identification of a plate photo is done by nearest-neighbour cosine similarity
  to the gallery, with the person's own name also stitched back from the photo
  metadata (filename stem) so accuracy metrics can be computed directly.
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np

from .config import FACE_DB_NPZ, FACE_METRICS_JSON, RUNTIME
from .io_utils import dump_json, ensure_dir, load_image_rgb

# <name>-<roll>-<split>
_DBID_RE = re.compile(r"^(.+)-(\d{6,})-(train|test|dev)$", re.IGNORECASE)


def parse_db_filename(stem: str) -> tuple[str, str, str]:
    m = _DBID_RE.match(stem)
    if not m:
        return stem, "unknown", "train"
    return m.group(1), m.group(2), m.group(3)


def parse_plate_filename(stem: str) -> tuple[str, str]:
    """Plate photo stem is ``<name>-<roll>`` (no train/test suffix)."""
    m = re.match(r"^(.+)-(\d{6,})$", stem)
    if not m:
        return stem, "unknown"
    return m.group(1), m.group(2)


class FaceIdentifier:
    def __init__(self, model_name: str = RUNTIME.face_model,
                 det_size: tuple = RUNTIME.face_det_size):
        self.model_name = model_name
        self.det_size = det_size
        self.app = self._load_app()
        self.persons: dict[str, dict] = {}
        self.embeddings: np.ndarray | None = None
        self.names: list[str] = []

    def _load_app(self):
        from insightface.app import FaceAnalysis

        app = FaceAnalysis(
            name=self.model_name,
            root=str(Path.home() / ".insightface"),
            providers=["CPUExecutionProvider"],
        )
        app.prepare(ctx_id=-1, det_size=self.det_size)  # CPU
        return app

    # ------------------------------------------------------------------ build
    def build_gallery(self, db_dir: Path, verbose: bool = True) -> dict:
        rows = []
        for path in sorted(Path(db_dir).glob("*.*")):
            if path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
                continue
            if parse_db_filename(path.stem)[2].lower() != "train":
                continue
            img = load_image_rgb(path)
            faces = self.app.get(img)
            if not faces:
                if verbose:
                    print(f"  no face: {path.name}")
                continue
            best = max(faces, key=lambda f: f.bbox[2] * f.bbox[3])  # largest face
            name, roll, split = parse_db_filename(path.stem)
            rows.append({
                "path": str(path), "name": name, "roll": roll,
                "split": split, "embedding": best.embedding.astype(np.float32),
                "bbox": list(best.bbox),
                "det_score": float(best.det_score),
                "gender": str(best.gender),
                "age": int(best.age) if best.age else None,
            })

        # gallery = mean embedding per person
        by_person: dict[str, list[dict]] = {}
        for r in rows:
            by_person.setdefault(r["name"], []).append(r)

        embeddings, names = [], []
        persons = {}
        for name, rs in by_person.items():
            embs = np.stack([r["embedding"] for r in rs])
            g = embs.mean(axis=0)
            g = g / (np.linalg.norm(g) + 1e-12)
            embeddings.append(g)
            names.append(name)
            persons[name] = {
                "name": name,
                "roll": rs[0]["roll"],
                "enrollments": [r["path"] for r in rs],
                "gallery_embedding": g.tolist(),
            }

        self.embeddings = np.stack(embeddings).astype(np.float32)
        self.names = names
        self.persons = persons
        ensure_dir(FACE_DB_NPZ.parent)
        np.savez(FACE_DB_NPZ,
                 gallery=self.embeddings,
                 names=np.array(self.names, dtype=object),
                 persons=np.array(list(persons.values()), dtype=object),
                 gallery_version=np.array([2], dtype=np.int32))
        if verbose:
            print(f"gallery: {len(self.names)} persons, {len(rows)} faces")
        return {"persons": len(self.names), "faces": len(rows)}

    # -------------------------------------------------------------- identify
    def load_gallery(self) -> bool:
        if not FACE_DB_NPZ.exists():
            return False
        d = np.load(FACE_DB_NPZ, allow_pickle=True)
        if int(d.get("gallery_version", np.array([0]))[0]) != 2:
            return False
        self.embeddings = d["gallery"]
        self.names = list(d["names"])
        self.persons = {n: {} for n in self.names}
        return True

    def identify(self, image_path, top_k: int = 3) -> dict:
        img = load_image_rgb(image_path)
        faces = self.app.get(img)
        if not faces:
            return {"path": str(image_path), "ok": False, "error": "no face detected"}
        best = max(faces, key=lambda f: f.bbox[2] * f.bbox[3])
        emb = best.embedding.astype(np.float32)
        emb = emb / (np.linalg.norm(emb) + 1e-12)
        sims = (self.embeddings @ emb).astype(np.float32)
        order = np.argsort(sims)[::-1][:top_k]
        results = []
        for idx in order:
            results.append({"name": self.names[idx], "score": float(sims[idx])})
        best_idx = int(order[0])
        return {
            "path": str(image_path),
            "ok": True,
            "box": list(best.bbox),
            "best": results[0]["name"],
            "score": float(sims[best_idx]),
            "topk": results,
            "gallery_size": len(self.names),
            "threshold": float(RUNTIME.face_threshold or 0.45),
        }

    # ------------------------------------------------------------- metrics
    def evaluate(self, db_dir, verbose: bool = True) -> dict:
        """Verification (every face vs every face) + rank-1 identification
        against the per-person gallery built during ``build_gallery``."""
        # ---- verification: every detected face embedding vs every other ----
        enroll_embs, enroll_labels = [], []
        for path in sorted(Path(db_dir).glob("*.*")):
            if path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
                continue
            name, _, _ = parse_db_filename(path.stem)
            e = self._embed(load_image_rgb(path))
            if e is not None:
                enroll_embs.append(e)
                enroll_labels.append(name)
        enroll_embs = np.stack(enroll_embs).astype(np.float32) if enroll_embs else np.zeros((0, 0))
        sim_matrix = enroll_embs @ enroll_embs.T
        nn = len(enroll_embs)
        same, diff = [], []
        for i in range(nn):
            for j in range(i + 1, nn):
                s = float(sim_matrix[i, j])
                if enroll_labels[i] == enroll_labels[j]:
                    same.append(s)
                else:
                    diff.append(s)

        # ---- identification: gallery = per-person mean; LOO test ----------
        G = self.embeddings.astype(np.float32)  # (n_persons, dim)
        n = len(self.names)
        correct, total = 0, 0
        per_image_accs = []
        for path in sorted(Path(db_dir).glob("*.*")):
            if path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
                continue
            if parse_db_filename(path.stem)[2].lower() != "test":
                continue
            name, _, _ = parse_db_filename(path.stem)
            emb_hold = self._embed(load_image_rgb(path))
            if emb_hold is None:
                continue
            sims = G @ emb_hold
            rank1 = int(np.argmax(sims))
            total += 1
            ok = self.names[rank1] == name
            correct += int(ok)
            per_image_accs.append(ok)

        eer_thr, eer = self._eer_threshold(np.array(same), np.array(diff))

        report = {
            "gallery_size": n,
            "gallery_persons": list(self.names),
            "rank1_accuracy_mean": correct / total if total else 0.0,
            "rank1_accuracy_median": float(np.median(per_image_accs)) if per_image_accs else 0.0,
            "rank1_total": total,
            "verification_same_mean": float(np.mean(same)) if same else 0.0,
            "verification_same_std": float(np.std(same)) if same else 0.0,
            "verification_diff_mean": float(np.mean(diff)) if diff else 0.0,
            "verification_diff_std": float(np.std(diff)) if diff else 0.0,
            "eer_threshold": eer_thr,
            "eer": eer,
            "roc": self._roc_points(np.array(same), np.array(diff)),
            "cmc_rank1": float(correct / total) if total else 0.0,
        }
        dump_json(report, FACE_METRICS_JSON)
        if verbose:
            print(f"[face] rank-1 = {report['rank1_accuracy_mean']:.3f} (n={total})")
            print(f"[face] gallery sims  same={report['verification_same_mean']:.3f} "
                  f"diff={report['verification_diff_mean']:.3f}  EER={report['eer']:.3f}"
                  f" @ thr={report['eer_threshold']:.3f}")
        return report

    # -------------------------------------------------------------- helpers
    def _embed(self, img) -> np.ndarray | None:
        faces = self.app.get(img)
        if not faces:
            return None
        best = max(faces, key=lambda f: f.bbox[2] * f.bbox[3])
        e = best.embedding.astype(np.float32)
        return e / (np.linalg.norm(e) + 1e-12)

    @staticmethod
    def _eer_threshold(same: np.ndarray, diff: np.ndarray) -> tuple[float, float]:
        if same.size == 0 or diff.size == 0:
            return 0.5, 1.0
        grid = np.unique(np.concatenate([same, diff]))
        best_thr, best_err = 0.5, 1.0
        for thr in grid:
            fa = (diff >= thr).mean()  # impostor accepted
            fr = (same < thr).mean()  # genuine rejected
            err = (fa + fr) / 2
            if err < best_err:
                best_err, best_thr = err, float(thr)
        return best_thr, float(best_err)

    @staticmethod
    def _roc_points(same: np.ndarray, diff: np.ndarray) -> list[dict]:
        if same.size == 0 or diff.size == 0:
            return []
        out = []
        for far_target in (1e-3, 1e-2, 1e-1, 0.5):
            thr = float(np.quantile(diff, 1 - far_target)) if len(diff) else 0.5
            tpr = float((same >= thr).mean())
            out.append({"far": far_target, "threshold": thr, "tpr": tpr})
        return out
