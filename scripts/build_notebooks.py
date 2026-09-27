"""Build the two pipeline notebooks. Run: python scripts/build_notebooks.py"""
from __future__ import annotations

import sys
from pathlib import Path

import nbformat as nbf

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

ROOT = Path(__file__).resolve().parent.parent
NB_DIR = ROOT / "notebooks"
NB_DIR.mkdir(exist_ok=True)


def _new():
    nb = nbf.v4.new_notebook()
    nb.metadata["kernelspec"] = {"display_name": "Python 3", "language": "python", "name": "python3"}
    nb.metadata["language_info"] = {"name": "python", "version": "3.11"}
    return nb


def md(text):
    return nbf.v4.new_markdown_cell(text)


def code(text):
    return nbf.v4.new_code_cell(text)


def write_nb(name, cells):
    nb = _new()
    # Generated notebooks should open cleanly and never retain stale results.
    for cell in cells:
        if cell.cell_type == "code":
            cell.execution_count = None
            cell.outputs = []
    nb.cells = cells
    path = NB_DIR / name
    nbf.write(nb, str(path))
    print("wrote", path, f"({len(cells)} cells)")


# ========================== NOTEBOOK 1 - TRAINING =========================
TRAIN = [
    md("""# Savor - Food-intake training pipeline

Estimates how much food each student actually ate from **before/after dinner
plate photos**, then fits a per-category `grams = a·pixels + b` model.

**Data**
- `dataset/dinner-2026-09-25-{train,test}/{pre|post}/*.png` - plate photos (before/after).
- `dataset/db/<name>-<roll>-<split>.jpeg` - face enrollment photos.
- `dataset/menu_config.json` and `dataset/student_history.json` - closed menu and mock taken/wasted history.
- `dataset/train_item_weights.json` - train-only item weights generated from available scale anchors.
- the plates sit on a digital scale - grams are readable in the photo where the display
  is not cropped.

**Closed-set rule:** OpenAI vision grounding and recommendations may only name
dishes on the printed menu; invalid names and units are rejected or repaired.

**Hardware** runs on CPU (no CUDA). OpenAI vision is called once per uncached image;
all results cache to `artifacts/data/`.
"""),

    code("""%load_ext autoreload
%autoreload 2
import sys
from pathlib import Path
ROOT = Path.cwd()
if not (ROOT / "src" / "granica").is_dir():
    ROOT = ROOT.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from granica.config import ROOT, RUNTIME, set_seed, TRAIN_WEIGHTS_JSON
from granica.menu import load_menu
set_seed()
print("service:", RUNTIME.day_of_week, RUNTIME.meal, "| on", RUNTIME.service_date)
print("device:", RUNTIME.device, "| vlm:", RUNTIME.vlm_model, "| llm:", RUNTIME.llm_model)
print("splits:", RUNTIME.people_ids())
"""),

    md("""## Training split - before and after photographs

These are every actual paired plate photo from the configured **train split**.
The separate held-out test split is used only in notebook 2."""),
    code("""from PIL import Image
from IPython.display import display
import matplotlib.pyplot as plt

train_pre = {p.stem: p for p in RUNTIME.plate_images("train", "pre")}
train_post = {p.stem: p for p in RUNTIME.plate_images("train", "post")}
train_ids = sorted(set(train_pre) | set(train_post))
fig, axes = plt.subplots(max(1, len(train_ids)), 2, figsize=(14, 4 * max(1, len(train_ids))), squeeze=False)
for row, person_id in enumerate(train_ids):
    for col, stage in enumerate(("pre", "post")):
        path = (train_pre if stage == "pre" else train_post).get(person_id)
        ax = axes[row][col]
        if path:
            ax.imshow(Image.open(path)); ax.set_title(f"TRAIN / {stage.upper()} / {person_id}")
        else:
            ax.text(0.5, 0.5, "no paired photo", ha="center", va="center")
        ax.axis("off")
plt.tight_layout(); display(fig); plt.close(fig)
print("training paired students:", len(train_ids), "| photos:", len(train_pre) + len(train_post))
"""),

    md("""## Step 1 - Menu config (closed set)

`dataset/menu_config.json` is a hand-transcribed copy of the printed menu. Each dish
is tagged with a **category** and a **countable unit -> grams** table. Only these
dish names may appear anywhere downstream."""),

    code("""menu = load_menu()
from granica.menu import menu_for, menu_items

dinner = menu_for(menu, RUNTIME.day_of_week, RUNTIME.meal)
print(f"{RUNTIME.day_of_week} {RUNTIME.meal}: {len(dinner)} dishes")
for e in dinner:
    print(f"  {e['name']:28s} {e['category']:13s} units={e['units']}")
print("\\nwhitelist:", menu_items(menu, RUNTIME.day_of_week, RUNTIME.meal))
"""),

    md("""## Step 2 - Face identification & DB retrieval

Build a gallery of the 9 students from `dataset/db/*` with InsightFace
`buffalo_l`, then measure:

1. **Verification** - genuine vs impostor cosine similarity + EER threshold.
2. **Rank-1 identification** - each enrollment face -> nearest gallery person.

Plate photos contain no faces, so a plate is attributed to its student via the
filename stem `<name>-<roll>`. `identify()` is demo'd on face photos; any future
ID photo behaves identically.
"""),

    code("""from granica.faceid import FaceIdentifier
from granica.io_utils import load_json

ident = FaceIdentifier()
metrics_path = ROOT / "artifacts/faces/face_metrics.json"
if not ident.load_gallery() or not metrics_path.exists():
    ident.build_gallery(RUNTIME.db_dir)
    rep = ident.evaluate(RUNTIME.db_dir)
else:
    rep = load_json(metrics_path)
    print("loaded cached train gallery and face metrics")
print("gallery persons:", len(ident.names))
for k in ["gallery_size", "rank1_accuracy_mean", "rank1_total",
          "verification_same_mean", "verification_diff_mean", "eer_threshold", "eer"]:
    print(f"  {k}: {rep[k]}")
"""),

    code("""from pathlib import Path
ident = FaceIdentifier()
ident.load_gallery()
sample = sorted(RUNTIME.db_dir.glob("*.jpeg"))[0]
res = ident.identify(sample)
print("face photo ->", res["best"], "score", round(res["score"], 3),
      "rank-1 candidates:", [r["name"] for r in res["topk"]])
from granica.faceid import parse_plate_filename, parse_db_filename
print("plate stem ->", parse_plate_filename("arjun-230102125"))
print("db stem   ->", parse_db_filename("arjun-230102125-train.jpeg"))
"""),

    md("""### Held-out face similarity against the train gallery

Each held-out test face is compared only with train enrollment images. The test
target is never included as a bar in its own comparison."""),
    code("""from visualize_outputs import _face_similarity
face_similarity_paths = _face_similarity()
for path in face_similarity_paths:
    display(Image.open(path))
print("test queries compared with train-only gallery:", len(face_similarity_paths))
"""),

    md("""## Step 3 - OpenAI vision grounding (closed set)

OpenAI vision inspects overlapping top and bottom crops of each wide source
photo. Crop boxes are mapped back to native full-image pixels and duplicates in
the overlap are removed. Every image stays restricted to the printed menu.
Results are cached to `artifacts/data/groundings.json`."""),

    code("""from granica.grounding import OpenAIGrounding, load_groundings

gmodel = OpenAIGrounding()
images = [p for stage in ("pre", "post") for p in RUNTIME.plate_images("train", stage)]
whitelist = menu_items(menu, RUNTIME.day_of_week, RUNTIME.meal)

dropped_total = 0
for p in images:
    dropped_total += gmodel.ground(p, whitelist)["dropped"]
print(f"grounded {len(images)} images; non-menu items dropped: {dropped_total}")

g = load_groundings()
k = str(images[0])
print("sample", Path(k).name, "->", [(b["item"], b["bbox"]) for b in g[k]["items"]])
"""),

    md("""## Step 4 - SAM segmentation inside each box

MobileSAM receives the full source image plus each corrected native-pixel food
box. Its mask is clipped to that box before pixels are counted; an empty mask is
recorded as zero, never replaced with a color-based guess. Each mask gets a
category label in the saved visualization."""),

    code("""from granica.segmentation import SAMSegmenter, load_segmentation
from granica.grounding import load_groundings

seg = SAMSegmenter()
g = load_groundings()
images = [p for stage in ("pre", "post") for p in RUNTIME.plate_images("train", stage)]
for p in images:
    rec = g.get(str(p), {})
    boxes = [it["bbox"] for it in rec.get("items", [])]
    if boxes:
        seg.segment(p, boxes, labels=[it["item"] for it in rec.get("items", [])])

seg_db = load_segmentation()
k = str(images[0])
print("segmentation for", Path(k).name)
for b in seg_db[k]["boxes"]:
    print(f"  pixels={b.get('pixel_count')} iou={b.get('iou', 0):.2f}")
"""),

    md("""### Training grounding and segmentation overlays

Each image below is generated from its paired training photo: menu labels appear
on the grounding boxes and on the opaque SAM mask overlay."""),
    code("""from visualize_outputs import _grounding_images
from granica.segmentation import load_segmentation
from granica.grounding import load_groundings

overlay_paths = _grounding_images(load_groundings(), load_segmentation())
train_overlays = [p for p in overlay_paths if Path(p).parent.parent.name == "dinner-2026-09-25-train"]
for path in train_overlays:
    display(Image.open(path))
print("inline train box+mask overlays:", len(train_overlays))
"""),

    md("""## Step 5 - Training weights and scale anchors

`scripts/analyze_images.py` keeps scale readings separately and creates
train-only item weights from plate totals. Per-item allocations are estimates,
not individually measured weights; the dataset config records this provenance.
"""),

    code("""import subprocess, sys
from pathlib import Path
from granica.io_utils import load_json

gt_path = ROOT / "artifacts/data/ground_truth_weights.json"
import os
env = os.environ.copy(); env["GRANICA_SPLITS"] = "train"
subprocess.run([sys.executable, str(ROOT / "scripts/analyze_images.py")], check=True, env=env)

gt = load_json(gt_path)
pl = gt["plates"]["train/pre/kunj-230103056"]
print("kunj train/pre total:", pl["total_grams"], "g (scale anchor:", pl["anchored"], ")")
for it in pl["items"]:
    print(f"  {it['item']:26s} {it['category']:11s} px={it['pixels']:5d}  g={it['grams']}")
print("\\nbaked plates:", len(gt["plates"]))
"""),

    md("""## Step 6 - Fit per-category pixel->weight models

Fit `grams = a*pixels + b` for every category present in the training plates.
Categories below `min_points_per_category` are pooled into a shared `__global__`
fallback so the model is never empty.
"""),

    code("""from granica.weights import fit_models

models = fit_models(gt)
print(f"{'category':14s} {'n':>3s} {'slope':>11s} {'intc':>9s} {'R2':>7s} {'MAE':>7s} used")
for c, m in models.items():
    print(f"{c:14s} {m['n']:3d} {m['slope']:11.6f} {m['intercept']:9.3f} {m['r2']:7.3f} {m['mae']:7.2f} {m['used']}")
"""),

    md("""### Per-category regression plots

These figures show the train points and the fitted pixel-to-grams line for each
food category. Per-item labels are scale-anchored plate allocations, so the plots
are training diagnostics rather than independent measured-item accuracy."""),
    code("""from visualize_outputs import _regressions
regression_paths = _regressions(gt, models)
for path in regression_paths:
    display(Image.open(path))
print("regression plots:", len(regression_paths))
"""),

md("""## Step 7 - Training fit check

This MSE is reported on the scale-anchored training examples used to fit the
line. It is a fit diagnostic, not held-out accuracy; the held-out test photos
and any readable scale totals are handled in notebook 2.
"""),

code("""points = [(it["pixels"], it["grams"], it["category"])
          for p in gt["plates"].values() if p["split"] == "train" and p.get("anchored") is not None
          for it in p["items"]]
errors = []
for pixels, grams, category in points:
    model = models.get(category, models.get("__global__"))
    errors.append((model["slope"] * pixels + model["intercept"] - grams) ** 2)
train_mse = sum(errors) / len(errors) if errors else float("nan")
print(f"scale-anchored train items: {len(points)}")
print(f"training fit MSE: {train_mse:.2f} g^2 (not test accuracy)")
"""),

    md("""## Step 8 - Consumption history & people store

Food eaten = pre_total - post_total. This notebook writes training-split
consumption; the inference notebook handles held-out test plates.
"""),

    code("""from granica.weights import build_consumption

people = build_consumption(gt, {}, out_path=ROOT / "artifacts/data/people.json")
kunj = people["people"]["kunj-230103056"]
print("kunj meals:", len(kunj["meals"]))
for m in kunj["meals"]:
    print(f"  {m['split']} {m['stage']:5s} total={m['total_grams']} consumed={m.get('consumed_grams')}")
"""),

    md("""## Summary

- `dataset/menu_config.json` - closed menu + unit->grams.
- `dataset/student_history.json` - mock meals with served, taken and wasted grams.
- `dataset/train_item_weights.json` - train-only per-item labels and provenance.
- `artifacts/data/groundings.json` - OpenAI vision labels (cached).
- `artifacts/data/segmentation.json` - SAM pixel counts (cached).
- `artifacts/data/ground_truth_weights.json` - train labels with scale readings kept separately.
- `artifacts/weights/weight_models.json` - fitted `grams = a*pixels + b` per category.
- `artifacts/data/people.json` - training-split intake calculation.

Open `notebooks/2_inference_pipeline.ipynb` for held-out test results and
several personalized recommendation examples."""),
]

# ========================== NOTEBOOK 2 - INFERENCE ========================
INFERENCE = [
    md("""# Savor - Inference: per-student recommendations

End-to-end: take **one** before-meal plate photo + a face photo, and produce a
personalised, **unit-counted** ("2 katori", "1 ladle", ...) dinner recommendation
tailored to that student's recent intake.

Uses only the existing held-out test split for inference examples. Grounding is
closed-menu, SAM masks are clipped to their native-image boxes, and the cached
validated LLM response is used for recommendation. Missing current-version test
groundings are requested once with the configured rate-limit buffer.
"""),
    code("""%load_ext autoreload
%autoreload 2
import sys
from pathlib import Path
ROOT = Path.cwd()
if not (ROOT / "src" / "granica").is_dir():
    ROOT = ROOT.parent
sys.path.insert(0, str(ROOT / "src"))

from granica.config import ROOT, RUNTIME
sys.path.insert(0, str(ROOT / "scripts"))
from granica.io_utils import load_json
from granica.menu import load_menu, menu_for, menu_items
from PIL import Image
from IPython.display import display
import matplotlib.pyplot as plt
menu = load_menu()
print("menu loaded:", RUNTIME.day_of_week, RUNTIME.meal, menu_items(menu, RUNTIME.day_of_week, RUNTIME.meal))
"""),

    md("""## Step 1 - Identity retrieval

Load the face gallery (built in notebook 1). Run it on a face photo (any
enrollment/ID photo). Plate photos have no face, so a plate is attributed via
its filename stem; for the live demo below we pass the face photo of the student.
Accuracy recap is printed from `artifacts/faces/face_metrics.json`."""),
    code("""from granica.faceid import FaceIdentifier, parse_plate_filename
from granica.io_utils import load_json

ident = FaceIdentifier(); ident.load_gallery()
print("gallery:", len(ident.names), "persons")

# accuracy recap from the training notebook
metrics = load_json(ROOT / "artifacts/faces/face_metrics.json")
print(f"rank-1 = {metrics['rank1_accuracy_mean']:.3f} | "
      f"same={metrics['verification_same_mean']:.3f} diff={metrics['verification_diff_mean']:.3f} "
      f"| EER={metrics['eer']:.3f}")

# identify a face photo
face_img = sorted(RUNTIME.db_dir.glob("*.jpeg"))[0]
res = ident.identify(face_img)
person = res["best"]
print("face photo ->", person, "score", round(res["score"], 3))

# identify a plate photo via filename convention
plate_img = RUNTIME.plate_images("test", "pre")[0]
person = parse_plate_filename(plate_img.stem)[0]   # name from filename
print("plate", plate_img.name, "-> student:", person)
"""),

    md("""### Test split - before/after photos

These are every actual paired photo from the configured **test split**. The
inference path uses only these held-out images; no external notebook images are added."""),
code("""test_pre = {p.stem: p for p in RUNTIME.plate_images("test", "pre")}
test_post = {p.stem: p for p in RUNTIME.plate_images("test", "post")}
test_ids = sorted(set(test_pre) | set(test_post))
fig, axes = plt.subplots(max(1, len(test_ids)), 2, figsize=(14, 4 * max(1, len(test_ids))), squeeze=False)
for row, person_id in enumerate(test_ids):
    for col, stage in enumerate(("pre", "post")):
        path = (test_pre if stage == "pre" else test_post).get(person_id)
        ax = axes[row][col]
        if path:
            ax.imshow(Image.open(path)); ax.set_title(f"TEST / {stage.upper()} / {person_id}")
        else:
            ax.text(0.5, 0.5, "no paired photo", ha="center", va="center")
        ax.axis("off")
plt.tight_layout(); display(fig); plt.close(fig)
"""),

    md("""## Step 2 - Ground the food on the plate

OpenAI vision labels visible foods and returns normalized full-image boxes,
restricted to the closed dinner whitelist. The image set is the held-out test split."""),
    code("""from granica.grounding import OpenAIGrounding
from granica.io_utils import load_json
gmodel = OpenAIGrounding()
test_images = [p for stage in ("pre", "post") for p in RUNTIME.plate_images("test", stage)]
whitelist = menu_items(menu, RUNTIME.day_of_week, RUNTIME.meal)
grounds = {}
for image_path in test_images:
    grounds[str(image_path)] = gmodel.ground(image_path, whitelist)
    print(image_path.name, [x["item"] for x in grounds[str(image_path)]["items"]])
g = grounds[str(plate_img)]
print("items on plate:", [b["item"] for b in g["items"]])
"""),

    md("""## Step 3 - Segment & predict weight with the fitted model

MobileSAM counts pixels inside each box; the per-category `grams = a*pixels + b`
models from notebook 1 turn those counts into grams.
"""),
code("""from granica.segmentation import SAMSegmenter
from granica.weights import predict_plate_items
models = load_json(ROOT / "artifacts/weights/weight_models.json")

seg = SAMSegmenter()
test_predictions = {}
test_segmentation = {}
for image_path in test_images:
    image_grounding = grounds[str(image_path)]
    image_segmentation = seg.segment(
        image_path, [b["bbox"] for b in image_grounding["items"]],
        labels=[b["item"] for b in image_grounding["items"]])
    test_segmentation[str(image_path)] = image_segmentation
    image_items = [{"item": b["item"], "pixels": sb.get("pixel_count", 0)}
                   for b, sb in zip(image_grounding["items"], image_segmentation["boxes"])]
    test_predictions[f"test/{image_path.parent.name}/{image_path.stem}"] = predict_plate_items(image_items, menu, models)
from granica.io_utils import dump_json
dump_json(test_predictions, ROOT / "artifacts/data/test_predictions.json")
seg_rec = test_segmentation[str(plate_img)]
items = [{"item": b["item"], "pixels": sb.get("pixel_count", 0)}
         for b, sb in zip(g["items"], seg_rec["boxes"])]

pred = predict_plate_items(items, menu, models)
print("predicted plate weight:", pred["total_grams"], "g")
for it in pred["predicted_items"]:
    print(f"  {it['item']:26s} {it['grams']}g")
viz_path = ROOT / "artifacts" / "visualizations" / "segmentation" / plate_img.parent.parent.name / plate_img.parent.name / f"{plate_img.stem}_segmentation.png"
display(Image.open(viz_path))
"""),

    md("""## Step 4 - Student history + recommendation examples

The selected student's prompt is shown for audit. Validated recommendations for
several students are loaded from the cache so notebook reruns avoid extra
recommendation API calls. Each recommendation is constrained to the menu and
serving-unit table.
"""),
    code("""from granica.recommend import recommend, build_recommendation_prompt, validate_and_repair
from granica.io_utils import load_json

people = load_json(ROOT / "dataset/student_history.json")
history = people["people"].get(person, {})

prompt = build_recommendation_prompt(menu, RUNTIME.day_of_week, RUNTIME.meal, history)
print("=== system + user prompt sent to OpenAI ===")
print(prompt[0]["content"][:400], "...\\n")
print(prompt[1]["content"][:400], "...")

all_recs = load_json(ROOT / "artifacts/recommendations/recommendations.json")["recommendations"]
rec = all_recs.get(person)
if rec is None:
    raise KeyError(f"No cached LLM recommendation for {person}; run scripts/run_recommendations.py")
print("\\n=== recommendation (validated) ===")
print("total:", rec["total_grams"], "g  within band:", rec["within_band"])
for r in rec["recommendations"]:
    print(f"  {r['item']:26s} {r['count']} {r['unit']} = {r['grams']}g")
example_people = [name for name, result in sorted(all_recs.items())
                  if isinstance(result, dict) and not result.get("error")][:3]
print("\\n=== additional validated examples ===")
for example in example_people:
    example_rec = all_recs[example]
    print(f"\\n{example}: {example_rec['total_grams']}g total")
    for line in example_rec.get("recommendations", []):
        print(f"  {line['item']}: {line['count']} {line['unit']} ({line['grams']}g)")
"""),

    md("""## Step 5 - Recommendation vs. past consumption

Compare the actual cached LLM serving suggestion with this student's average
amount taken across the historical meals."""),
    code("""from visualize_outputs import _recommendation_history
from granica.config import STUDENT_HISTORY_JSON

history_db = load_json(STUDENT_HISTORY_JSON)
example_recs = {name: all_recs[name] for name in example_people}
charts = _recommendation_history(example_recs, history_db)
print("historical meals:", len(history.get("consumption", [])))
for path in charts:
    display(Image.open(path))

# Also show the generated box+mask composite for every held-out test image.
from visualize_outputs import _grounding_images
from granica.grounding import load_groundings
from granica.segmentation import load_segmentation
all_overlays = _grounding_images(load_groundings(), load_segmentation())
test_split_name = "dinner-2026-09-25-test"
for path in all_overlays:
    if Path(path).parent.parent.name == test_split_name:
        display(Image.open(path))
"""),

    md("""## Summary

The steps above show train-only face comparisons, all held-out test plates,
grounding boxes and labeled SAM overlays, weight predictions, several validated
recommendations, and history-alignment charts. Recommendations are closed-set:
every item is on tonight's menu and every unit has a fixed gram value.
"""),
]

write_nb("1_train_pipeline.ipynb", TRAIN)
write_nb("2_inference_pipeline.ipynb", INFERENCE)
print("done")
