"""Live Savor dashboard. Run with: streamlit run app.py"""
from __future__ import annotations

import os
import sys
from datetime import date
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
from granica.config import ARTIFACTS_DIR, RUNTIME, STUDENT_HISTORY_JSON, WEIGHT_MODELS_JSON
from granica.faceid import FaceIdentifier
from granica.grounding import OpenAIGrounding
from granica.io_utils import dump_json, load_json
from granica.menu import load_menu, menu_for
from granica.recommend import recommend
from granica.segmentation import SAMSegmenter
from granica.weights import predict_plate_items

LIVE_DIR = ARTIFACTS_DIR / "live_runs"
LIVE_DIR.mkdir(parents=True, exist_ok=True)


def save_upload(uploaded, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(uploaded.getbuffer())
    return path


def identify(path: Path) -> dict:
    ident = FaceIdentifier()
    if not ident.load_gallery():
        ident.build_gallery(RUNTIME.db_dir, verbose=False)
    return ident.identify(path)


def analyse_plate(path: Path, day: str, meal: str):
    menu = load_menu()
    whitelist = [e["name"] for e in menu_for(menu, day, meal)]
    grounding = OpenAIGrounding().ground(path, whitelist, day, meal)
    segmentation = SAMSegmenter().segment(
        path, [x["bbox"] for x in grounding.get("items", [])],
        labels=[x["item"] for x in grounding.get("items", [])], use_cache=False)
    models = load_json(WEIGHT_MODELS_JSON) if WEIGHT_MODELS_JSON.exists() else {}
    items = [{"item": item["item"], "category": item.get("category", ""),
              "pixels": int(seg.get("pixel_count", 0))}
             for item, seg in zip(grounding.get("items", []), segmentation.get("boxes", []))]
    return grounding, segmentation, predict_plate_items(items, menu, models)


def append_history(person: str, pre: dict, post: dict, meal: str) -> dict:
    history = load_json(STUDENT_HISTORY_JSON) if STUDENT_HISTORY_JSON.exists() else {"people": {}}
    profile = history.setdefault("people", {}).setdefault(person, {"name": person, "consumption": []})
    before, after = {}, {}
    for row in pre.get("predicted_items", []): before[row["item"]] = before.get(row["item"], 0) + float(row["grams"])
    for row in post.get("predicted_items", []): after[row["item"]] = after.get(row["item"], 0) + float(row["grams"])
    items = []
    for item in sorted(set(before) | set(after)):
        served = round(before.get(item, 0), 1); wasted = round(after.get(item, 0), 1)
        items.append({"item": item, "served_grams": served,
                      "taken_grams": round(max(0, served - wasted), 1),
                      "wasted_grams": wasted, "grams": round(max(0, served - wasted), 1)})
    record = {"date": str(date.today()), "meal": meal, "items": items,
              "served_grams": round(sum(x["served_grams"] for x in items), 1),
              "taken_grams": round(sum(x["taken_grams"] for x in items), 1),
              "wasted_grams": round(sum(x["wasted_grams"] for x in items), 1),
              "total_grams": round(sum(x["taken_grams"] for x in items), 1)}
    profile.setdefault("consumption", []).insert(0, record)
    history["source"] = "Savor live dashboard"
    dump_json(history, STUDENT_HISTORY_JSON)
    return record


def show_plate(title, uploaded, path, result):
    st.markdown(f"#### {title}")
    st.image(uploaded, use_container_width=True)
    viz = ARTIFACTS_DIR / "visualizations" / "segmentation" / path.parent.parent.name / path.parent.name / f"{path.stem}_segmentation.png"
    if viz.exists(): st.image(str(viz), caption="Grounding boxes + point-guided SAM masks", use_container_width=True)
    st.metric("Estimated food", f"{result[2]['total_grams']:.1f} g")
    st.dataframe(result[2]["predicted_items"], use_container_width=True, hide_index=True)


st.set_page_config(page_title="Savor Live", page_icon="🍽️", layout="wide")
st.title("Savor Live")
st.caption("Identity-aware recommendations and before/after food-waste estimation")
menu = load_menu()
days = list(menu["days"])
day = st.selectbox("Service day", days, index=days.index(RUNTIME.day_of_week))
meal = st.selectbox("Meal", ["BREAKFAST", "LUNCH", "DINNER"], index=2)
if not os.environ.get("OPENAI_API_KEY"):
    st.warning("Set OPENAI_API_KEY before launching for live grounding and LLM recommendations.")

st.header("1. Student recommendation")
person_upload = st.file_uploader("Upload a student face photo", type=["jpg", "jpeg", "png"], key="person")
if person_upload and st.button("Identify student", type="primary"):
    path = save_upload(person_upload, LIVE_DIR / str(date.today()) / "person" / person_upload.name)
    with st.spinner("Matching against the train-only gallery..."):
        st.session_state["match"] = identify(path)
match = st.session_state.get("match")
if match and match.get("ok"):
    st.success(f"Matched {match['best']}  |  similarity {match['score']:.3f}")
    st.json({"best": match["best"], "score": match["score"], "topk": match["topk"]})
    if st.button("Generate recommendation", type="primary"):
        db = load_json(STUDENT_HISTORY_JSON)
        with st.spinner("Generating a closed-menu recommendation from history..."):
            try: st.session_state["rec"] = recommend(menu, day, meal, db.get("people", {}).get(match["best"], {}))
            except Exception as exc: st.error(f"Recommendation failed: {exc}")
    if st.session_state.get("rec"):
        st.subheader("Today's recommendation")
        st.write(st.session_state["rec"].get("reasoning", ""))
        st.dataframe(st.session_state["rec"].get("recommendations", []), use_container_width=True, hide_index=True)

st.header("2. Before and after plates")
pre_upload = st.file_uploader("Before meal plate", type=["jpg", "jpeg", "png"], key="pre")
post_upload = st.file_uploader("After meal plate", type=["jpg", "jpeg", "png"], key="post")
if match and match.get("ok") and pre_upload and post_upload and st.button("Analyse meal and update history", type="primary"):
    run_dir = LIVE_DIR / str(date.today()) / match["best"]
    pre_path = save_upload(pre_upload, run_dir / "pre" / pre_upload.name)
    post_path = save_upload(post_upload, run_dir / "post" / post_upload.name)
    with st.spinner("Grounding and segmenting both plates..."):
        try:
            st.session_state["pre_result"] = analyse_plate(pre_path, day, meal)
            st.session_state["post_result"] = analyse_plate(post_path, day, meal)
            st.session_state["pre_path"] = pre_path; st.session_state["post_path"] = post_path
            st.session_state["record"] = append_history(match["best"], st.session_state["pre_result"][2], st.session_state["post_result"][2], meal)
        except Exception as exc: st.error(f"Plate analysis failed: {exc}")
if st.session_state.get("pre_result") and st.session_state.get("post_result"):
    left, right = st.columns(2)
    with left: show_plate("Before", pre_upload, st.session_state["pre_path"], st.session_state["pre_result"])
    with right: show_plate("After", post_upload, st.session_state["post_path"], st.session_state["post_result"])
    record = st.session_state.get("record")
    if record:
        st.subheader("Food waste result")
        a, b, c = st.columns(3)
        a.metric("Served", f"{record['served_grams']:.1f} g")
        b.metric("Consumed", f"{record['taken_grams']:.1f} g")
        c.metric("Wasted", f"{record['wasted_grams']:.1f} g")
        st.dataframe(record["items"], use_container_width=True, hide_index=True)
        st.success(f"History updated for {match['best']}")
