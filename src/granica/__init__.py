"""Savor: food-intake estimation and identity-aware recommendations.

Sub-modules
-----------
config        paths, runtime flags, device / seeds
io_utils      JSON + image helpers
menu          static menu config loader + closed-set resolution
units         countable unit -> grams tables per category
faceid        InsightFace gallery + identification metrics
grounding     OpenAI vision closed-set food grounding (cached)
segmentation  MobileSAM pixel counts inside boxes (cached)
weights       scale-anchored training labels + per-category pixel->weight models
recommend     OpenAI recommendation with validation/repair
"""
from . import config, io_utils, menu, units, faceid, grounding, segmentation, weights, recommend

__all__ = ["config", "io_utils", "menu", "units", "faceid", "grounding",
           "segmentation", "weights", "recommend"]
