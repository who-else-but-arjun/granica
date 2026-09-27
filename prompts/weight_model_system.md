# Weight Model Contract

Fit one linear `grams = slope * segmented_pixels + intercept` model per menu
category using only training plates and clipped food masks. Save slope,
intercept, sample count, R2, MAE, and the fallback global model. Evaluate every
held-out before/after plate, preserve per-item and total predicted-versus-ground-
truth values in JSON, and save one labeled regression plot per category plus a
test comparison plot.
