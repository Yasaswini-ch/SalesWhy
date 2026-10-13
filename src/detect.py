"""Flag revenue-shortfall windows and name the likely cause, deterministically.

Cause attribution never touches an LLM: it compares each candidate feature's
value inside a flagged window against a baseline drawn from the 14 days
right before the window, and ranks candidates by size of change. explain.py
is the only place an LLM gets involved, and only to phrase a finding that
was already decided here (CLAUDE.md principle 1).

Pipeline:
  1. Predict expected revenue for a seller (predict.py, or a baseline).
  2. Find runs of 2+ consecutive days where actual is well below expected.
  3. For each run, rank price/freight/active-products as the cause by how
     much each moved inside the window vs. the pre-window baseline.
  4. If no candidate moved enough, cause = "unknown" (CLAUDE.md's "not sure").
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

SHORTFALL_THRESHOLD = 0.20  # actual must be >=20% below predicted
MIN_CONSECUTIVE_DAYS = 2
MAX_WINDOW_DAYS = 8  # split long shortfall runs so one noisy stretch can't
                      # swallow several distinct events into one mislabeled window
BASELINE_LOOKBACK_DAYS = 14
CONFIDENCE_FLOOR_CHANGE_PCT = 10.0  # candidate must move at least this much to count
CANDIDATE_FEATURES = {
    "price_increase": "avg_price",
    "freight_spike": "freight",
    "stockout": "n_active_products",
}
# Upside (spike) causes -- CLAUDE.md Tier 2's "upside detection": reuse the
# same mechanism with the sign flipped, never a separate model or LLM call.
UPSIDE_CANDIDATE_FEATURES = {
    "price_decrease": "avg_price",
    "freight_decrease": "freight",
    "more_products": "n_active_products",
}
# Causes where the metric moving DOWN is what supports them; everything else
# assumes the metric moving UP is what supports it.
FALLING_SUPPORTS_CAUSE = {"stockout", "price_decrease", "freight_decrease"}


def find_shortfall_windows(actual: pd.Series, predicted: pd.Series, direction: str = "dip") -> list[tuple[int, int]]:
    with np.errstate(divide="ignore", invalid="ignore"):
        if direction == "dip":
            signal = (predicted - actual) / predicted.replace(0, np.nan)
        else:
            signal = (actual - predicted) / predicted.replace(0, np.nan)
    is_short = (signal >= SHORTFALL_THRESHOLD).fillna(False).to_numpy()

    windows = []
    i = 0
    n = len(is_short)
    while i < n:
        if is_short[i]:
            j = i
            while j < n and is_short[j]:
                j += 1
            run_len = j - i
            if run_len >= MIN_CONSECUTIVE_DAYS:
                for chunk_start in range(i, j, MAX_WINDOW_DAYS):
                    chunk_end = min(chunk_start + MAX_WINDOW_DAYS, j) - 1
                    if chunk_end - chunk_start + 1 >= MIN_CONSECUTIVE_DAYS:
                        windows.append((chunk_start, chunk_end))
            i = j
        else:
            i += 1
    return windows


ATTRIBUTION_WINDOW_CAP = 5  # a shortfall run can be longer than the real
                             # event (forecast lag after the cause ends), so
                             # attribute using the leading days, closest to
                             # the actual onset, not the full flagged span


def attribute_cause(df: pd.DataFrame, start: int, end: int, candidate_features: dict[str, str] = CANDIDATE_FEATURES) -> dict:
    baseline_start = max(0, start - BASELINE_LOOKBACK_DAYS)
    baseline_end = start - 1
    if baseline_end < baseline_start:
        baseline_start, baseline_end = max(0, start - 1), max(0, start - 1)

    attribution_end = min(end, start + ATTRIBUTION_WINDOW_CAP - 1)

    candidates = []
    for cause, col in candidate_features.items():
        baseline_val = df[col].iloc[baseline_start:baseline_end + 1].mean()
        window_vals = df[col].iloc[start:attribution_end + 1]

        if baseline_val == 0:
            candidates.append((cause, 0.0))
            continue

        if cause in FALLING_SUPPORTS_CAUSE:
            # This cause is supported by the metric moving DOWN (e.g.
            # products going away, or price/freight dropping to explain a
            # spike) -- sign flips vs. causes supported by a rise.
            per_day = (baseline_val - window_vals) / baseline_val * 100.0
        else:
            per_day = (window_vals - baseline_val) / baseline_val * 100.0
        candidates.append((cause, per_day.mean()))

    candidates.sort(key=lambda c: c[1], reverse=True)
    top_cause, top_change = candidates[0]

    if top_change < CONFIDENCE_FLOOR_CHANGE_PCT:
        return {"cause": "unknown", "change_pct": round(top_change, 1), "confidence": 0.0}

    confidence = round(min(0.95, 0.3 + top_change / 100.0), 2)
    return {"cause": top_cause, "change_pct": round(top_change, 1), "confidence": confidence}


def detect(df: pd.DataFrame, actual_col: str, predicted_col: str, direction: str = "dip") -> list[dict]:
    df = df.reset_index(drop=True)
    windows = find_shortfall_windows(df[actual_col], df[predicted_col], direction=direction)
    candidate_features = CANDIDATE_FEATURES if direction == "dip" else UPSIDE_CANDIDATE_FEATURES

    findings = []
    for start, end in windows:
        actual_mean = df[actual_col].iloc[start:end + 1].mean()
        predicted_mean = df[predicted_col].iloc[start:end + 1].mean()
        if predicted_mean == 0:
            shortfall_pct = 0.0
        elif direction == "dip":
            shortfall_pct = round((predicted_mean - actual_mean) / predicted_mean * 100.0, 1)
        else:
            shortfall_pct = round((actual_mean - predicted_mean) / predicted_mean * 100.0, 1)
        attribution = attribute_cause(df, start, end, candidate_features=candidate_features)
        findings.append({
            "window_start": df["date"].iloc[start],
            "window_end": df["date"].iloc[end],
            "shortfall_pct": shortfall_pct,
            "direction": direction,
            **attribution,
        })
    return findings


def main() -> None:
    root = Path(__file__).resolve().parent.parent
    features_dir = root / "results" / "injected"
    predictions_dir = root / "results" / "injected_predictions"
    out_dir = root / "results" / "detections"
    out_dir.mkdir(parents=True, exist_ok=True)

    all_findings = []
    for csv_path in sorted(features_dir.glob("*.csv")):
        if csv_path.name == "ground_truth.csv":
            continue
        seller_id = csv_path.stem
        pred_path = predictions_dir / csv_path.name
        if not pred_path.exists():
            continue

        df = pd.read_csv(csv_path, parse_dates=["date"])
        preds = pd.read_csv(pred_path, parse_dates=["date"])
        merged = df.merge(preds[["date", "predicted_revenue"]], on="date", how="inner")

        findings = detect(merged, "revenue", "predicted_revenue")
        for f in findings:
            f["seller_id"] = seller_id
        all_findings.extend(findings)
        print(f"{seller_id}: {len(findings)} windows flagged")

    out_df = pd.DataFrame(all_findings)
    out_df.to_csv(out_dir / "findings.csv", index=False)
    print(f"Total findings: {len(out_df)}")


if __name__ == "__main__":
    main()
