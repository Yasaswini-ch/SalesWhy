"""What-if: "if you had kept the price/freight the same, sales would have
been about X." Reverts the causal raw column for a finding's window back to
its pre-window baseline, rebuilds features, and re-predicts with TabPFN --
the same model and methodology as the real prediction, just fed a
counterfactual input. This is explicitly an estimate (CLAUDE.md: "Label it
clearly as an estimate"), not a causal guarantee -- revenue could still have
moved for other reasons even if the price hadn't.
"""
from __future__ import annotations

import pandas as pd

from detect import BASELINE_LOOKBACK_DAYS, CANDIDATE_FEATURES, UPSIDE_CANDIDATE_FEATURES
from features import build_features
from predict import predict_seller

# Only price/freight are continuous enough to usefully ask "what if this had
# stayed the same" -- a stockout's "what if" is just "what if it didn't
# happen," which isn't a slider, and CLAUDE.md scopes this to price/freight.
# Covers both directions: dips (price/freight went up) and spikes (price/
# freight came down) since the mechanism is identical either way.
SUPPORTED_CAUSES = {**CANDIDATE_FEATURES, **UPSIDE_CANDIDATE_FEATURES}
SUPPORTED_CAUSES = {
    c: col for c, col in SUPPORTED_CAUSES.items() if c not in ("stockout", "more_products")
}


class WhatIfError(ValueError):
    pass


def apply_whatif(daily: pd.DataFrame, finding: dict) -> pd.DataFrame:
    cause = finding["cause"]
    if cause not in SUPPORTED_CAUSES:
        raise WhatIfError(f"What-if isn't available for cause '{cause}'.")

    col = SUPPORTED_CAUSES[cause]
    df = daily.copy().reset_index(drop=True)
    df["date"] = pd.to_datetime(df["date"])

    window_start = pd.Timestamp(finding["window_start"])
    window_end = pd.Timestamp(finding["window_end"])
    window_mask = (df["date"] >= window_start) & (df["date"] <= window_end)
    if not window_mask.any():
        raise WhatIfError("This finding's window isn't in the data provided.")

    start_idx = df.index[window_mask][0]
    baseline_start = max(0, start_idx - BASELINE_LOOKBACK_DAYS)
    baseline_val = df[col].iloc[baseline_start:start_idx].mean()
    if pd.isna(baseline_val):
        raise WhatIfError("Not enough history before this window to estimate a baseline.")

    df.loc[window_mask, col] = baseline_val
    return df


def run_whatif(daily: pd.DataFrame, finding: dict) -> dict:
    """Returns the counterfactual predicted revenue for the finding's window,
    plus a plain-language estimate of the difference."""
    modified_daily = apply_whatif(daily, finding)
    featured = build_features(modified_daily)
    preds, backend = predict_seller(featured)

    preds["date"] = pd.to_datetime(preds["date"])
    window_start = pd.Timestamp(finding["window_start"])
    window_end = pd.Timestamp(finding["window_end"])
    window_preds = preds[(preds["date"] >= window_start) & (preds["date"] <= window_end)]
    if window_preds.empty:
        raise WhatIfError("This finding's window falls outside the model's predicted range.")

    actual_mean = window_preds["revenue"].mean()
    whatif_mean = window_preds["predicted_revenue"].mean()
    direction = "higher" if whatif_mean > actual_mean else "lower"
    delta_pct = abs(whatif_mean - actual_mean) / actual_mean * 100.0 if actual_mean else 0.0

    feature_label = {
        "price_increase": "price", "freight_spike": "shipping cost",
        "price_decrease": "price", "freight_decrease": "shipping cost",
    }[finding["cause"]]
    message = (
        f"Estimate: if {feature_label} had stayed the same, sales in this window would have "
        f"been about {delta_pct:.0f}% {direction} (roughly {whatif_mean:,.0f} vs the actual "
        f"{actual_mean:,.0f})."
    )

    return {
        "backend": backend,
        "dates": [str(d.date()) for d in window_preds["date"]],
        "whatif_revenue": [float(v) for v in window_preds["predicted_revenue"]],
        "message": message,
    }
