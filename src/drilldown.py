"""Product drill-down: for a flagged window, which products lost the most
revenue compared to their normal daily pace? Pure arithmetic, no model --
"expected" here just means "this product's average daily revenue outside
the window, scaled to the window's length," not a TabPFN prediction.
"""
from __future__ import annotations

import pandas as pd

from detect import BASELINE_LOOKBACK_DAYS

TOP_N = 3


class DrilldownError(ValueError):
    pass


def product_drilldown(df: pd.DataFrame, mapping: dict[str, str | None], finding: dict) -> list[dict]:
    date_col, amount_col, product_col = mapping.get("date"), mapping.get("amount"), mapping.get("product")
    if not product_col or product_col not in df.columns:
        raise DrilldownError("No product column was mapped for this file, so a per-product breakdown isn't available.")

    work = df[[date_col, amount_col, product_col]].copy()
    work.columns = ["date", "revenue", "product"]
    work["date"] = pd.to_datetime(work["date"], errors="coerce", utc=True).dt.tz_localize(None)
    work["revenue"] = pd.to_numeric(work["revenue"], errors="coerce")
    work = work.dropna(subset=["date", "revenue", "product"])
    if work.empty:
        raise DrilldownError("No valid rows to break down by product.")

    window_start = pd.Timestamp(finding["window_start"])
    window_end = pd.Timestamp(finding["window_end"])
    window_days = (window_end - window_start).days + 1

    baseline_start = window_start - pd.Timedelta(days=BASELINE_LOOKBACK_DAYS)
    baseline_mask = (work["date"] >= baseline_start) & (work["date"] < window_start)
    window_mask = (work["date"] >= window_start) & (work["date"] <= window_end)

    if not baseline_mask.any():
        raise DrilldownError("Not enough history before this window to estimate normal product sales.")

    baseline_days = (window_start - baseline_start).days
    baseline_per_product = work[baseline_mask].groupby("product")["revenue"].sum() / baseline_days
    window_per_product = work[window_mask].groupby("product")["revenue"].sum()

    is_spike = finding.get("direction") == "spike"
    all_products = set(baseline_per_product.index) | set(window_per_product.index)
    rows = []
    for product in all_products:
        expected = baseline_per_product.get(product, 0.0) * window_days
        actual = window_per_product.get(product, 0.0)
        # "impact" is always positive-is-notable: how much this product lost
        # (dip windows) or gained (spike windows) vs. its normal pace.
        impact = (actual - expected) if is_spike else (expected - actual)
        rows.append({
            "product": str(product),
            "expected_revenue": round(float(expected), 2),
            "actual_revenue": round(float(actual), 2),
            "impact_revenue": round(float(impact), 2),
            "direction": "spike" if is_spike else "dip",
        })

    rows.sort(key=lambda r: r["impact_revenue"], reverse=True)
    return rows[:TOP_N]
