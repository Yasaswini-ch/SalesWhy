"""End-to-end: an orders CSV -> daily aggregate -> features -> prediction ->
findings -> explanations. This is what app.py calls for both the "Try sample
shop" button and a real upload, so the web demo and the offline eval/CLI
path run the exact same logic.
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd

from detect import detect
from drilldown import product_drilldown
from explain import explain
from features import build_features
from mapper import guess_mapping
from predict import predict_seller
from whatif import run_whatif

MIN_ROWS = 20  # below this, there isn't enough history to train/predict on


class PipelineError(ValueError):
    """Raised when the uploaded CSV can't be analyzed (bad/missing columns, too little data)."""


def aggregate_daily(df: pd.DataFrame, mapping: dict[str, str | None]) -> pd.DataFrame:
    date_col, amount_col = mapping.get("date"), mapping.get("amount")
    if not date_col or not amount_col:
        raise PipelineError("Couldn't find a date and an amount column in this file.")

    quantity_col = mapping.get("quantity")
    cols = [date_col, amount_col] + ([quantity_col] if quantity_col else [])
    work = df[cols].copy()
    work.columns = ["date", "amount"] + (["quantity"] if quantity_col else [])
    work["date"] = pd.to_datetime(work["date"], errors="coerce").dt.date
    work["amount"] = pd.to_numeric(work["amount"], errors="coerce")

    # "amount" is sometimes a per-unit price rather than a line total (real
    # exports vary -- Shopify's "Total" already includes quantity, but a
    # plain "UnitPrice" column doesn't). A header containing "total" is a
    # reasonably safe signal it's already a line total; otherwise, if a
    # quantity column exists, treat amount as per-unit and multiply.
    amount_is_total = "total" in re.sub(r"[^a-z0-9]", "", amount_col.lower())
    if quantity_col and not amount_is_total:
        work["quantity"] = pd.to_numeric(work["quantity"], errors="coerce")
        work["revenue"] = work["amount"] * work["quantity"]
    else:
        work["revenue"] = work["amount"]

    work = work.dropna(subset=["date", "revenue"])
    if work.empty:
        raise PipelineError("Couldn't read any valid date/amount rows from this file.")

    product_col = mapping.get("product")
    agg_kwargs = {"revenue": ("revenue", "sum"), "n_orders": ("revenue", "count")}
    if product_col and product_col in df.columns:
        work["product"] = df[product_col]
        agg_kwargs["n_active_products"] = ("product", "nunique")

    daily = work.groupby("date").agg(**agg_kwargs).reset_index()

    full_range = pd.date_range(daily["date"].min(), daily["date"].max(), freq="D")
    daily["date"] = pd.to_datetime(daily["date"])
    daily = daily.set_index("date").reindex(full_range).rename_axis("date").reset_index()
    daily["revenue"] = daily["revenue"].fillna(0.0)
    daily["n_orders"] = daily["n_orders"].fillna(0).astype(int)

    if "n_active_products" not in daily.columns:
        # No product column -- fall back to a flat placeholder so detect.py's
        # stockout candidate degrades gracefully to "never triggers" instead
        # of crashing on a missing column.
        daily["n_active_products"] = (daily["n_orders"] > 0).astype(int)
    else:
        daily["n_active_products"] = daily["n_active_products"].fillna(0).astype(int)

    daily["avg_price"] = daily["revenue"] / daily["n_orders"].replace(0, np.nan)
    daily["avg_price"] = daily["avg_price"].ffill().bfill().fillna(0.0)
    daily["freight"] = 0.0  # unknown for generic CSVs without a freight column
    daily["n_items"] = daily["n_orders"]
    return daily


def run(df: pd.DataFrame, mapping: dict[str, str | None] | None = None, language: str = "en") -> dict:
    if mapping is None:
        mapping = guess_mapping(df)

    daily = aggregate_daily(df, mapping)
    if len(daily) < MIN_ROWS:
        raise PipelineError(f"Only found {len(daily)} days of data -- need at least {MIN_ROWS} to analyze trends.")

    featured = build_features(daily)
    preds, backend = predict_seller(featured)
    merged = featured.merge(preds[["date", "predicted_revenue"]], on="date", how="inner")

    dips = detect(merged, "revenue", "predicted_revenue", direction="dip")
    spikes = detect(merged, "revenue", "predicted_revenue", direction="spike")
    findings = sorted(dips + spikes, key=lambda f: f["window_start"])

    biggest_id = max(range(len(findings)), key=lambda i: findings[i]["shortfall_pct"], default=None)
    for i, f in enumerate(findings):
        f["window_start"] = str(f["window_start"].date())
        f["window_end"] = str(f["window_end"].date())
        f["shortfall_pct"] = float(f["shortfall_pct"])
        f["change_pct"] = float(f["change_pct"])
        f["confidence"] = float(f["confidence"])
        # Gemma is slow on CPU (~15-20s/call); only explain the single
        # biggest dip/spike per CLAUDE.md's fallback guidance, not every finding.
        f["explanation"] = explain(f, language=language) if i == biggest_id else None

    return {
        "mapping": mapping,
        "backend": backend,
        "daily": [
            {"date": str(row.date.date()), "revenue": float(row.revenue), "predicted_revenue": float(row.predicted_revenue)}
            for row in merged.itertuples()
        ],
        "findings": findings,
    }


def whatif(df: pd.DataFrame, mapping: dict[str, str | None], finding: dict) -> dict:
    """Re-derives the raw daily aggregate (stateless API: nothing's cached
    between requests) and runs the what-if counterfactual for one finding."""
    daily = aggregate_daily(df, mapping)
    return run_whatif(daily, finding)


def drilldown(df: pd.DataFrame, mapping: dict[str, str | None], finding: dict) -> list[dict]:
    return product_drilldown(df, mapping, finding)
