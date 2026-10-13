"""Inject synthetic sales drops with known causes into held-out test days.

Only touches the test portion of each seller's series (the same tail split
predict.py trains on vs. predicts), so the model that generates "expected
revenue" never sees an injected day during training. Three cause types,
matching the evaluation table in CLAUDE.md:

  price_increase: trailing avg price up 15%, revenue down (price elasticity).
  stockout:       5 days with zero active products / zero orders.
  freight_spike:  freight roughly doubles, revenue down (checkout friction).

Note: CLAUDE.md's Data section also lists "promo removal" as a synthetic
cause, but detect.py's candidate causes are price/freight/active-products/
calendar -- there's no feature that distinguishes "price reverted after a
promo" from a plain price increase. Decision (logged per CLAUDE.md's
ambiguous-decision rule): treat promo removal as a price_increase injection
for now; this is itself a finding worth a line in the write-up (the tool
can't tell these apart from sales data alone).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from features import build_features
from predict import TRAIN_FRACTION

# Inject into the raw daily aggregates, not the already-rolled features --
# a trailing-7-day average absorbs a 4-day bump into noise if you scale the
# rolled column directly instead of the raw one feeding it. Recompute
# features afterwards so trailing stats propagate the change correctly.
DAILY_DIR = Path(__file__).resolve().parent.parent / "results" / "daily"
OUT_DIR = Path(__file__).resolve().parent.parent / "results" / "injected"

CAUSES = ["price_increase", "stockout", "freight_spike"]
WINDOW_LEN = {"price_increase": 4, "stockout": 5, "freight_spike": 4}
TARGET_PER_CAUSE_PER_SELLER = 2  # x5 sellers = 10 per cause overall
# Gap must exceed detect.py's shortfall-run tolerance and the baseline's
# trailing lookback (14d), or two nearby injections merge into one flagged
# window / contaminate each other's baseline. 16 keeps them cleanly apart.
GAP_DAYS = 16
RNG_SEED = 42


def _apply_injection(df: pd.DataFrame, start: int, cause: str) -> dict:
    length = WINDOW_LEN[cause]
    idx = range(start, start + length)

    if cause == "price_increase":
        change_pct = 15.0
        df.loc[idx, "avg_price"] *= 1.15
        df.loc[idx, "revenue"] *= 0.75  # demand response to the price rise

    elif cause == "freight_spike":
        change_pct = 90.0
        df.loc[idx, "freight"] *= 1.9
        df.loc[idx, "revenue"] *= 0.80

    elif cause == "stockout":
        change_pct = -100.0
        df.loc[idx, "n_active_products"] = 0
        df.loc[idx, "n_orders"] = 0
        df.loc[idx, "n_items"] = 0
        df.loc[idx, "revenue"] = 0.0
        # avg_price intentionally left alone: a stockout means nothing sold,
        # not that the price fell to zero.

    else:
        raise ValueError(cause)

    return {
        "window_start": df.loc[start, "date"],
        "window_end": df.loc[start + length - 1, "date"],
        "cause": cause,
        "change_pct": change_pct,
    }


def inject_for_seller(df: pd.DataFrame, seller_id: str, rng: np.random.Generator) -> tuple[pd.DataFrame, list[dict]]:
    df = df.copy().reset_index(drop=True)
    test_start = int(len(df) * TRAIN_FRACTION)

    causes_cycle = list(CAUSES) * TARGET_PER_CAUSE_PER_SELLER
    rng.shuffle(causes_cycle)

    ground_truth = []
    cursor = test_start
    for cause in causes_cycle:
        length = WINDOW_LEN[cause]
        if cursor + length >= len(df) - GAP_DAYS:
            break
        record = _apply_injection(df, cursor, cause)
        record["seller_id"] = seller_id
        ground_truth.append(record)
        cursor += length + GAP_DAYS

    return df, ground_truth


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(RNG_SEED)

    all_ground_truth = []
    for csv_path in sorted(DAILY_DIR.glob("*.csv")):
        seller_id = csv_path.stem
        daily = pd.read_csv(csv_path, parse_dates=["date"])
        injected_daily, ground_truth = inject_for_seller(daily, seller_id, rng)
        injected = build_features(injected_daily)
        injected.to_csv(OUT_DIR / csv_path.name, index=False)
        all_ground_truth.extend(ground_truth)
        print(f"{seller_id}: injected {len(ground_truth)} windows")

    gt_df = pd.DataFrame(all_ground_truth)
    gt_df.to_csv(OUT_DIR / "ground_truth.csv", index=False)
    print(gt_df["cause"].value_counts())


if __name__ == "__main__":
    main()
