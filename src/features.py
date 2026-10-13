"""Build model-ready features from a per-seller daily aggregate CSV.

Input: a CSV produced by prep.py (date, revenue, freight, n_orders, n_items,
n_active_products, avg_price).

Adds: day_of_week, month, days_since_last_price_change, plus trailing
averages for price and freight so the model has signal beyond the raw
same-day values (which are themselves part of what we're trying to explain).
"""
from pathlib import Path

import pandas as pd


def build_features(daily: pd.DataFrame) -> pd.DataFrame:
    df = daily.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    df["day_of_week"] = df["date"].dt.dayofweek
    df["month"] = df["date"].dt.month

    # Trailing 7-day average price/freight, shifted by 1 day so a day's
    # features never include that same day's own avg_price/freight.
    df["avg_price_trailing7"] = df["avg_price"].shift(1).rolling(7, min_periods=1).mean()
    df["avg_freight_trailing7"] = df["freight"].shift(1).rolling(7, min_periods=1).mean()
    df["n_active_products_trailing7"] = (
        df["n_active_products"].shift(1).rolling(7, min_periods=1).mean()
    )

    # Days since the trailing-7 avg price last moved by more than 1%.
    price_changed = df["avg_price_trailing7"].pct_change().abs().fillna(0) > 0.01
    change_group = price_changed.cumsum()
    df["days_since_last_price_change"] = df.groupby(change_group).cumcount()

    df[["avg_price_trailing7", "avg_freight_trailing7", "n_active_products_trailing7"]] = df[
        ["avg_price_trailing7", "avg_freight_trailing7", "n_active_products_trailing7"]
    ].bfill()

    return df


FEATURE_COLUMNS = [
    "day_of_week",
    "month",
    "avg_price_trailing7",
    "avg_freight_trailing7",
    "n_active_products_trailing7",
    "days_since_last_price_change",
]
TARGET_COLUMN = "revenue"


def main() -> None:
    daily_dir = Path(__file__).resolve().parent.parent / "results" / "daily"
    out_dir = Path(__file__).resolve().parent.parent / "results" / "features"
    out_dir.mkdir(parents=True, exist_ok=True)

    for csv_path in sorted(daily_dir.glob("*.csv")):
        daily = pd.read_csv(csv_path)
        featured = build_features(daily)
        out_path = out_dir / csv_path.name
        featured.to_csv(out_path, index=False)
        print(f"{csv_path.stem}: {len(featured)} rows -> {out_path}")


if __name__ == "__main__":
    main()
