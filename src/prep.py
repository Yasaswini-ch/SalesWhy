"""Join Olist orders/order_items/products and aggregate to daily revenue per seller.

Picks the top N sellers by order count and writes one CSV per seller to
results/daily/<seller_id>.csv with columns: date, revenue, freight, n_orders,
n_items, n_active_products, avg_price.
"""
import argparse
from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
OUT_DIR = Path(__file__).resolve().parent.parent / "results" / "daily"

# Only count orders that actually shipped/delivered; excludes canceled/unavailable.
VALID_STATUSES = {"delivered", "shipped", "invoiced", "approved", "processing"}


def load_joined() -> pd.DataFrame:
    orders = pd.read_csv(
        DATA_DIR / "olist_orders_dataset.csv",
        usecols=["order_id", "order_purchase_timestamp", "order_status"],
        parse_dates=["order_purchase_timestamp"],
    )
    items = pd.read_csv(
        DATA_DIR / "olist_order_items_dataset.csv",
        usecols=["order_id", "product_id", "seller_id", "price", "freight_value"],
    )
    products = pd.read_csv(
        DATA_DIR / "olist_products_dataset.csv",
        usecols=["product_id", "product_category_name"],
    )

    orders = orders[orders["order_status"].isin(VALID_STATUSES)]
    df = items.merge(orders, on="order_id", how="inner")
    df = df.merge(products, on="product_id", how="left")
    df["date"] = df["order_purchase_timestamp"].dt.date
    return df


def top_sellers(df: pd.DataFrame, n: int) -> list[str]:
    counts = df.groupby("seller_id")["order_id"].nunique().sort_values(ascending=False)
    return counts.head(n).index.tolist()


def daily_aggregate(df: pd.DataFrame, seller_id: str) -> pd.DataFrame:
    sub = df[df["seller_id"] == seller_id]
    daily = sub.groupby("date").agg(
        revenue=("price", "sum"),
        freight=("freight_value", "sum"),
        n_orders=("order_id", "nunique"),
        n_items=("product_id", "count"),
        n_active_products=("product_id", "nunique"),
        avg_price=("price", "mean"),
    ).reset_index()

    full_range = pd.date_range(daily["date"].min(), daily["date"].max(), freq="D")
    daily["date"] = pd.to_datetime(daily["date"])
    daily = daily.set_index("date").reindex(full_range).rename_axis("date").reset_index()

    # avg_price is undefined on a zero-order day, not 0 -- filling it with 0
    # would read as "price crashed" to any downstream trailing-average
    # feature. Carry the last known price forward instead; everything else
    # (revenue, freight, counts) is genuinely 0 on a day with no orders.
    daily["avg_price"] = daily["avg_price"].ffill().bfill()
    other_cols = ["revenue", "freight", "n_orders", "n_items", "n_active_products"]
    daily[other_cols] = daily[other_cols].fillna(0.0)
    for col in ("n_orders", "n_items", "n_active_products"):
        daily[col] = daily[col].astype(int)
    return daily


def main(n_sellers: int) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = load_joined()
    sellers = top_sellers(df, n_sellers)
    print(f"Top {n_sellers} sellers by order count: {sellers}")

    for seller_id in sellers:
        daily = daily_aggregate(df, seller_id)
        out_path = OUT_DIR / f"{seller_id}.csv"
        daily.to_csv(out_path, index=False)
        print(
            f"{seller_id}: {len(daily)} days, "
            f"{daily['date'].min().date()} to {daily['date'].max().date()}, "
            f"total revenue {daily['revenue'].sum():.2f}"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-sellers", type=int, default=5)
    args = parser.parse_args()
    main(args.n_sellers)
