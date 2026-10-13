"""Guess which columns in an uploaded orders CSV are date/amount/quantity/product.

Works on header names first (common Shopify/Etsy/Olist-style exports), then
falls back to value-pattern sniffing (parseable dates, numeric-looking
columns) for headers it doesn't recognize. Returns a best guess per field
plus the full column list, so the UI can pre-select a dropdown per field
and let the user override it.
"""
from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

FIELDS = ["date", "amount", "quantity", "product"]

# Lowercased, non-alphanumeric-stripped header -> field. Covers Shopify,
# Etsy and Olist-style export headers seen in practice.
HEADER_HINTS: dict[str, list[str]] = {
    "date": [
        "createdat", "date", "orderdate", "purchasedate", "ordercreatedate",
        "orderpurchasetimestamp", "transactiondate", "saledate",
        "shippinglimitdate",
    ],
    "amount": [
        "total", "totalprice", "ordertotal", "grandtotal", "price",
        "itemtotal", "saleamount", "amount", "revenue", "subtotal",
        "netamount",
    ],
    "quantity": [
        "quantity", "qty", "itemquantity", "numitems", "units",
        "lineitemquantity",
    ],
    "product": [
        "productname", "lineitemname", "itemname", "title", "sku",
        "productid", "variant", "producttitle", "description",
        "itemdescription", "productdescription",
    ],
}

# Headers that end in one of these almost always hold an identifier, not a
# revenue amount -- an invoice/order number can look exactly like a big
# dollar figure to a plain numeric-column sniff (e.g. InvoiceNo "536365").
_ID_HEADER_SUFFIXES = ("id", "no", "num", "number", "code")


def _normalize(header: str) -> str:
    return re.sub(r"[^a-z0-9]", "", header.lower())


def _match_by_header(columns: list[str]) -> dict[str, str]:
    normalized = {col: _normalize(col) for col in columns}
    guesses: dict[str, str] = {}
    for field, hints in HEADER_HINTS.items():
        for col, norm in normalized.items():
            if norm in hints:
                guesses[field] = col
                break
    return guesses


def _looks_like_date(series: pd.Series, sample: int = 20) -> bool:
    values = series.dropna().head(sample)
    if values.empty:
        return False
    # Bare small integers/floats parse "successfully" as dates (pandas reads
    # them as epoch offsets), which misfires on id/quantity columns. Dates
    # are written as strings with separators in every export we target, so
    # require the values to already look like strings, not plain numbers.
    if pd.to_numeric(values, errors="coerce").notna().mean() > 0.8:
        return False
    parsed = pd.to_datetime(values, errors="coerce", format="mixed")
    return parsed.notna().mean() > 0.8


def _looks_like_number(series: pd.Series, sample: int = 20) -> bool:
    values = series.dropna().head(sample)
    if values.empty:
        return False
    coerced = pd.to_numeric(values, errors="coerce")
    return coerced.notna().mean() > 0.8


def _looks_like_id_header(header: str) -> bool:
    norm = _normalize(header)
    return norm.endswith(_ID_HEADER_SUFFIXES)


def _has_decimals(series: pd.Series, sample: int = 50) -> bool:
    coerced = pd.to_numeric(series.dropna().head(sample), errors="coerce").dropna()
    if coerced.empty:
        return False
    return (coerced % 1 != 0).mean() > 0.3


def _match_by_values(df: pd.DataFrame, already_guessed: set[str]) -> dict[str, str]:
    guesses: dict[str, str] = {}
    remaining = [c for c in df.columns if c not in already_guessed]

    if "date" not in already_guessed:
        for col in remaining:
            if _looks_like_date(df[col]):
                guesses["date"] = col
                break

    if "amount" not in already_guessed:
        numeric_candidates = [
            col for col in remaining
            if col not in guesses.values() and _looks_like_number(df[col]) and not _looks_like_id_header(col)
        ]
        if numeric_candidates:
            # A revenue/price column usually has decimal values; an id or a
            # quantity column is usually whole numbers. Prefer a column with
            # decimals if one exists, then fall back to largest mean (an id
            # column that slips past the header check is still more likely
            # to be huge and round than a price column is).
            decimal_candidates = [c for c in numeric_candidates if _has_decimals(df[c])]
            pool = decimal_candidates or numeric_candidates
            means = {c: pd.to_numeric(df[c], errors="coerce").mean() for c in pool}
            guesses["amount"] = max(means, key=means.get)

    return guesses


def guess_mapping(df: pd.DataFrame) -> dict[str, str | None]:
    columns = list(df.columns)
    guesses = _match_by_header(columns)
    value_guesses = _match_by_values(df, set(guesses.values()))
    for field, col in value_guesses.items():
        guesses.setdefault(field, col)

    return {field: guesses.get(field) for field in FIELDS}


def guess_mapping_from_csv(path: str | Path, nrows: int = 200) -> dict[str, str | None]:
    df = pd.read_csv(path, nrows=nrows)
    return guess_mapping(df)


if __name__ == "__main__":
    import sys

    for csv_path in sys.argv[1:]:
        print(csv_path, "->", guess_mapping_from_csv(csv_path))
