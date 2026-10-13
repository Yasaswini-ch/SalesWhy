"""Predict expected daily revenue per seller.

Trains on the earlier ~70% of a seller's date range, predicts the later
~30%, so "expected revenue" on a given day never peeks at that day's own
outcome. Tries the local TabPFN package first (runs on this machine, no
data leaves it); falls back to gradient boosting (xgboost) if TabPFN
raises (e.g. dataset too large, no GPU/out of memory) or isn't installed.
Writes which backend actually ran per seller so the UI/write-up can report
it honestly (see CLAUDE.md: "never claim 100% local unless true" -- here
TabPFN-local keeps it true, but we still log it rather than assume).
"""
from pathlib import Path

import numpy as np
import pandas as pd

from features import FEATURE_COLUMNS, TARGET_COLUMN

TRAIN_FRACTION = 0.7


def split_train_test(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    cutoff = int(len(df) * TRAIN_FRACTION)
    return df.iloc[:cutoff], df.iloc[cutoff:]


def fit_predict_tabpfn(train: pd.DataFrame, test: pd.DataFrame) -> np.ndarray:
    from tabpfn import TabPFNRegressor

    model = TabPFNRegressor()
    model.fit(train[FEATURE_COLUMNS].values, train[TARGET_COLUMN].values)
    return model.predict(test[FEATURE_COLUMNS].values)


def fit_predict_gbm(train: pd.DataFrame, test: pd.DataFrame) -> np.ndarray:
    from xgboost import XGBRegressor

    model = XGBRegressor(n_estimators=200, max_depth=4, learning_rate=0.05)
    model.fit(train[FEATURE_COLUMNS].values, train[TARGET_COLUMN].values)
    return model.predict(test[FEATURE_COLUMNS].values)


def predict_seller(df: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    train, test = split_train_test(df)
    try:
        preds = fit_predict_tabpfn(train, test)
        backend = "tabpfn-local"
    except Exception as exc:  # pragma: no cover - environment dependent
        print(f"  TabPFN failed ({exc!r}), falling back to gradient boosting")
        preds = fit_predict_gbm(train, test)
        backend = "xgboost-fallback"

    out = test[["date", TARGET_COLUMN]].copy()
    out["predicted_revenue"] = np.clip(preds, 0, None)
    return out, backend


def run(features_dir: Path, out_dir: Path, backend_log_name: str = "predict_backend_log.csv") -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    backend_log = []
    for csv_path in sorted(features_dir.glob("*.csv")):
        if csv_path.name == "ground_truth.csv":
            continue
        df = pd.read_csv(csv_path, parse_dates=["date"])
        preds, backend = predict_seller(df)
        preds.to_csv(out_dir / csv_path.name, index=False)

        mae = (preds[TARGET_COLUMN] - preds["predicted_revenue"]).abs().mean()
        baseline_mae = (preds[TARGET_COLUMN] - preds[TARGET_COLUMN].mean()).abs().mean()
        print(
            f"{csv_path.stem}: backend={backend} test_days={len(preds)} "
            f"MAE={mae:.2f} (mean-baseline MAE={baseline_mae:.2f})"
        )
        backend_log.append({"seller_id": csv_path.stem, "backend": backend, "mae": mae})

    pd.DataFrame(backend_log).to_csv(out_dir.parent / backend_log_name, index=False)


def main() -> None:
    root = Path(__file__).resolve().parent.parent
    run(root / "results" / "features", root / "results" / "predictions")


if __name__ == "__main__":
    main()
