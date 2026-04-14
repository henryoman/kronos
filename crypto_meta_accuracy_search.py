# /// script
# requires-python = ">=3.14"
# dependencies = [
#   "numpy",
#   "pandas",
#   "scikit-learn",
# ]
# ///

from __future__ import annotations

from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


INPUT_PATH = Path("outputs/multi_asset/enriched_5m_trades.csv")
OUT_DIR = Path("outputs/crypto_meta_accuracy_search")
SUMMARY_PATH = OUT_DIR / "best_by_asset.csv"
GRID_PATH = OUT_DIR / "grid.csv"


def load_data() -> pd.DataFrame:
    df = pd.read_csv(
        INPUT_PATH,
        parse_dates=["signal_timestamp", "entry_timestamp", "exit_timestamp"],
    )
    df = df.sort_values(["asset", "signal_timestamp"]).reset_index(drop=True)
    df["hour"] = df["signal_timestamp"].dt.hour.astype(float)
    df["weekday_num"] = df["signal_timestamp"].dt.weekday.astype(float)
    df["raw_p"] = df["upside_probability"].astype(float)
    df["pred_ret"] = df["forecast_return"].astype(float)
    df["abs_pred_ret"] = df["pred_ret"].abs()
    df["target_up"] = df["actual_up_entry"].astype(int)
    return add_splits(df)


def add_splits(df: pd.DataFrame) -> pd.DataFrame:
    parts = []
    for _, asset_df in df.groupby("asset", sort=False):
        asset_df = asset_df.sort_values("signal_timestamp").copy()
        n = len(asset_df)
        asset_df["split"] = np.where(
            np.arange(n) < int(n * 0.60),
            "train",
            np.where(np.arange(n) < int(n * 0.80), "validation", "test"),
        )
        parts.append(asset_df)
    return pd.concat(parts, ignore_index=True)


def fit_scores(asset_df: pd.DataFrame) -> pd.DataFrame:
    train = asset_df[asset_df["split"] == "train"].copy()
    features = [
        "raw_p",
        "pred_ret",
        "abs_pred_ret",
        "atr_24_pct_live_pct",
        "vol_24_live_pct",
        "vol_72_live_pct",
        "volume_ratio_24_live_pct",
        "rsi_14",
        "ret_24",
        "ret_72",
        "above_sma_24",
        "above_sma_72",
        "above_sma_288",
        "hour",
        "weekday_num",
    ]

    scored = asset_df.copy()
    scored["score_raw"] = scored["raw_p"]

    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(train["raw_p"], train["target_up"])
    scored["score_iso"] = iso.predict(scored["raw_p"])

    logit = make_pipeline(
        StandardScaler(),
        LogisticRegression(C=0.25, solver="lbfgs", max_iter=2000),
    )
    logit.fit(train[features], train["target_up"])
    scored["score_logit"] = logit.predict_proba(scored[features])[:, 1]
    return scored


def evaluate(split_df: pd.DataFrame, score_col: str, side: str, long_t: float, short_t: float, min_abs_pred_ret: float) -> dict[str, float | int | str]:
    eligible = split_df[split_df["abs_pred_ret"] >= min_abs_pred_ret].copy()
    score = eligible[score_col]
    position = np.zeros(len(eligible), dtype=int)
    if side in {"both", "long"}:
        position = np.where(score >= long_t, 1, position)
    if side in {"both", "short"}:
        position = np.where(score <= short_t, -1, position)
    eligible["position"] = position
    taken = eligible[eligible["position"] != 0].copy()

    if taken.empty:
        return {
            "trades": 0,
            "accuracy": 0.0,
            "avg_directional_edge": 0.0,
            "avg_original_net_return": 0.0,
        }

    correct = np.where(taken["position"] == 1, taken["actual_up_entry"], ~taken["actual_up_entry"])
    edge = np.where(
        taken["position"] == 1,
        taken["exit_close"] / taken["entry_open"] - 1.0,
        1.0 - taken["exit_close"] / taken["entry_open"],
    )
    return {
        "trades": int(len(taken)),
        "accuracy": float(np.mean(correct)),
        "avg_directional_edge": float(np.mean(edge)),
        "avg_original_net_return": float(taken["net_return"].mean()),
    }


def search_asset(asset_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    scored = fit_scores(asset_df)
    train = scored[scored["split"] == "train"].copy()
    validation = scored[scored["split"] == "validation"].copy()
    test = scored[scored["split"] == "test"].copy()

    rows = []
    for score_col, side, long_t, short_t, min_abs_pred_ret in product(
        ["score_raw", "score_iso", "score_logit"],
        ["both", "long", "short"],
        [0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90],
        [0.45, 0.40, 0.35, 0.30, 0.25, 0.20, 0.15, 0.10],
        [0.0, 0.001, 0.002, 0.005, 0.01],
    ):
        train_result = evaluate(train, score_col, side, long_t, short_t, min_abs_pred_ret)
        validation_result = evaluate(validation, score_col, side, long_t, short_t, min_abs_pred_ret)
        test_result = evaluate(test, score_col, side, long_t, short_t, min_abs_pred_ret)
        rows.append(
            {
                "asset": asset_df["asset"].iloc[0],
                "score_col": score_col,
                "side": side,
                "long_threshold": long_t,
                "short_threshold": short_t,
                "min_abs_pred_ret": min_abs_pred_ret,
                **{f"train_{key}": value for key, value in train_result.items()},
                **{f"validation_{key}": value for key, value in validation_result.items()},
                **{f"test_{key}": value for key, value in test_result.items()},
            }
        )

    results = pd.DataFrame(rows)
    candidates = results[
        (results["train_trades"] >= 20)
        & (results["validation_trades"] >= 10)
        & (results["test_trades"] >= 10)
    ].copy()
    if candidates.empty:
        candidates = results[(results["validation_trades"] >= 10) & (results["test_trades"] >= 10)].copy()
    best = candidates.sort_values(
        ["validation_accuracy", "test_accuracy", "validation_trades", "train_accuracy"],
        ascending=[False, False, False, False],
    ).iloc[0]
    return results, best


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = load_data()
    grids = []
    best_rows = []

    for _, asset_df in df.groupby("asset", sort=False):
        grid, best = search_asset(asset_df)
        grids.append(grid)
        best_rows.append(best)

    full_grid = pd.concat(grids, ignore_index=True)
    best_df = pd.DataFrame(best_rows).reset_index(drop=True)

    full_grid.to_csv(GRID_PATH, index=False)
    best_df.to_csv(SUMMARY_PATH, index=False)

    cols = [
        "asset",
        "score_col",
        "side",
        "long_threshold",
        "short_threshold",
        "min_abs_pred_ret",
        "train_trades",
        "train_accuracy",
        "validation_trades",
        "validation_accuracy",
        "test_trades",
        "test_accuracy",
        "test_avg_directional_edge",
        "test_avg_original_net_return",
    ]
    print("Best meta-model setup by asset")
    print(
        best_df[cols].to_string(
            index=False,
            formatters={
                "train_accuracy": "{:.2%}".format,
                "validation_accuracy": "{:.2%}".format,
                "test_accuracy": "{:.2%}".format,
                "test_avg_directional_edge": "{:.4%}".format,
                "test_avg_original_net_return": "{:.4%}".format,
            },
        )
    )
    print()
    print(f"saved_grid: {GRID_PATH}")
    print(f"saved_summary: {SUMMARY_PATH}")


if __name__ == "__main__":
    main()
