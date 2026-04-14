# /// script
# requires-python = ">=3.14"
# dependencies = [
#   "numpy",
#   "pandas",
#   "scikit-learn",
# ]
# ///

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

import clean_walkforward_research as cwr


OUT_DIR = Path("outputs/alpha_pocket_search")
FULL_GRID_PATH = OUT_DIR / "filter_grid.csv"
BEST_PATH = OUT_DIR / "best_filters.csv"


def summarize_taken(df: pd.DataFrame) -> dict[str, float | int]:
    if df.empty:
        return {
            "trades": 0,
            "accuracy": 0.0,
            "avg_net_return": 0.0,
            "compounded_return": 0.0,
        }
    step_returns = df.groupby("signal_timestamp", sort=True)["policy_net_return"].mean()
    compounded = float((1.0 + step_returns).prod() - 1.0)
    return {
        "trades": int(len(df)),
        "accuracy": float(df["policy_correct"].mean()),
        "avg_net_return": float(df["policy_net_return"].mean()),
        "compounded_return": compounded,
    }


def add_filter_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    hours = out["signal_timestamp"].dt.hour
    out["hour_0_7"] = hours < 8
    out["hour_8_15"] = (hours >= 8) & (hours < 16)
    out["hour_16_23"] = hours >= 16
    out["weekday"] = out["is_weekend"] == 0
    out["weekend"] = out["is_weekend"] == 1
    out["ret_72_up"] = out["ret_72"] > 0
    out["ret_72_down"] = out["ret_72"] <= 0
    out["ret_288_up"] = out["ret_288"] > 0
    out["ret_288_down"] = out["ret_288"] <= 0
    out["vol_high"] = out["vol_24_live_pct"] >= 0.67
    out["vol_low"] = out["vol_24_live_pct"] <= 0.33
    out["atr_high"] = out["atr_24_pct_live_pct"] >= 0.67
    out["atr_low"] = out["atr_24_pct_live_pct"] <= 0.33
    out["rsi_low"] = out["rsi_14"] <= 35
    out["rsi_high"] = out["rsi_14"] >= 65
    out["rsi_mid"] = (out["rsi_14"] >= 45) & (out["rsi_14"] <= 55)
    out["score_high"] = out["policy_score"] >= 0.75
    out["score_extreme"] = (out["policy_score"] >= 0.85) | (out["policy_score"] <= 0.15)
    return out


def scope_frame(full_df: pd.DataFrame, scope: str) -> tuple[pd.DataFrame, bool]:
    if scope == "all_assets_pooled":
        return full_df.copy(), True
    return full_df[full_df["asset"] == scope].copy(), False


def get_taken_trades(config: pd.Series, full_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    scope_df, include_asset = scope_frame(full_df, str(config["scope"]))
    validation_scored = cwr.validation_predictions(scope_df, include_asset=include_asset)
    _, validation_taken = cwr.apply_policy(
        validation_scored,
        score_col=str(config["score_col"]),
        side=str(config["side"]),
        long_t=float(config["long_threshold"]),
        short_t=float(config["short_threshold"]),
        min_abs_pred_ret=float(config["min_abs_pred_ret"]),
    )

    dev_df = scope_df[scope_df["block"] < cwr.TEST_BLOCK].copy()
    test_df = scope_df[scope_df["block"] == cwr.TEST_BLOCK].copy()
    test_scored = cwr.score_models(dev_df, test_df, include_asset=include_asset)
    _, test_taken = cwr.apply_policy(
        test_scored,
        score_col=str(config["score_col"]),
        side=str(config["side"]),
        long_t=float(config["long_threshold"]),
        short_t=float(config["short_threshold"]),
        min_abs_pred_ret=float(config["min_abs_pred_ret"]),
    )
    return add_filter_columns(validation_taken), add_filter_columns(test_taken)


def filter_specs() -> dict[str, callable]:
    return {
        "weekday": lambda df: df["weekday"],
        "weekend": lambda df: df["weekend"],
        "hour_0_7": lambda df: df["hour_0_7"],
        "hour_8_15": lambda df: df["hour_8_15"],
        "hour_16_23": lambda df: df["hour_16_23"],
        "above_sma_72": lambda df: df["above_sma_72"] == 1,
        "below_sma_72": lambda df: df["above_sma_72"] == 0,
        "above_sma_288": lambda df: df["above_sma_288"] == 1,
        "below_sma_288": lambda df: df["above_sma_288"] == 0,
        "ret_72_up": lambda df: df["ret_72_up"],
        "ret_72_down": lambda df: df["ret_72_down"],
        "ret_288_up": lambda df: df["ret_288_up"],
        "ret_288_down": lambda df: df["ret_288_down"],
        "vol_high": lambda df: df["vol_high"],
        "vol_low": lambda df: df["vol_low"],
        "atr_high": lambda df: df["atr_high"],
        "atr_low": lambda df: df["atr_low"],
        "rsi_low": lambda df: df["rsi_low"],
        "rsi_mid": lambda df: df["rsi_mid"],
        "rsi_high": lambda df: df["rsi_high"],
        "score_high": lambda df: df["score_high"],
        "score_extreme": lambda df: df["score_extreme"],
    }


def run_search() -> tuple[pd.DataFrame, pd.DataFrame]:
    configs = pd.read_csv(cwr.TEST_SUMMARY_PATH)
    full_df = cwr.load_data()
    specs = filter_specs()

    rows: list[dict[str, float | int | str]] = []
    for _, config in configs.iterrows():
        validation_taken, test_taken = get_taken_trades(config, full_df)
        for name, fn in specs.items():
            val_subset = validation_taken[fn(validation_taken)].copy()
            test_subset = test_taken[fn(test_taken)].copy()
            val_stats = summarize_taken(val_subset)
            test_stats = summarize_taken(test_subset)
            rows.append(
                {
                    "scope": config["scope"],
                    "filter_name": name,
                    **{f"validation_{k}": v for k, v in val_stats.items()},
                    **{f"test_{k}": v for k, v in test_stats.items()},
                }
            )

    grid = pd.DataFrame(rows)
    best = (
        grid[
            (grid["validation_trades"] >= 10)
            & (grid["test_trades"] >= 3)
            & (grid["validation_avg_net_return"] > 0)
            & (grid["test_avg_net_return"] > 0)
        ]
        .sort_values(
            [
                "validation_avg_net_return",
                "test_avg_net_return",
                "validation_compounded_return",
                "test_compounded_return",
                "validation_trades",
            ],
            ascending=[False, False, False, False, False],
        )
        .reset_index(drop=True)
    )
    return grid, best


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    grid, best = run_search()
    grid.to_csv(FULL_GRID_PATH, index=False)
    best.to_csv(BEST_PATH, index=False)

    cols = [
        "scope",
        "filter_name",
        "validation_trades",
        "validation_accuracy",
        "validation_avg_net_return",
        "test_trades",
        "test_accuracy",
        "test_avg_net_return",
    ]
    print("Best alpha pockets")
    if best.empty:
        print("No filters passed the validation/test positivity screen.")
    else:
        print(
            best[cols].head(20).to_string(
                index=False,
                formatters={
                    "validation_accuracy": "{:.2%}".format,
                    "validation_avg_net_return": "{:.4%}".format,
                    "test_accuracy": "{:.2%}".format,
                    "test_avg_net_return": "{:.4%}".format,
                },
            )
        )
    print()
    print(f"saved_grid: {FULL_GRID_PATH}")
    print(f"saved_best: {BEST_PATH}")


if __name__ == "__main__":
    main()
