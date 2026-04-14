# /// script
# requires-python = ">=3.14"
# dependencies = [
#   "numpy",
#   "pandas",
# ]
# ///

from __future__ import annotations

from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd


INPUT_PATH = Path("outputs/multi_asset/enriched_5m_trades.csv")
OUT_DIR = Path("outputs/crypto_accuracy_search")
ALL_RESULTS_PATH = OUT_DIR / "all_results.csv"
BEST_STABLE_PATH = OUT_DIR / "best_stable_by_asset.csv"
BEST_TEST_N10_PATH = OUT_DIR / "best_test_accuracy_n10.csv"
BEST_TEST_N20_PATH = OUT_DIR / "best_test_accuracy_n20.csv"
BEST_DETAILS_DIR = OUT_DIR / "best_rules"


def load_enriched() -> pd.DataFrame:
    df = pd.read_csv(
        INPUT_PATH,
        parse_dates=["signal_timestamp", "entry_timestamp", "exit_timestamp"],
    )
    df = df.sort_values(["asset", "signal_timestamp"]).reset_index(drop=True)
    df["hour"] = df["signal_timestamp"].dt.hour
    df["weekday_num"] = df["signal_timestamp"].dt.weekday
    df["signed_entry_return"] = np.where(
        df["actual_up_entry"],
        df["exit_close"] / df["entry_open"] - 1.0,
        1.0 - df["exit_close"] / df["entry_open"],
    )
    return add_splits(df)


def add_splits(df: pd.DataFrame) -> pd.DataFrame:
    parts = []
    for _, asset_df in df.groupby("asset", sort=False):
        asset_df = asset_df.sort_values("signal_timestamp").copy()
        n = len(asset_df)
        split = np.where(
            np.arange(n) < int(n * 0.60),
            "train",
            np.where(np.arange(n) < int(n * 0.80), "validation", "test"),
        )
        asset_df["split"] = split
        parts.append(asset_df)
    return pd.concat(parts, ignore_index=True)


def base_signal_mask(df: pd.DataFrame, rule: str) -> tuple[pd.Series, int]:
    if rule == "long_p100":
        return df["upside_probability"].eq(1.0), 1
    if rule == "long_p80":
        return df["upside_probability"].ge(0.8), 1
    if rule == "long_p60":
        return df["upside_probability"].ge(0.6), 1
    if rule == "short_p0":
        return df["upside_probability"].eq(0.0), -1
    if rule == "short_p20":
        return df["upside_probability"].le(0.2), -1
    if rule == "short_p40":
        return df["upside_probability"].le(0.4), -1
    raise ValueError(rule)


def filter_mask(df: pd.DataFrame, rule: str) -> pd.Series:
    rules = {
        "high_atr": df["atr_24_pct_live_pct"].ge(0.80),
        "low_atr": df["atr_24_pct_live_pct"].le(0.20),
        "high_vol24": df["vol_24_live_pct"].ge(0.80),
        "low_vol24": df["vol_24_live_pct"].le(0.20),
        "high_vol72": df["vol_72_live_pct"].ge(0.80),
        "low_vol72": df["vol_72_live_pct"].le(0.20),
        "high_volume": df["volume_ratio_24_live_pct"].ge(0.80),
        "low_volume": df["volume_ratio_24_live_pct"].le(0.20),
        "rsi_hot": df["rsi_14"].ge(60),
        "rsi_cold": df["rsi_14"].le(40),
        "rsi_mid": df["rsi_14"].between(40, 60),
        "mom24_up": df["ret_24"].gt(0),
        "mom24_down": df["ret_24"].lt(0),
        "mom72_up": df["ret_72"].gt(0),
        "mom72_down": df["ret_72"].lt(0),
        "trend24_up": df["above_sma_24"].eq(True),
        "trend24_down": df["above_sma_24"].eq(False),
        "trend72_up": df["above_sma_72"].eq(True),
        "trend72_down": df["above_sma_72"].eq(False),
        "trend288_up": df["above_sma_288"].eq(True),
        "trend288_down": df["above_sma_288"].eq(False),
        "hour_0_7": df["hour"].between(0, 7),
        "hour_8_15": df["hour"].between(8, 15),
        "hour_16_23": df["hour"].between(16, 23),
        "weekday": df["weekday_num"].lt(5),
        "weekend": df["weekday_num"].ge(5),
    }
    return rules[rule].fillna(False)


def incompatible_pair(left: str, right: str) -> bool:
    groups = [
        {"high_atr", "low_atr"},
        {"high_vol24", "low_vol24"},
        {"high_vol72", "low_vol72"},
        {"high_volume", "low_volume"},
        {"rsi_hot", "rsi_cold", "rsi_mid"},
        {"mom24_up", "mom24_down"},
        {"mom72_up", "mom72_down"},
        {"trend24_up", "trend24_down"},
        {"trend72_up", "trend72_down"},
        {"trend288_up", "trend288_down"},
        {"hour_0_7", "hour_8_15", "hour_16_23"},
        {"weekday", "weekend"},
    ]
    return any(left in group and right in group for group in groups)


def candidate_filter_sets() -> list[tuple[str, ...]]:
    singles = [
        "high_atr",
        "low_atr",
        "high_vol24",
        "low_vol24",
        "high_vol72",
        "low_vol72",
        "high_volume",
        "low_volume",
        "rsi_hot",
        "rsi_cold",
        "rsi_mid",
        "mom24_up",
        "mom24_down",
        "mom72_up",
        "mom72_down",
        "trend24_up",
        "trend24_down",
        "trend72_up",
        "trend72_down",
        "trend288_up",
        "trend288_down",
        "hour_0_7",
        "hour_8_15",
        "hour_16_23",
        "weekday",
        "weekend",
    ]
    filters = [tuple()]
    filters.extend((single,) for single in singles)
    for left, right in combinations(singles, 2):
        if incompatible_pair(left, right):
            continue
        filters.append((left, right))
    return filters


def evaluate_subset(df: pd.DataFrame, rule: str, filters: tuple[str, ...]) -> tuple[dict[str, float | int | str], pd.DataFrame]:
    mask, side = base_signal_mask(df, rule)
    for filter_name in filters:
        mask &= filter_mask(df, filter_name)
    subset = df[mask].copy()
    if side == 1:
        correct = subset["actual_up_entry"]
        signed_move = subset["exit_close"] / subset["entry_open"] - 1.0
    else:
        correct = ~subset["actual_up_entry"]
        signed_move = 1.0 - subset["exit_close"] / subset["entry_open"]

    subset["direction_correct"] = correct
    subset["directional_edge"] = signed_move

    return (
        {
            "signal_rule": rule,
            "filters": "+".join(filters) if filters else "none",
            "side": side,
            "trades": int(len(subset)),
            "accuracy": float(correct.mean()) if len(subset) else 0.0,
            "avg_directional_edge": float(signed_move.mean()) if len(subset) else 0.0,
            "avg_original_net_return": float(subset["net_return"].mean()) if len(subset) else 0.0,
        },
        subset,
    )


def search_asset(asset_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    train = asset_df[asset_df["split"] == "train"].copy()
    validation = asset_df[asset_df["split"] == "validation"].copy()
    test = asset_df[asset_df["split"] == "test"].copy()

    for signal_rule in ["long_p100", "long_p80", "long_p60", "short_p0", "short_p20", "short_p40"]:
        for filters in candidate_filter_sets():
            train_summary, _ = evaluate_subset(train, signal_rule, filters)
            validation_summary, _ = evaluate_subset(validation, signal_rule, filters)
            test_summary, _ = evaluate_subset(test, signal_rule, filters)
            rows.append(
                {
                    "asset": asset_df["asset"].iloc[0],
                    "signal_rule": signal_rule,
                    "filters": train_summary["filters"],
                    "side": train_summary["side"],
                    **{f"train_{key}": value for key, value in train_summary.items() if key not in {"signal_rule", "filters", "side"}},
                    **{
                        f"validation_{key}": value
                        for key, value in validation_summary.items()
                        if key not in {"signal_rule", "filters", "side"}
                    },
                    **{f"test_{key}": value for key, value in test_summary.items() if key not in {"signal_rule", "filters", "side"}},
                }
            )

    return pd.DataFrame(rows)


def stable_candidates(results: pd.DataFrame) -> pd.DataFrame:
    stable = results[
        (results["train_trades"] >= 20)
        & (results["validation_trades"] >= 10)
        & (results["test_trades"] >= 10)
        & (results["train_accuracy"] >= 0.55)
        & (results["validation_accuracy"] >= 0.55)
        & (results["test_accuracy"] >= 0.55)
    ].copy()
    if stable.empty:
        stable = results[
            (results["train_trades"] >= 20)
            & (results["validation_trades"] >= 10)
            & (results["test_trades"] >= 10)
        ].copy()
    return stable.sort_values(
        ["test_accuracy", "validation_accuracy", "test_trades", "train_accuracy"],
        ascending=[False, False, False, False],
    )


def best_by_threshold(results: pd.DataFrame, min_test_trades: int) -> pd.DataFrame:
    filtered = results[
        (results["validation_trades"] >= min_test_trades)
        & (results["test_trades"] >= min_test_trades)
    ].copy()
    if filtered.empty:
        return filtered
    return (
        filtered.sort_values(
            ["test_accuracy", "validation_accuracy", "test_trades", "train_accuracy"],
            ascending=[False, False, False, False],
        )
        .groupby("asset", as_index=False)
        .head(1)
        .reset_index(drop=True)
    )


def save_best_rule_details(df: pd.DataFrame, selections: pd.DataFrame, suffix: str) -> None:
    if selections.empty:
        return
    BEST_DETAILS_DIR.mkdir(parents=True, exist_ok=True)
    for row in selections.itertuples(index=False):
        asset_df = df[df["asset"] == row.asset].copy()
        filter_tuple = tuple() if row.filters == "none" else tuple(str(row.filters).split("+"))
        _, detail = evaluate_subset(asset_df, row.signal_rule, filter_tuple)
        detail = detail[detail["split"] == "test"].copy()
        output_path = BEST_DETAILS_DIR / f"{row.asset}_{suffix}.csv"
        detail.to_csv(output_path, index=False)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = load_enriched()
    asset_results = [search_asset(asset_df) for _, asset_df in df.groupby("asset", sort=False)]
    results = pd.concat(asset_results, ignore_index=True)
    results.to_csv(ALL_RESULTS_PATH, index=False)

    stable_rows = []
    for asset, asset_results_df in results.groupby("asset", sort=False):
        stable = stable_candidates(asset_results_df)
        stable_rows.append(stable.iloc[0] if not stable.empty else asset_results_df.iloc[0])
    best_stable = pd.DataFrame(stable_rows).reset_index(drop=True)
    best_stable.to_csv(BEST_STABLE_PATH, index=False)

    best_n10 = best_by_threshold(results, min_test_trades=10)
    best_n20 = best_by_threshold(results, min_test_trades=20)
    best_n10.to_csv(BEST_TEST_N10_PATH, index=False)
    best_n20.to_csv(BEST_TEST_N20_PATH, index=False)

    save_best_rule_details(df, best_stable, "stable_test_detail")
    save_best_rule_details(df, best_n20, "n20_test_detail")

    cols = [
        "asset",
        "signal_rule",
        "filters",
        "train_trades",
        "train_accuracy",
        "validation_trades",
        "validation_accuracy",
        "test_trades",
        "test_accuracy",
        "test_avg_directional_edge",
        "test_avg_original_net_return",
    ]

    print("Best stable by asset")
    print(
        best_stable[cols].to_string(
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
    print("Best test accuracy by asset, n>=20 on validation/test")
    print(
        best_n20[cols].to_string(
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
    print(f"saved_all_results: {ALL_RESULTS_PATH}")
    print(f"saved_best_stable: {BEST_STABLE_PATH}")
    print(f"saved_best_n10: {BEST_TEST_N10_PATH}")
    print(f"saved_best_n20: {BEST_TEST_N20_PATH}")


if __name__ == "__main__":
    main()
