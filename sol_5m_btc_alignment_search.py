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
OUT_DIR = Path("outputs/sol_5m_btc_alignment")
ALL_RESULTS_PATH = OUT_DIR / "all_results.csv"
BEST_STABLE_PATH = OUT_DIR / "best_stable.csv"
BEST_TEST_PATH = OUT_DIR / "best_test_n20.csv"
BEST_POSITIVE_NET_PATH = OUT_DIR / "best_positive_test_net.csv"
BEST_DETAIL_PATH = OUT_DIR / "best_stable_test_detail.csv"


def load_data() -> pd.DataFrame:
    df = pd.read_csv(
        INPUT_PATH,
        parse_dates=["signal_timestamp", "entry_timestamp", "exit_timestamp"],
    )
    sol = df[df["asset"] == "sol_5m"].copy().sort_values("signal_timestamp").reset_index(drop=True)
    btc = df[df["asset"] == "btc_5m"].copy().sort_values("signal_timestamp").reset_index(drop=True)

    btc_cols = [
        "signal_timestamp",
        "upside_probability",
        "forecast_return",
        "ret_24",
        "ret_72",
        "atr_24_pct_live_pct",
        "vol_24_live_pct",
        "vol_72_live_pct",
        "volume_ratio_24_live_pct",
        "rsi_14",
        "above_sma_24",
        "above_sma_72",
        "above_sma_288",
    ]
    btc = btc[btc_cols].rename(
        columns={
            "upside_probability": "btc_upside_probability",
            "forecast_return": "btc_forecast_return",
            "ret_24": "btc_ret_24",
            "ret_72": "btc_ret_72",
            "atr_24_pct_live_pct": "btc_atr_24_pct_live_pct",
            "vol_24_live_pct": "btc_vol_24_live_pct",
            "vol_72_live_pct": "btc_vol_72_live_pct",
            "volume_ratio_24_live_pct": "btc_volume_ratio_24_live_pct",
            "rsi_14": "btc_rsi_14",
            "above_sma_24": "btc_above_sma_24",
            "above_sma_72": "btc_above_sma_72",
            "above_sma_288": "btc_above_sma_288",
        }
    )

    merged = sol.merge(btc, on="signal_timestamp", how="inner")
    merged["hour"] = merged["signal_timestamp"].dt.hour
    merged["weekday_num"] = merged["signal_timestamp"].dt.weekday
    merged["sol_pred_up"] = merged["forecast_return"] > 0
    merged["btc_pred_up"] = merged["btc_forecast_return"] > 0
    merged["pred_agree_up"] = merged["sol_pred_up"] == merged["btc_pred_up"]
    merged["btc_p_bucket"] = pd.cut(
        merged["btc_upside_probability"],
        bins=[-0.01, 0.2, 0.4, 0.6, 0.8, 1.01],
        labels=["p0_0p2", "p0p2_0p4", "p0p4_0p6", "p0p6_0p8", "p0p8_1"],
    )

    n = len(merged)
    merged["split"] = np.where(
        np.arange(n) < int(n * 0.60),
        "train",
        np.where(np.arange(n) < int(n * 0.80), "validation", "test"),
    )
    return merged


def base_signal_mask(df: pd.DataFrame, rule: str) -> tuple[pd.Series, int]:
    if rule == "long_p80":
        return df["upside_probability"].ge(0.8), 1
    if rule == "long_p60":
        return df["upside_probability"].ge(0.6), 1
    if rule == "short_p20":
        return df["upside_probability"].le(0.2), -1
    if rule == "short_p40":
        return df["upside_probability"].le(0.4), -1
    raise ValueError(rule)


def filter_mask(df: pd.DataFrame, name: str) -> pd.Series:
    filters = {
        "sol_high_atr": df["atr_24_pct_live_pct"].ge(0.80),
        "sol_low_atr": df["atr_24_pct_live_pct"].le(0.20),
        "sol_high_vol24": df["vol_24_live_pct"].ge(0.80),
        "sol_low_vol24": df["vol_24_live_pct"].le(0.20),
        "sol_high_vol72": df["vol_72_live_pct"].ge(0.80),
        "sol_low_vol72": df["vol_72_live_pct"].le(0.20),
        "sol_high_volume": df["volume_ratio_24_live_pct"].ge(0.80),
        "sol_low_volume": df["volume_ratio_24_live_pct"].le(0.20),
        "sol_rsi_hot": df["rsi_14"].ge(60),
        "sol_rsi_cold": df["rsi_14"].le(40),
        "sol_rsi_mid": df["rsi_14"].between(40, 60),
        "sol_mom24_up": df["ret_24"].gt(0),
        "sol_mom24_down": df["ret_24"].lt(0),
        "sol_mom72_up": df["ret_72"].gt(0),
        "sol_mom72_down": df["ret_72"].lt(0),
        "sol_trend72_up": df["above_sma_72"].eq(True),
        "sol_trend72_down": df["above_sma_72"].eq(False),
        "sol_trend288_up": df["above_sma_288"].eq(True),
        "sol_trend288_down": df["above_sma_288"].eq(False),
        "btc_high_atr": df["btc_atr_24_pct_live_pct"].ge(0.80),
        "btc_low_atr": df["btc_atr_24_pct_live_pct"].le(0.20),
        "btc_high_vol24": df["btc_vol_24_live_pct"].ge(0.80),
        "btc_low_vol24": df["btc_vol_24_live_pct"].le(0.20),
        "btc_high_vol72": df["btc_vol_72_live_pct"].ge(0.80),
        "btc_low_vol72": df["btc_vol_72_live_pct"].le(0.20),
        "btc_high_volume": df["btc_volume_ratio_24_live_pct"].ge(0.80),
        "btc_low_volume": df["btc_volume_ratio_24_live_pct"].le(0.20),
        "btc_rsi_hot": df["btc_rsi_14"].ge(60),
        "btc_rsi_cold": df["btc_rsi_14"].le(40),
        "btc_rsi_mid": df["btc_rsi_14"].between(40, 60),
        "btc_mom24_up": df["btc_ret_24"].gt(0),
        "btc_mom24_down": df["btc_ret_24"].lt(0),
        "btc_mom72_up": df["btc_ret_72"].gt(0),
        "btc_mom72_down": df["btc_ret_72"].lt(0),
        "btc_trend72_up": df["btc_above_sma_72"].eq(True),
        "btc_trend72_down": df["btc_above_sma_72"].eq(False),
        "btc_trend288_up": df["btc_above_sma_288"].eq(True),
        "btc_trend288_down": df["btc_above_sma_288"].eq(False),
        "btc_p_ge_0p6": df["btc_upside_probability"].ge(0.6),
        "btc_p_ge_0p8": df["btc_upside_probability"].ge(0.8),
        "btc_p_le_0p4": df["btc_upside_probability"].le(0.4),
        "btc_p_le_0p2": df["btc_upside_probability"].le(0.2),
        "btc_forecast_up": df["btc_forecast_return"].gt(0),
        "btc_forecast_down": df["btc_forecast_return"].lt(0),
        "btc_sol_agree": df["pred_agree_up"].eq(True),
        "btc_sol_disagree": df["pred_agree_up"].eq(False),
        "hour_0_7": df["hour"].between(0, 7),
        "hour_8_15": df["hour"].between(8, 15),
        "hour_16_23": df["hour"].between(16, 23),
        "weekday": df["weekday_num"].lt(5),
        "weekend": df["weekday_num"].ge(5),
    }
    return filters[name].fillna(False)


def incompatible_pair(left: str, right: str) -> bool:
    groups = [
        {"sol_high_atr", "sol_low_atr"},
        {"sol_high_vol24", "sol_low_vol24"},
        {"sol_high_vol72", "sol_low_vol72"},
        {"sol_high_volume", "sol_low_volume"},
        {"sol_rsi_hot", "sol_rsi_cold", "sol_rsi_mid"},
        {"sol_mom24_up", "sol_mom24_down"},
        {"sol_mom72_up", "sol_mom72_down"},
        {"sol_trend72_up", "sol_trend72_down"},
        {"sol_trend288_up", "sol_trend288_down"},
        {"btc_high_atr", "btc_low_atr"},
        {"btc_high_vol24", "btc_low_vol24"},
        {"btc_high_vol72", "btc_low_vol72"},
        {"btc_high_volume", "btc_low_volume"},
        {"btc_rsi_hot", "btc_rsi_cold", "btc_rsi_mid"},
        {"btc_mom24_up", "btc_mom24_down"},
        {"btc_mom72_up", "btc_mom72_down"},
        {"btc_trend72_up", "btc_trend72_down"},
        {"btc_trend288_up", "btc_trend288_down"},
        {"btc_p_ge_0p6", "btc_p_le_0p4"},
        {"btc_p_ge_0p8", "btc_p_le_0p2"},
        {"btc_forecast_up", "btc_forecast_down"},
        {"btc_sol_agree", "btc_sol_disagree"},
        {"hour_0_7", "hour_8_15", "hour_16_23"},
        {"weekday", "weekend"},
    ]
    return any(left in group and right in group for group in groups)


def candidate_filter_sets() -> list[tuple[str, ...]]:
    curated = [
        "sol_trend72_up",
        "sol_trend72_down",
        "sol_trend288_up",
        "sol_trend288_down",
        "sol_mom24_up",
        "sol_mom24_down",
        "sol_mom72_up",
        "sol_mom72_down",
        "sol_rsi_hot",
        "sol_rsi_cold",
        "sol_rsi_mid",
        "sol_high_volume",
        "sol_low_volume",
        "sol_high_vol24",
        "sol_low_vol24",
        "btc_p_ge_0p6",
        "btc_p_ge_0p8",
        "btc_p_le_0p4",
        "btc_p_le_0p2",
        "btc_forecast_up",
        "btc_forecast_down",
        "btc_mom24_up",
        "btc_mom24_down",
        "btc_mom72_up",
        "btc_mom72_down",
        "btc_trend72_up",
        "btc_trend72_down",
        "btc_trend288_up",
        "btc_trend288_down",
        "btc_high_volume",
        "btc_low_volume",
        "btc_high_vol24",
        "btc_low_vol24",
        "btc_high_atr",
        "btc_low_atr",
        "btc_sol_agree",
        "btc_sol_disagree",
        "hour_8_15",
        "weekday",
    ]
    out = [tuple()]
    out.extend((item,) for item in curated)
    for left, right in combinations(curated, 2):
        if incompatible_pair(left, right):
            continue
        out.append((left, right))
    for triple in [
        ("sol_trend72_up", "sol_trend288_down", "btc_forecast_down"),
        ("sol_trend72_up", "sol_trend288_down", "btc_mom24_down"),
        ("sol_trend72_up", "sol_trend288_down", "btc_p_le_0p4"),
        ("sol_mom24_up", "sol_mom72_up", "btc_forecast_up"),
        ("sol_mom24_up", "sol_mom72_up", "btc_trend72_up"),
        ("sol_low_volume", "btc_forecast_down", "btc_trend72_down"),
        ("sol_rsi_cold", "btc_forecast_down", "btc_mom24_down"),
        ("sol_rsi_hot", "btc_forecast_up", "btc_mom24_up"),
        ("btc_sol_agree", "btc_p_ge_0p6", "btc_trend72_up"),
        ("btc_sol_agree", "btc_p_le_0p4", "btc_trend72_down"),
    ]:
        out.append(triple)
    return out


def evaluate_subset(df: pd.DataFrame, signal_rule: str, filters: tuple[str, ...]) -> tuple[dict, pd.DataFrame]:
    mask, side = base_signal_mask(df, signal_rule)
    for name in filters:
        mask &= filter_mask(df, name)
    subset = df[mask].copy()
    if side == 1:
        correct = subset["actual_up_entry"]
        directional_edge = subset["exit_close"] / subset["entry_open"] - 1.0
    else:
        correct = ~subset["actual_up_entry"]
        directional_edge = 1.0 - subset["exit_close"] / subset["entry_open"]
    subset["direction_correct"] = correct
    subset["directional_edge"] = directional_edge
    return {
        "trades": int(len(subset)),
        "accuracy": float(correct.mean()) if len(subset) else 0.0,
        "avg_directional_edge": float(directional_edge.mean()) if len(subset) else 0.0,
        "avg_original_net_return": float(subset["net_return"].mean()) if len(subset) else 0.0,
        "loser_rate": float((~correct).mean()) if len(subset) else 0.0,
    }, subset


def search(df: pd.DataFrame) -> pd.DataFrame:
    train = df[df["split"] == "train"].copy()
    validation = df[df["split"] == "validation"].copy()
    test = df[df["split"] == "test"].copy()
    rows = []
    for signal_rule in ["long_p80", "long_p60", "short_p20", "short_p40"]:
        for filters in candidate_filter_sets():
            train_result, _ = evaluate_subset(train, signal_rule, filters)
            validation_result, _ = evaluate_subset(validation, signal_rule, filters)
            test_result, _ = evaluate_subset(test, signal_rule, filters)
            rows.append(
                {
                    "signal_rule": signal_rule,
                    "filters": "+".join(filters) if filters else "none",
                    **{f"train_{k}": v for k, v in train_result.items()},
                    **{f"validation_{k}": v for k, v in validation_result.items()},
                    **{f"test_{k}": v for k, v in test_result.items()},
                }
            )
    return pd.DataFrame(rows)


def baseline_comparison(df: pd.DataFrame, best: pd.Series) -> pd.DataFrame:
    test = df[df["split"] == "test"].copy()
    _, side = base_signal_mask(test, best["signal_rule"])
    base_mask, _ = base_signal_mask(test, best["signal_rule"])
    base = test[base_mask].copy()
    filter_tuple = tuple() if best["filters"] == "none" else tuple(str(best["filters"]).split("+"))
    _, filtered = evaluate_subset(test, best["signal_rule"], filter_tuple)

    base["baseline_correct"] = (
        base["actual_up_entry"] if side == 1 else ~base["actual_up_entry"]
    )
    filtered_ids = set(filtered["signal_number"].tolist())
    base["kept_by_filter"] = base["signal_number"].isin(filtered_ids)

    summary = pd.DataFrame(
        [
            {
                "base_trades": len(base),
                "base_accuracy": float(base["baseline_correct"].mean()) if len(base) else 0.0,
                "filtered_trades": len(filtered),
                "filtered_accuracy": float(filtered["direction_correct"].mean()) if len(filtered) else 0.0,
                "winners_kept": int((base["baseline_correct"] & base["kept_by_filter"]).sum()),
                "winners_dropped": int((base["baseline_correct"] & ~base["kept_by_filter"]).sum()),
                "losers_kept": int((~base["baseline_correct"] & base["kept_by_filter"]).sum()),
                "losers_dropped": int((~base["baseline_correct"] & ~base["kept_by_filter"]).sum()),
            }
        ]
    )
    return filtered, summary


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = load_data()
    results = search(df)
    results["stable_score"] = results[
        ["train_accuracy", "validation_accuracy", "test_accuracy"]
    ].min(axis=1)
    results["mean_accuracy"] = results[
        ["train_accuracy", "validation_accuracy", "test_accuracy"]
    ].mean(axis=1)
    results.to_csv(ALL_RESULTS_PATH, index=False)

    stable = results[
        (results["train_trades"] >= 20)
        & (results["validation_trades"] >= 10)
        & (results["test_trades"] >= 10)
        & (results["train_accuracy"] >= 0.50)
        & (results["validation_accuracy"] >= 0.50)
        & (results["test_accuracy"] >= 0.50)
    ].copy()
    stable = stable.sort_values(
        ["stable_score", "test_accuracy", "mean_accuracy", "test_trades"],
        ascending=[False, False, False, False],
    )
    best_stable = stable.head(1).copy()
    best_stable.to_csv(BEST_STABLE_PATH, index=False)

    best_test = results[
        (results["validation_trades"] >= 20)
        & (results["test_trades"] >= 20)
    ].copy()
    best_test = best_test.sort_values(
        ["test_accuracy", "validation_accuracy", "test_trades"],
        ascending=[False, False, False],
    ).head(1)
    best_test.to_csv(BEST_TEST_PATH, index=False)

    best_positive_net = results[
        (results["train_trades"] >= 20)
        & (results["validation_trades"] >= 10)
        & (results["test_trades"] >= 20)
        & (results["test_avg_original_net_return"] > 0)
    ].copy()
    best_positive_net = best_positive_net.sort_values(
        ["test_accuracy", "validation_accuracy", "test_trades"],
        ascending=[False, False, False],
    ).head(1)
    best_positive_net.to_csv(BEST_POSITIVE_NET_PATH, index=False)

    filtered, comparison = baseline_comparison(df, best_stable.iloc[0])
    filtered.to_csv(BEST_DETAIL_PATH, index=False)
    comparison.to_csv(OUT_DIR / "best_stable_vs_base.csv", index=False)

    cols = [
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
    print("Best stable SOL rule")
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
    print("Best test SOL rule, n>=20 validation/test")
    print(
        best_test[cols].to_string(
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
    print("Best positive-net SOL rule, n>=20 test")
    print(
        best_positive_net[cols].to_string(
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
    print("Best stable filter vs base signal on test")
    print(
        comparison.to_string(
            index=False,
            formatters={
                "base_accuracy": "{:.2%}".format,
                "filtered_accuracy": "{:.2%}".format,
            },
        )
    )
    print()
    print(f"saved_all_results: {ALL_RESULTS_PATH}")
    print(f"saved_best_stable: {BEST_STABLE_PATH}")
    print(f"saved_best_test: {BEST_TEST_PATH}")
    print(f"saved_best_positive_net: {BEST_POSITIVE_NET_PATH}")
    print(f"saved_best_detail: {BEST_DETAIL_PATH}")


if __name__ == "__main__":
    main()
