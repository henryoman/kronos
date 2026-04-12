# /// script
# requires-python = ">=3.14"
# dependencies = [
#   "matplotlib",
#   "numpy",
#   "pandas",
# ]
# ///

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DATA_PATH = Path("data/BINANCE_BTCUSDT.P, 60.csv")
TRADES_PATH = Path("outputs/trades.csv")

ENRICHED_PATH = Path("outputs/volatility_enriched_trades.csv")
VOL_BUCKET_PATH = Path("outputs/volatility_accuracy_buckets.csv")
SUBSET_PATH = Path("outputs/volatility_best_subsets.csv")
PLOT_PATH = Path("outputs/volatility_accuracy.png")


def load_price_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH)
    df.columns = [str(c).strip().lower() for c in df.columns]
    df = df.rename(columns={"time": "timestamp", "vol": "volume"})
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s", utc=True).dt.tz_convert(None)
    df = df.sort_values("timestamp").reset_index(drop=True)

    df["log_close"] = np.log(df["close"])
    df["log_ret"] = df["log_close"].diff()
    prev_close = df["close"].shift(1)
    true_range = pd.concat(
        [
            df["high"] - df["low"],
            (df["high"] - prev_close).abs(),
            (df["low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    df["atr_24_pct"] = true_range.rolling(24).mean() / df["close"]
    df["pre_vol_24"] = df["log_ret"].rolling(24).std()
    df["pre_vol_72"] = df["log_ret"].rolling(72).std()
    df["pre_abs_ret_24"] = df["log_ret"].abs().rolling(24).mean()
    return df


def enrich(trades: pd.DataFrame, price_df: pd.DataFrame) -> pd.DataFrame:
    by_time = price_df.set_index("timestamp")
    rows = []

    for _, row in trades.iterrows():
        signal_time = pd.Timestamp(row["signal_timestamp"])
        entry_time = pd.Timestamp(row["entry_timestamp"])
        exit_time = pd.Timestamp(row["exit_timestamp"])

        signal_bar = by_time.loc[signal_time]
        future = price_df[(price_df["timestamp"] > signal_time) & (price_df["timestamp"] <= exit_time)]
        future_vol = float(future["log_ret"].std()) if len(future) > 1 else np.nan
        future_abs_ret = abs(float(np.log(row["exit_close"] / row["current_close"])))

        actual_up = row["exit_close"] > row["current_close"]
        mean_pred_up = row["mean_final_pred_close"] > row["current_close"]
        prob_long = row["upside_probability"] >= 0.60
        prob_short = row["upside_probability"] <= 0.40
        prob_signal_correct = (
            (prob_long and row["exit_close"] > row["entry_open"])
            or (prob_short and row["exit_close"] < row["entry_open"])
        )

        rows.append(
            {
                **row.to_dict(),
                "actual_up": actual_up,
                "mean_pred_up": mean_pred_up,
                "mean_pred_correct": mean_pred_up == actual_up,
                "prob_signal_correct": prob_signal_correct,
                "atr_24_pct": float(signal_bar["atr_24_pct"]),
                "pre_vol_24": float(signal_bar["pre_vol_24"]),
                "pre_vol_72": float(signal_bar["pre_vol_72"]),
                "pre_abs_ret_24": float(signal_bar["pre_abs_ret_24"]),
                "future_vol_24": future_vol,
                "future_abs_ret_24": future_abs_ret,
            }
        )

    out = pd.DataFrame(rows)
    for col in ["atr_24_pct", "pre_vol_24", "pre_vol_72", "pre_abs_ret_24", "future_vol_24"]:
        out[f"{col}_quintile"] = pd.qcut(out[col], 5, labels=False, duplicates="drop") + 1
    for col in ["atr_24_pct", "pre_vol_24", "pre_vol_72"]:
        out[f"{col}_live_percentile"] = rolling_prior_percentile(out[col])
    return out


def rolling_prior_percentile(series: pd.Series, window: int = 365, min_periods: int = 50) -> pd.Series:
    values = series.to_numpy(dtype=float)
    out = np.full(len(values), np.nan)
    for i, value in enumerate(values):
        start = max(0, i - window)
        history = values[start:i]
        history = history[~np.isnan(history)]
        if len(history) < min_periods or np.isnan(value):
            continue
        out[i] = float((history <= value).mean())
    return pd.Series(out, index=series.index)


def bucket_stats(enriched: pd.DataFrame) -> pd.DataFrame:
    frames = []
    for group_col in [
        "atr_24_pct_quintile",
        "pre_vol_24_quintile",
        "pre_vol_72_quintile",
        "future_vol_24_quintile",
    ]:
        stats = (
            enriched.groupby(group_col)
            .agg(
                n=("signal_number", "count"),
                actual_up_rate=("actual_up", "mean"),
                mean_pred_accuracy=("mean_pred_correct", "mean"),
                prob_signal_accuracy=("prob_signal_correct", "mean"),
                net_win_rate=("net_return", lambda x: (x > 0).mean()),
                avg_net_return=("net_return", "mean"),
                avg_atr_24_pct=("atr_24_pct", "mean"),
                avg_future_abs_ret_24=("future_abs_ret_24", "mean"),
            )
            .reset_index()
            .rename(columns={group_col: "bucket"})
        )
        stats.insert(0, "bucket_type", group_col)
        frames.append(stats)
    return pd.concat(frames, ignore_index=True)


def subset_search(enriched: pd.DataFrame) -> pd.DataFrame:
    rows = []
    p_rules = [
        ("p_eq_0", lambda d: d["upside_probability"].eq(0.0)),
        ("p_le_0p2", lambda d: d["upside_probability"].le(0.2)),
        ("p_le_0p4", lambda d: d["upside_probability"].le(0.4)),
        ("p_ge_0p6", lambda d: d["upside_probability"].ge(0.6)),
        ("p_ge_0p8", lambda d: d["upside_probability"].ge(0.8)),
        ("p_eq_1", lambda d: d["upside_probability"].eq(1.0)),
    ]
    vol_cols = ["atr_24_pct_quintile", "pre_vol_24_quintile", "pre_vol_72_quintile"]
    live_vol_rules = [
        ("live_high_atr_80", lambda d: d["atr_24_pct_live_percentile"].ge(0.80)),
        ("live_high_pre24_80", lambda d: d["pre_vol_24_live_percentile"].ge(0.80)),
        ("live_high_pre72_80", lambda d: d["pre_vol_72_live_percentile"].ge(0.80)),
        ("live_low_atr_20", lambda d: d["atr_24_pct_live_percentile"].le(0.20)),
        ("live_low_pre24_20", lambda d: d["pre_vol_24_live_percentile"].le(0.20)),
        ("live_low_pre72_20", lambda d: d["pre_vol_72_live_percentile"].le(0.20)),
    ]
    vol_rules = [
        ("all_vol", lambda d, c: pd.Series(True, index=d.index)),
        ("low_vol_q1", lambda d, c: d[c].eq(1)),
        ("low_vol_q1_q2", lambda d, c: d[c].le(2)),
        ("mid_vol_q2_q4", lambda d, c: d[c].between(2, 4)),
        ("high_vol_q4_q5", lambda d, c: d[c].ge(4)),
        ("high_vol_q5", lambda d, c: d[c].eq(5)),
    ]

    for p_name, p_masker in p_rules:
        for vol_col in vol_cols:
            for vol_name, vol_masker in vol_rules:
                mask = p_masker(enriched) & vol_masker(enriched, vol_col)
                subset = enriched[mask].copy()
                if len(subset) < 20:
                    continue
                if p_name.startswith("p_ge") or p_name == "p_eq_1":
                    correct = subset["actual_up"]
                    side = "long"
                else:
                    correct = ~subset["actual_up"]
                    side = "short"
                rows.append(
                    {
                        "rule": f"{p_name}+{vol_col}:{vol_name}",
                        "side": side,
                        "n": len(subset),
                        "direction_accuracy": float(correct.mean()),
                        "avg_unlevered_direction_return": float(
                            np.where(correct, subset["future_abs_ret_24"], -subset["future_abs_ret_24"]).mean()
                        ),
                        "avg_original_net_return": float(subset["net_return"].mean()),
                        "vol_col": vol_col,
                        "vol_rule": vol_name,
                        "p_rule": p_name,
                        "leakage_safe": False,
                    }
                )
        for vol_name, vol_masker in live_vol_rules:
            mask = p_masker(enriched) & vol_masker(enriched)
            subset = enriched[mask].copy()
            if len(subset) < 20:
                continue
            if p_name.startswith("p_ge") or p_name == "p_eq_1":
                correct = subset["actual_up"]
                side = "long"
            else:
                correct = ~subset["actual_up"]
                side = "short"
            rows.append(
                {
                    "rule": f"{p_name}+{vol_name}",
                    "side": side,
                    "n": len(subset),
                    "direction_accuracy": float(correct.mean()),
                    "avg_unlevered_direction_return": float(
                        np.where(correct, subset["future_abs_ret_24"], -subset["future_abs_ret_24"]).mean()
                    ),
                    "avg_original_net_return": float(subset["net_return"].mean()),
                    "vol_col": "rolling_prior_percentile",
                    "vol_rule": vol_name,
                    "p_rule": p_name,
                    "leakage_safe": True,
                }
            )

    return pd.DataFrame(rows).sort_values(
        ["leakage_safe", "direction_accuracy", "n"],
        ascending=[False, False, False],
    )


def plot_bucket_stats(stats: pd.DataFrame) -> None:
    plot_data = stats[stats["bucket_type"] == "atr_24_pct_quintile"]
    plt.figure(figsize=(9, 5))
    plt.plot(plot_data["bucket"], plot_data["mean_pred_accuracy"], marker="o", label="Mean forecast direction")
    plt.plot(plot_data["bucket"], plot_data["prob_signal_accuracy"], marker="o", label="Probability signal")
    plt.axhline(0.5, color="gray", linestyle="--", linewidth=1)
    plt.title("Kronos Accuracy by ATR Volatility Quintile")
    plt.xlabel("ATR volatility quintile")
    plt.ylabel("Accuracy")
    plt.ylim(0, 1)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(PLOT_PATH, dpi=150)
    plt.close()


def main() -> None:
    price_df = load_price_data()
    trades = pd.read_csv(TRADES_PATH, parse_dates=["signal_timestamp", "entry_timestamp", "exit_timestamp"])
    enriched = enrich(trades, price_df)
    stats = bucket_stats(enriched)
    subsets = subset_search(enriched)

    enriched.to_csv(ENRICHED_PATH, index=False)
    stats.to_csv(VOL_BUCKET_PATH, index=False)
    subsets.to_csv(SUBSET_PATH, index=False)
    plot_bucket_stats(stats)

    print("Accuracy by ATR volatility quintile")
    atr = stats[stats["bucket_type"] == "atr_24_pct_quintile"]
    print(atr.to_string(index=False, formatters={
        "actual_up_rate": "{:.2%}".format,
        "mean_pred_accuracy": "{:.2%}".format,
        "prob_signal_accuracy": "{:.2%}".format,
        "net_win_rate": "{:.2%}".format,
        "avg_net_return": "{:.4%}".format,
        "avg_atr_24_pct": "{:.4%}".format,
        "avg_future_abs_ret_24": "{:.4%}".format,
    }))
    print()
    print("Best volatility-filtered directional subsets, min n=20")
    print(subsets.head(20).to_string(index=False, formatters={
        "direction_accuracy": "{:.2%}".format,
        "avg_unlevered_direction_return": "{:.4%}".format,
        "avg_original_net_return": "{:.4%}".format,
    }))
    print()
    print(f"saved_enriched: {ENRICHED_PATH}")
    print(f"saved_buckets: {VOL_BUCKET_PATH}")
    print(f"saved_subsets: {SUBSET_PATH}")
    print(f"saved_plot: {PLOT_PATH}")


if __name__ == "__main__":
    main()
