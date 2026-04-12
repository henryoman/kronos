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


ASSETS = {
    "btc_5m": {
        "csv": Path("data/MEXC_BTCUSDT.P, 5.csv"),
        "trades": Path("outputs/multi_asset/btc_5m_trades.csv"),
    },
    "sol_5m": {
        "csv": Path("data/MEXC_SOLUSDT.P, 5 (1).csv"),
        "trades": Path("outputs/multi_asset/sol_5m_trades.csv"),
    },
    "zec_5m": {
        "csv": Path("data/BINANCE_ZECUSDT, 5.csv"),
        "trades": Path("outputs/multi_asset/zec_5m_trades.csv"),
    },
}

OUT_DIR = Path("outputs/multi_asset")
SUMMARY_PATH = OUT_DIR / "signal_accuracy_summary.csv"
BUCKET_PATH = OUT_DIR / "probability_buckets.csv"
REGIME_PATH = OUT_DIR / "regime_search.csv"

FEE_BPS_PER_SIDE = 5
SLIPPAGE_BPS_PER_SIDE = 2
ROUND_TRIP_COST_1X = 2 * ((FEE_BPS_PER_SIDE + SLIPPAGE_BPS_PER_SIDE) / 10_000)


def rolling_prior_percentile(series: pd.Series, window: int = 365, min_periods: int = 50) -> pd.Series:
    values = series.to_numpy(dtype=float)
    out = np.full(len(values), np.nan)
    for i, value in enumerate(values):
        history = values[max(0, i - window) : i]
        history = history[~np.isnan(history)]
        if len(history) >= min_periods and not np.isnan(value):
            out[i] = float((history <= value).mean())
    return pd.Series(out, index=series.index)


def rsi(close: pd.Series, window: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(window).mean()
    loss = (-delta.clip(upper=0)).rolling(window).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - 100 / (1 + rs)


def load_price_features(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df.columns = [str(c).strip().lower() for c in df.columns]
    df = df.rename(columns={"time": "timestamp", "vol": "volume"})
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s", utc=True).dt.tz_convert(None)
    df = df.sort_values("timestamp").reset_index(drop=True)

    df["log_close"] = np.log(df["close"])
    df["log_ret"] = df["log_close"].diff()
    for window in [24, 72, 288]:
        df[f"ret_{window}"] = df["close"] / df["close"].shift(window) - 1.0
        df[f"vol_{window}"] = df["log_ret"].rolling(window).std()
        df[f"sma_{window}"] = df["close"].rolling(window).mean()
        df[f"above_sma_{window}"] = df["close"] > df[f"sma_{window}"]

    prev_close = df["close"].shift(1)
    tr = pd.concat(
        [
            df["high"] - df["low"],
            (df["high"] - prev_close).abs(),
            (df["low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    df["atr_24_pct"] = tr.rolling(24).mean() / df["close"]
    df["volume_ratio_24"] = df["volume"] / df["volume"].rolling(24).mean()
    df["rsi_14"] = rsi(df["close"])

    for col in ["atr_24_pct", "vol_24", "vol_72", "volume_ratio_24", "rsi_14", "ret_24", "ret_72"]:
        df[f"{col}_live_pct"] = rolling_prior_percentile(df[col])

    return df


def enrich(asset: str, config: dict) -> pd.DataFrame:
    trades = pd.read_csv(config["trades"], parse_dates=["signal_timestamp", "entry_timestamp", "exit_timestamp"])
    features = load_price_features(config["csv"])
    feature_cols = [
        "timestamp",
        "ret_24",
        "ret_72",
        "ret_288",
        "atr_24_pct",
        "vol_24",
        "vol_72",
        "volume_ratio_24",
        "rsi_14",
        "above_sma_24",
        "above_sma_72",
        "above_sma_288",
        "atr_24_pct_live_pct",
        "vol_24_live_pct",
        "vol_72_live_pct",
        "volume_ratio_24_live_pct",
        "rsi_14_live_pct",
        "ret_24_live_pct",
        "ret_72_live_pct",
    ]
    out = trades.merge(
        features[feature_cols],
        left_on="signal_timestamp",
        right_on="timestamp",
        how="left",
    ).drop(columns=["timestamp"])
    out["asset"] = asset
    out["actual_up_current"] = out["exit_close"] > out["current_close"]
    out["actual_up_entry"] = out["exit_close"] > out["entry_open"]
    out["mean_pred_up"] = out["mean_final_pred_close"] > out["current_close"]
    out["mean_pred_correct"] = out["mean_pred_up"] == out["actual_up_current"]
    out["prob_signal_correct"] = (
        ((out["upside_probability"] >= 0.60) & out["actual_up_entry"])
        | ((out["upside_probability"] <= 0.40) & ~out["actual_up_entry"])
    )
    out["future_abs_ret"] = np.log(out["exit_close"] / out["current_close"]).abs()
    return out


def summary(enriched: pd.DataFrame) -> dict:
    return {
        "asset": enriched["asset"].iloc[0],
        "signals": len(enriched),
        "mean_forecast_direction_accuracy": float(enriched["mean_pred_correct"].mean()),
        "probability_signal_accuracy": float(enriched["prob_signal_correct"].mean()),
        "gross_trade_win_rate": float((enriched["gross_return"] > 0).mean()),
        "net_trade_win_rate": float((enriched["net_return"] > 0).mean()),
        "average_net_return": float(enriched["net_return"].mean()),
        "compounded_return": float((1.0 + enriched["net_return"]).prod() - 1.0),
        "max_drawdown": float(((1.0 + enriched["net_return"]).cumprod() / (1.0 + enriched["net_return"]).cumprod().cummax() - 1.0).min()),
    }


def buckets(enriched: pd.DataFrame) -> pd.DataFrame:
    return (
        enriched.groupby(["asset", "upside_probability"])
        .agg(
            n=("signal_number", "count"),
            actual_up_rate=("actual_up_current", "mean"),
            signal_accuracy=("prob_signal_correct", "mean"),
            avg_net_return=("net_return", "mean"),
        )
        .reset_index()
    )


def base_mask(df: pd.DataFrame, rule: str) -> tuple[pd.Series, int]:
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


def regime_mask(df: pd.DataFrame, rule: str) -> pd.Series:
    rules = {
        "all": pd.Series(True, index=df.index),
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
        "trend72_up": df["above_sma_72"].eq(True),
        "trend72_down": df["above_sma_72"].eq(False),
        "trend288_up": df["above_sma_288"].eq(True),
        "trend288_down": df["above_sma_288"].eq(False),
    }
    return rules[rule].fillna(False)


def evaluate_rule(df: pd.DataFrame, p_rule: str, filter_rule: str) -> dict:
    mask, side = base_mask(df, p_rule)
    mask &= regime_mask(df, filter_rule)
    subset = df[mask].copy()
    if side == 1:
        correct = subset["actual_up_entry"]
        underlying = subset["exit_close"] / subset["entry_open"] - 1.0
    else:
        correct = ~subset["actual_up_entry"]
        underlying = 1.0 - subset["exit_close"] / subset["entry_open"]
    net_1x = underlying - ROUND_TRIP_COST_1X
    net_10x = 10.0 * underlying - 10.0 * ROUND_TRIP_COST_1X
    return {
        "asset": subset["asset"].iloc[0] if len(subset) else df["asset"].iloc[0],
        "p_rule": p_rule,
        "filter": filter_rule,
        "side": side,
        "n": len(subset),
        "accuracy": float(correct.mean()) if len(subset) else 0.0,
        "avg_net_1x": float(net_1x.mean()) if len(subset) else 0.0,
        "total_1x": float((1.0 + net_1x).prod() - 1.0) if len(subset) else 0.0,
        "avg_net_10x": float(net_10x.mean()) if len(subset) else 0.0,
        "total_10x": float((1.0 + net_10x).prod() - 1.0) if len(subset) else 0.0,
    }


def regime_search(enriched: pd.DataFrame) -> pd.DataFrame:
    p_rules = ["long_p100", "long_p80", "long_p60", "short_p0", "short_p20", "short_p40"]
    filters = [
        "all",
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
        "trend72_up",
        "trend72_down",
        "trend288_up",
        "trend288_down",
    ]
    rows = []
    for asset, asset_df in enriched.groupby("asset"):
        for p_rule in p_rules:
            for filter_rule in filters:
                result = evaluate_rule(asset_df, p_rule, filter_rule)
                if result["n"] >= 20:
                    rows.append(result)
    return pd.DataFrame(rows).sort_values(["asset", "avg_net_1x"], ascending=[True, False])


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    enriched_parts = [enrich(asset, config) for asset, config in ASSETS.items()]
    enriched = pd.concat(enriched_parts, ignore_index=True)
    enriched.to_csv(OUT_DIR / "enriched_5m_trades.csv", index=False)

    summary_df = pd.DataFrame([summary(part) for _, part in enriched.groupby("asset")])
    bucket_df = buckets(enriched)
    regime_df = regime_search(enriched)

    summary_df.to_csv(SUMMARY_PATH, index=False)
    bucket_df.to_csv(BUCKET_PATH, index=False)
    regime_df.to_csv(REGIME_PATH, index=False)

    print("Raw 5m accuracy")
    print(summary_df.to_string(index=False, formatters={
        "mean_forecast_direction_accuracy": "{:.2%}".format,
        "probability_signal_accuracy": "{:.2%}".format,
        "gross_trade_win_rate": "{:.2%}".format,
        "net_trade_win_rate": "{:.2%}".format,
        "average_net_return": "{:.4%}".format,
        "compounded_return": "{:.2%}".format,
        "max_drawdown": "{:.2%}".format,
    }))
    print()
    print("Probability buckets")
    print(bucket_df.to_string(index=False, formatters={
        "actual_up_rate": "{:.2%}".format,
        "signal_accuracy": "{:.2%}".format,
        "avg_net_return": "{:.4%}".format,
    }))
    print()
    print("Top regimes by asset, min n=20")
    top = regime_df.groupby("asset").head(12)
    print(top.to_string(index=False, formatters={
        "accuracy": "{:.2%}".format,
        "avg_net_1x": "{:.4%}".format,
        "total_1x": "{:.2%}".format,
        "avg_net_10x": "{:.4%}".format,
        "total_10x": "{:.2%}".format,
    }))
    print()
    print(f"saved_summary: {SUMMARY_PATH}")
    print(f"saved_buckets: {BUCKET_PATH}")
    print(f"saved_regimes: {REGIME_PATH}")


if __name__ == "__main__":
    main()
