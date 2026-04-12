# /// script
# requires-python = ">=3.14"
# dependencies = [
#   "matplotlib",
#   "numpy",
#   "pandas",
# ]
# ///

from __future__ import annotations

from itertools import product
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DATA_PATH = Path("data/BINANCE_BTCUSDT.P, 60.csv")
TRADES_PATH = Path("outputs/trades.csv")

ENRICHED_PATH = Path("outputs/regime_enriched_trades.csv")
SEARCH_PATH = Path("outputs/regime_search_results.csv")
BEST_TEST_PATH = Path("outputs/regime_best_test_trades.csv")
BEST_TEST_EQUITY_PATH = Path("outputs/regime_best_test_equity.png")

FEE_BPS_PER_SIDE = 5
SLIPPAGE_BPS_PER_SIDE = 2
ROUND_TRIP_COST_PER_1X = 2 * ((FEE_BPS_PER_SIDE + SLIPPAGE_BPS_PER_SIDE) / 10_000)


def rolling_prior_percentile(series: pd.Series, window: int = 365, min_periods: int = 50) -> pd.Series:
    values = series.to_numpy(dtype=float)
    out = np.full(len(values), np.nan)
    for i, value in enumerate(values):
        start = max(0, i - window)
        hist = values[start:i]
        hist = hist[~np.isnan(hist)]
        if len(hist) >= min_periods and not np.isnan(value):
            out[i] = float((hist <= value).mean())
    return pd.Series(out, index=series.index)


def rsi(close: pd.Series, window: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(window).mean()
    loss = (-delta.clip(upper=0)).rolling(window).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - 100 / (1 + rs)


def load_price_features() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH)
    df.columns = [str(c).strip().lower() for c in df.columns]
    df = df.rename(columns={"time": "timestamp", "vol": "volume"})
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s", utc=True).dt.tz_convert(None)
    df = df.sort_values("timestamp").reset_index(drop=True)

    df["log_close"] = np.log(df["close"])
    df["log_ret"] = df["log_close"].diff()
    for window in [6, 12, 24, 72, 168]:
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
    df["volume_ma_24"] = df["volume"].rolling(24).mean()
    df["volume_ratio_24"] = df["volume"] / df["volume_ma_24"]
    df["rsi_14"] = rsi(df["close"], 14)

    signal_like = df.copy()
    for col in ["atr_24_pct", "vol_24", "vol_72", "volume_ratio_24", "rsi_14", "ret_24", "ret_72", "ret_168"]:
        signal_like[f"{col}_live_pct"] = rolling_prior_percentile(signal_like[col])
    return signal_like


def enrich_trades(trades: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
    feature_cols = [
        "timestamp",
        "ret_6", "ret_12", "ret_24", "ret_72", "ret_168",
        "vol_24", "vol_72", "atr_24_pct",
        "volume_ratio_24", "rsi_14",
        "above_sma_24", "above_sma_72", "above_sma_168",
        "atr_24_pct_live_pct", "vol_24_live_pct", "vol_72_live_pct",
        "volume_ratio_24_live_pct", "rsi_14_live_pct",
        "ret_24_live_pct", "ret_72_live_pct", "ret_168_live_pct",
    ]
    merged = trades.merge(
        features[feature_cols],
        left_on="signal_timestamp",
        right_on="timestamp",
        how="left",
    ).drop(columns=["timestamp"])
    merged["actual_up"] = merged["exit_close"] > merged["current_close"]
    merged["entry_up"] = merged["exit_close"] > merged["entry_open"]
    merged["future_abs_ret_24"] = np.log(merged["exit_close"] / merged["current_close"]).abs()
    return merged


def chronological_split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    n = len(df)
    return (
        df.iloc[: int(n * 0.60)].copy(),
        df.iloc[int(n * 0.60) : int(n * 0.80)].copy(),
        df.iloc[int(n * 0.80) :].copy(),
    )


def base_signal_mask(df: pd.DataFrame, p_rule: str) -> tuple[pd.Series, int]:
    if p_rule == "long_p100":
        return df["upside_probability"].eq(1.0), 1
    if p_rule == "long_p80":
        return df["upside_probability"].ge(0.8), 1
    if p_rule == "long_p60":
        return df["upside_probability"].ge(0.6), 1
    if p_rule == "short_p0":
        return df["upside_probability"].eq(0.0), -1
    if p_rule == "short_p20":
        return df["upside_probability"].le(0.2), -1
    if p_rule == "short_p40":
        return df["upside_probability"].le(0.4), -1
    raise ValueError(p_rule)


def filter_mask(df: pd.DataFrame, rule: str) -> pd.Series:
    true = pd.Series(True, index=df.index)
    rules = {
        "all": true,
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
        "trend168_up": df["above_sma_168"].eq(True),
        "trend168_down": df["above_sma_168"].eq(False),
        "hour_0_7": pd.to_datetime(df["signal_timestamp"]).dt.hour.between(0, 7),
        "hour_8_15": pd.to_datetime(df["signal_timestamp"]).dt.hour.between(8, 15),
        "hour_16_23": pd.to_datetime(df["signal_timestamp"]).dt.hour.between(16, 23),
        "weekday": pd.to_datetime(df["signal_timestamp"]).dt.weekday.lt(5),
        "weekend": pd.to_datetime(df["signal_timestamp"]).dt.weekday.ge(5),
    }
    return rules[rule].fillna(False)


def max_drawdown(equity: pd.Series) -> float:
    if equity.empty:
        return 0.0
    return float((equity / equity.cummax() - 1.0).min())


def evaluate(df: pd.DataFrame, p_rule: str, filters: tuple[str, ...], leverage: float = 1.0) -> dict:
    mask, side = base_signal_mask(df, p_rule)
    for rule in filters:
        mask &= filter_mask(df, rule)
    subset = df[mask].copy()

    if side == 1:
        underlying = subset["exit_close"] / subset["entry_open"] - 1.0
        correct = subset["exit_close"] > subset["entry_open"]
    else:
        underlying = 1.0 - subset["exit_close"] / subset["entry_open"]
        correct = subset["exit_close"] < subset["entry_open"]

    net = leverage * underlying - leverage * ROUND_TRIP_COST_PER_1X
    equity = (1.0 + net).cumprod()
    return {
        "p_rule": p_rule,
        "filters": "+".join(filters) if filters else "none",
        "side": side,
        "leverage": leverage,
        "trades": int(len(subset)),
        "accuracy": float(correct.mean()) if len(subset) else 0.0,
        "avg_net": float(net.mean()) if len(subset) else 0.0,
        "total_return": float(equity.iloc[-1] - 1.0) if len(subset) else 0.0,
        "max_drawdown": max_drawdown(equity) if len(subset) else 0.0,
    }


def candidate_filters() -> list[tuple[str, ...]]:
    singles = [
        "all",
        "high_atr", "low_atr", "high_vol24", "low_vol24", "high_vol72", "low_vol72",
        "high_volume", "low_volume", "rsi_hot", "rsi_cold", "rsi_mid",
        "mom24_up", "mom24_down", "mom72_up", "mom72_down",
        "trend24_up", "trend24_down", "trend72_up", "trend72_down",
        "trend168_up", "trend168_down",
        "weekday", "weekend",
    ]
    pairs = [
        ("high_vol24", "mom24_up"),
        ("high_vol24", "trend72_up"),
        ("high_vol24", "rsi_hot"),
        ("high_vol24", "high_volume"),
        ("high_atr", "mom24_up"),
        ("high_atr", "trend72_up"),
        ("high_atr", "rsi_hot"),
        ("high_atr", "high_volume"),
        ("low_vol24", "mom24_down"),
        ("low_atr", "rsi_mid"),
        ("trend168_up", "mom24_up"),
        ("trend168_down", "mom24_down"),
    ]
    return [(x,) for x in singles] + pairs


def main() -> None:
    trades = pd.read_csv(TRADES_PATH, parse_dates=["signal_timestamp", "entry_timestamp", "exit_timestamp"])
    features = load_price_features()
    enriched = enrich_trades(trades, features)
    enriched.to_csv(ENRICHED_PATH, index=False)

    train, val, test = chronological_split(enriched)
    rows = []
    p_rules = ["long_p100", "long_p80", "long_p60", "short_p0", "short_p20", "short_p40"]

    for p_rule, filters, leverage in product(p_rules, candidate_filters(), [1.0, 2.0, 3.0, 5.0, 10.0]):
        train_result = evaluate(train, p_rule, filters, leverage)
        val_result = evaluate(val, p_rule, filters, leverage)
        test_result = evaluate(test, p_rule, filters, leverage)
        rows.append(
            {
                "p_rule": p_rule,
                "filters": train_result["filters"],
                "side": train_result["side"],
                "leverage": leverage,
                **{f"train_{k}": v for k, v in train_result.items() if k not in {"p_rule", "filters", "side", "leverage"}},
                **{f"val_{k}": v for k, v in val_result.items() if k not in {"p_rule", "filters", "side", "leverage"}},
                **{f"test_{k}": v for k, v in test_result.items() if k not in {"p_rule", "filters", "side", "leverage"}},
            }
        )

    results = pd.DataFrame(rows)
    stable = results[
        (results["train_trades"] >= 10)
        & (results["val_trades"] >= 5)
        & (results["test_trades"] >= 5)
        & (results["train_avg_net"] > 0)
        & (results["val_avg_net"] > 0)
        & (results["test_avg_net"] > 0)
    ].copy()
    if stable.empty:
        stable = results[(results["val_trades"] >= 5) & (results["test_trades"] >= 5)].copy()
    stable = stable.sort_values(["test_avg_net", "val_avg_net", "test_trades"], ascending=[False, False, False])
    results.to_csv(SEARCH_PATH, index=False)

    best = stable.iloc[0]
    best_filters = tuple(best["filters"].split("+")) if best["filters"] != "none" else tuple()
    _, side = base_signal_mask(test, best["p_rule"])
    mask, _ = base_signal_mask(test, best["p_rule"])
    for rule in best_filters:
        mask &= filter_mask(test, rule)
    best_test = test[mask].copy()
    if side == 1:
        best_test["rule_net_return"] = best["leverage"] * (best_test["exit_close"] / best_test["entry_open"] - 1.0) - best["leverage"] * ROUND_TRIP_COST_PER_1X
    else:
        best_test["rule_net_return"] = best["leverage"] * (1.0 - best_test["exit_close"] / best_test["entry_open"]) - best["leverage"] * ROUND_TRIP_COST_PER_1X
    best_test["rule_equity"] = (1.0 + best_test["rule_net_return"]).cumprod()
    best_test.to_csv(BEST_TEST_PATH, index=False)

    plt.figure(figsize=(10, 5))
    plt.plot(pd.to_datetime(best_test["signal_timestamp"]), best_test["rule_equity"])
    plt.title("Best Regime Rule Test Equity")
    plt.xlabel("Signal time")
    plt.ylabel("Equity")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(BEST_TEST_EQUITY_PATH, dpi=150)
    plt.close()

    print("Top stable rules by untouched test avg net")
    cols = [
        "p_rule", "filters", "leverage",
        "train_trades", "train_accuracy", "train_avg_net", "train_total_return", "train_max_drawdown",
        "val_trades", "val_accuracy", "val_avg_net", "val_total_return", "val_max_drawdown",
        "test_trades", "test_accuracy", "test_avg_net", "test_total_return", "test_max_drawdown",
    ]
    print(stable[cols].head(30).to_string(index=False, formatters={
        c: "{:.2%}".format
        for c in cols
        if c.endswith("accuracy") or c.endswith("avg_net") or c.endswith("total_return") or c.endswith("max_drawdown")
    }))
    print()
    print(f"saved_enriched: {ENRICHED_PATH}")
    print(f"saved_search: {SEARCH_PATH}")
    print(f"saved_best_test: {BEST_TEST_PATH}")
    print(f"saved_plot: {BEST_TEST_EQUITY_PATH}")


if __name__ == "__main__":
    main()
