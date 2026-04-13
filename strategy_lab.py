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


ASSETS = {
    "btc_1h": {
        "csv": Path("data/BINANCE_BTCUSDT.P, 60.csv"),
        "trades": Path("outputs/trades.csv"),
        "time_unit": "s",
    },
    "btc_5m": {
        "csv": Path("data/MEXC_BTCUSDT.P, 5.csv"),
        "trades": Path("outputs/multi_asset/btc_5m_trades.csv"),
        "time_unit": "s",
    },
    "sol_5m": {
        "csv": Path("data/MEXC_SOLUSDT.P, 5 (1).csv"),
        "trades": Path("outputs/multi_asset/sol_5m_trades.csv"),
        "time_unit": "s",
    },
    "zec_5m": {
        "csv": Path("data/BINANCE_ZECUSDT, 5.csv"),
        "trades": Path("outputs/multi_asset/zec_5m_trades.csv"),
        "time_unit": "s",
    },
}

OUT_DIR = Path("outputs/strategy_lab")
RESULTS_PATH = OUT_DIR / "results.csv"
BEST_TRADES_PATH = OUT_DIR / "best_test_trades.csv"
BEST_EQUITY_PATH = OUT_DIR / "best_test_equity.png"

FEE_BPS_PER_SIDE = 5
SLIPPAGE_BPS_PER_SIDE = 2
ROUND_TRIP_COST_PER_1X = 2 * ((FEE_BPS_PER_SIDE + SLIPPAGE_BPS_PER_SIDE) / 10_000)


def rolling_prior_percentile(series: pd.Series, window: int = 365, min_periods: int = 50) -> pd.Series:
    values = series.to_numpy(dtype=float)
    out = np.full(len(values), np.nan)
    for i, value in enumerate(values):
        hist = values[max(0, i - window):i]
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


def load_price_features(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df.columns = [str(c).strip().lower() for c in df.columns]
    df = df.rename(columns={"time": "timestamp", "vol": "volume"})
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s", utc=True).dt.tz_convert(None)
    df = df.sort_values("timestamp").reset_index(drop=True)

    df["log_close"] = np.log(df["close"])
    df["log_ret"] = df["log_close"].diff()
    for window in [24, 72, 168, 288]:
        df[f"ret_{window}"] = df["close"] / df["close"].shift(window) - 1.0
        df[f"vol_{window}"] = df["log_ret"].rolling(window).std()
        df[f"sma_{window}"] = df["close"].rolling(window).mean()
        df[f"above_sma_{window}"] = df["close"] > df[f"sma_{window}"]
    prev_close = df["close"].shift(1)
    tr = pd.concat(
        [df["high"] - df["low"], (df["high"] - prev_close).abs(), (df["low"] - prev_close).abs()],
        axis=1,
    ).max(axis=1)
    df["atr_24_pct"] = tr.rolling(24).mean() / df["close"]
    df["atr_72_pct"] = tr.rolling(72).mean() / df["close"]
    df["volume_ratio_24"] = df["volume"] / df["volume"].rolling(24).mean()
    df["rsi_14"] = rsi(df["close"], 14)
    for col in ["atr_24_pct", "atr_72_pct", "vol_24", "vol_72", "volume_ratio_24", "rsi_14", "ret_24", "ret_72"]:
        df[f"{col}_live_pct"] = rolling_prior_percentile(df[col])
    return df


def enrich(asset: str, cfg: dict) -> pd.DataFrame:
    trades = pd.read_csv(cfg["trades"], parse_dates=["signal_timestamp", "entry_timestamp", "exit_timestamp"])
    features = load_price_features(cfg["csv"])
    feature_cols = [
        "timestamp", "ret_24", "ret_72", "ret_168", "ret_288", "atr_24_pct", "atr_72_pct",
        "vol_24", "vol_72", "volume_ratio_24", "rsi_14", "above_sma_24", "above_sma_72",
        "above_sma_168", "above_sma_288", "atr_24_pct_live_pct", "atr_72_pct_live_pct",
        "vol_24_live_pct", "vol_72_live_pct", "volume_ratio_24_live_pct", "rsi_14_live_pct",
        "ret_24_live_pct", "ret_72_live_pct",
    ]
    out = trades.merge(features[feature_cols], left_on="signal_timestamp", right_on="timestamp", how="left").drop(columns=["timestamp"])
    out["asset"] = asset
    return out, features


def split_label(df: pd.DataFrame) -> pd.Series:
    n = len(df)
    idx = np.arange(n)
    return pd.Series(
        np.where(idx < int(n * 0.60), "train", np.where(idx < int(n * 0.80), "validation", "test")),
        index=df.index,
    )


def signal_mask(df: pd.DataFrame, signal: str) -> tuple[pd.Series, int]:
    p = df["upside_probability"]
    if signal == "long_p100":
        return p.eq(1.0), 1
    if signal == "long_p80":
        return p.ge(0.8), 1
    if signal == "long_p60":
        return p.ge(0.6), 1
    if signal == "short_p0":
        return p.eq(0.0), -1
    if signal == "short_p20":
        return p.le(0.2), -1
    if signal == "short_p40":
        return p.le(0.4), -1
    raise ValueError(signal)


def filter_mask(df: pd.DataFrame, filt: str) -> pd.Series:
    true = pd.Series(True, index=df.index)
    rules = {
        "all": true,
        "high_atr24": df["atr_24_pct_live_pct"].ge(0.8),
        "high_atr72": df["atr_72_pct_live_pct"].ge(0.8),
        "low_atr24": df["atr_24_pct_live_pct"].le(0.2),
        "high_vol24": df["vol_24_live_pct"].ge(0.8),
        "high_vol72": df["vol_72_live_pct"].ge(0.8),
        "low_vol24": df["vol_24_live_pct"].le(0.2),
        "high_volume": df["volume_ratio_24_live_pct"].ge(0.8),
        "low_volume": df["volume_ratio_24_live_pct"].le(0.2),
        "rsi_hot": df["rsi_14"].ge(60),
        "rsi_cold": df["rsi_14"].le(40),
        "rsi_mid": df["rsi_14"].between(40, 60),
        "mom24_up": df["ret_24"].gt(0),
        "mom24_down": df["ret_24"].lt(0),
        "mom72_up": df["ret_72"].gt(0),
        "mom72_down": df["ret_72"].lt(0),
        "trend72_up": df["above_sma_72"].eq(True),
        "trend72_down": df["above_sma_72"].eq(False),
        "trend168_up": df["above_sma_168"].eq(True),
        "trend168_down": df["above_sma_168"].eq(False),
    }
    return rules[filt].fillna(False)


def make_filters() -> list[tuple[str, ...]]:
    singles = [
        "all", "high_atr24", "high_atr72", "low_atr24", "high_vol24", "high_vol72", "low_vol24",
        "high_volume", "low_volume", "rsi_hot", "rsi_cold", "rsi_mid", "mom24_up", "mom24_down",
        "mom72_up", "mom72_down", "trend72_up", "trend72_down", "trend168_up", "trend168_down",
    ]
    pairs = [
        ("high_atr24", "mom24_up"), ("high_atr24", "mom24_down"),
        ("high_atr24", "trend72_up"), ("high_atr24", "trend72_down"),
        ("high_vol24", "mom24_up"), ("high_vol24", "mom24_down"),
        ("high_vol24", "trend72_up"), ("high_vol24", "trend72_down"),
        ("high_vol72", "mom72_up"), ("high_vol72", "mom72_down"),
        ("rsi_cold", "mom24_down"), ("rsi_hot", "mom24_up"),
        ("low_volume", "mom72_down"), ("high_volume", "high_atr24"),
    ]
    return [(x,) for x in singles] + pairs


def path_for(row: pd.Series, prices: pd.DataFrame) -> pd.DataFrame:
    return prices[(prices["timestamp"] >= row["entry_timestamp"]) & (prices["timestamp"] <= row["exit_timestamp"])]


def simulate(row: pd.Series, prices: pd.DataFrame, side: int, leverage: float, exit_kind: str, stop_margin: float | None, take_margin: float | None) -> tuple[str, float]:
    entry = float(row["entry_open"])
    path = path_for(row, prices)
    if path.empty:
        exit_price = float(row["exit_close"])
        underlying = side * (exit_price / entry - 1.0)
        return "time_missing_path", leverage * underlying - leverage * ROUND_TRIP_COST_PER_1X

    stop_move = None if stop_margin is None else max((stop_margin - leverage * ROUND_TRIP_COST_PER_1X) / leverage, 0.0)
    take_move = None if take_margin is None else take_margin / leverage

    if exit_kind == "atr":
        atr = float(row["atr_24_pct"])
        if np.isfinite(atr):
            stop_move = 1.5 * atr
            take_move = 3.0 * atr

    if side == 1:
        stop_price = None if stop_move is None else entry * (1.0 - stop_move)
        take_price = None if take_move is None else entry * (1.0 + take_move)
    else:
        stop_price = None if stop_move is None else entry * (1.0 + stop_move)
        take_price = None if take_move is None else entry * (1.0 - take_move)

    for _, bar in path.iterrows():
        if side == 1:
            stop_hit = stop_price is not None and float(bar["low"]) <= stop_price
            take_hit = take_price is not None and float(bar["high"]) >= take_price
        else:
            stop_hit = stop_price is not None and float(bar["high"]) >= stop_price
            take_hit = take_price is not None and float(bar["low"]) <= take_price

        if stop_hit and take_hit:
            exit_price = stop_price
            reason = "stop_same_bar"
            break
        if stop_hit:
            exit_price = stop_price
            reason = "stop"
            break
        if take_hit:
            exit_price = take_price
            reason = "take"
            break
    else:
        exit_price = float(row["exit_close"])
        reason = "time"

    underlying = side * (exit_price / entry - 1.0)
    return reason, leverage * underlying - leverage * ROUND_TRIP_COST_PER_1X


def evaluate_subset(df: pd.DataFrame, prices: pd.DataFrame, signal: str, filters: tuple[str, ...], leverage: float, exit_kind: str, stop_margin: float | None, take_margin: float | None) -> tuple[dict, pd.DataFrame]:
    mask, side = signal_mask(df, signal)
    for filt in filters:
        mask &= filter_mask(df, filt)
    subset = df[mask].copy()

    reasons = []
    returns = []
    for _, row in subset.iterrows():
        reason, net = simulate(row, prices, side, leverage, exit_kind, stop_margin, take_margin)
        reasons.append(reason)
        returns.append(net)
    subset["strategy_exit"] = reasons
    subset["strategy_net_return"] = returns
    subset["strategy_equity"] = (1.0 + subset["strategy_net_return"]).cumprod() if len(subset) else pd.Series(dtype=float)

    wins = subset["strategy_net_return"] > 0
    equity = subset["strategy_equity"]
    max_dd = float((equity / equity.cummax() - 1.0).min()) if len(equity) else 0.0
    summary = {
        "trades": int(len(subset)),
        "win_rate": float(wins.mean()) if len(subset) else 0.0,
        "avg_net": float(subset["strategy_net_return"].mean()) if len(subset) else 0.0,
        "total_return": float(equity.iloc[-1] - 1.0) if len(equity) else 0.0,
        "max_drawdown": max_dd,
        "stops": int(subset["strategy_exit"].str.startswith("stop").sum()) if len(subset) else 0,
        "takes": int((subset["strategy_exit"] == "take").sum()) if len(subset) else 0,
    }
    return summary, subset


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    detail_candidates: dict[str, tuple[pd.DataFrame, pd.DataFrame, dict]] = {}

    configs = []
    for signal, filters, leverage, exit_kind in product(
        ["long_p100", "long_p80", "long_p60", "short_p0", "short_p20", "short_p40"],
        make_filters(),
        [1.0, 2.0, 3.0, 5.0, 10.0],
        ["time", "margin_stop", "stop_take", "atr"],
    ):
        if exit_kind == "time":
            configs.append((signal, filters, leverage, exit_kind, None, None))
        elif exit_kind == "margin_stop":
            for stop in [0.10, 0.15, 0.30, 0.60]:
                configs.append((signal, filters, leverage, exit_kind, stop, None))
        elif exit_kind == "stop_take":
            for stop, take in [(0.10, 0.15), (0.10, 0.30), (0.15, 0.30), (0.30, 0.60), (0.60, 1.20)]:
                configs.append((signal, filters, leverage, exit_kind, stop, take))
        else:
            configs.append((signal, filters, leverage, exit_kind, None, None))

    for asset, cfg in ASSETS.items():
        print(f"asset={asset}", flush=True)
        enriched, prices = enrich(asset, cfg)
        enriched["split"] = split_label(enriched)
        enriched.to_csv(OUT_DIR / f"{asset}_enriched.csv", index=False)

        for signal, filters, leverage, exit_kind, stop_margin, take_margin in configs:
            result = {
                "asset": asset,
                "signal": signal,
                "filters": "+".join(filters),
                "leverage": leverage,
                "exit_kind": exit_kind,
                "stop_margin": stop_margin,
                "take_margin": take_margin,
            }
            enough = True
            for split in ["train", "validation", "test"]:
                split_df = enriched[enriched["split"] == split]
                summary, detail = evaluate_subset(split_df, prices, signal, filters, leverage, exit_kind, stop_margin, take_margin)
                for key, value in summary.items():
                    result[f"{split}_{key}"] = value
                if summary["trades"] < (10 if split == "train" else 5):
                    enough = False
                if split == "test":
                    test_detail = detail
            result["enough_samples"] = enough
            rows.append(result)

    results = pd.DataFrame(rows)
    results.to_csv(RESULTS_PATH, index=False)

    stable = results[
        (results["enough_samples"])
        & (results["train_avg_net"] > 0)
        & (results["validation_avg_net"] > 0)
        & (results["test_avg_net"] > 0)
        & (results["test_max_drawdown"] > -0.80)
    ].copy()
    if stable.empty:
        stable = results[(results["validation_trades"] >= 5) & (results["test_trades"] >= 5)].copy()
    stable = stable.sort_values(["test_avg_net", "validation_avg_net", "test_trades"], ascending=[False, False, False])

    print("Top stable strategy-lab results")
    cols = [
        "asset", "signal", "filters", "leverage", "exit_kind", "stop_margin", "take_margin",
        "train_trades", "train_win_rate", "train_avg_net", "train_total_return", "train_max_drawdown",
        "validation_trades", "validation_win_rate", "validation_avg_net", "validation_total_return", "validation_max_drawdown",
        "test_trades", "test_win_rate", "test_avg_net", "test_total_return", "test_max_drawdown",
    ]
    print(stable[cols].head(40).to_string(index=False, formatters={
        c: "{:.2%}".format
        for c in cols
        if c.endswith("win_rate") or c.endswith("avg_net") or c.endswith("total_return") or c.endswith("max_drawdown")
    }))

    # Recompute and save the best test detail.
    best = stable.iloc[0]
    enriched, prices = enrich(best["asset"], ASSETS[best["asset"]])
    enriched["split"] = split_label(enriched)
    filters = tuple(str(best["filters"]).split("+"))
    _, best_detail = evaluate_subset(
        enriched[enriched["split"] == "test"],
        prices,
        best["signal"],
        filters,
        float(best["leverage"]),
        best["exit_kind"],
        None if pd.isna(best["stop_margin"]) else float(best["stop_margin"]),
        None if pd.isna(best["take_margin"]) else float(best["take_margin"]),
    )
    best_detail.to_csv(BEST_TRADES_PATH, index=False)
    plt.figure(figsize=(10, 5))
    plt.plot(pd.to_datetime(best_detail["signal_timestamp"]), best_detail["strategy_equity"])
    plt.title("Strategy Lab Best Test Equity")
    plt.xlabel("Signal time")
    plt.ylabel("Equity")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(BEST_EQUITY_PATH, dpi=150)
    plt.close()

    print()
    print(f"saved_results: {RESULTS_PATH}")
    print(f"saved_best_test_trades: {BEST_TRADES_PATH}")
    print(f"saved_best_test_equity: {BEST_EQUITY_PATH}")


if __name__ == "__main__":
    main()
