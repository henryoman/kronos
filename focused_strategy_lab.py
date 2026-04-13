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
    "btc_1h": {"csv": Path("data/BINANCE_BTCUSDT.P, 60.csv"), "trades": Path("outputs/trades.csv")},
    "btc_5m": {"csv": Path("data/MEXC_BTCUSDT.P, 5.csv"), "trades": Path("outputs/multi_asset/btc_5m_trades.csv")},
    "sol_5m": {"csv": Path("data/MEXC_SOLUSDT.P, 5 (1).csv"), "trades": Path("outputs/multi_asset/sol_5m_trades.csv")},
    "zec_5m": {"csv": Path("data/BINANCE_ZECUSDT, 5.csv"), "trades": Path("outputs/multi_asset/zec_5m_trades.csv")},
}

OUT_DIR = Path("outputs/focused_lab")
RESULTS_PATH = OUT_DIR / "results.csv"
BEST_TRADES_PATH = OUT_DIR / "best_test_trades.csv"
BEST_EQUITY_PATH = OUT_DIR / "best_test_equity.png"

FEE_BPS_PER_SIDE = 5
SLIPPAGE_BPS_PER_SIDE = 2
COST_1X = 2 * ((FEE_BPS_PER_SIDE + SLIPPAGE_BPS_PER_SIDE) / 10_000)


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


def price_features(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df.columns = [str(c).strip().lower() for c in df.columns]
    df = df.rename(columns={"time": "timestamp", "vol": "volume"})
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s", utc=True).dt.tz_convert(None)
    df = df.sort_values("timestamp").reset_index(drop=True)
    df["log_ret"] = np.log(df["close"]).diff()
    for window in [24, 72, 168, 288]:
        df[f"ret_{window}"] = df["close"] / df["close"].shift(window) - 1
        df[f"vol_{window}"] = df["log_ret"].rolling(window).std()
        df[f"sma_{window}"] = df["close"].rolling(window).mean()
        df[f"above_sma_{window}"] = df["close"] > df[f"sma_{window}"]
    prev_close = df["close"].shift(1)
    tr = pd.concat([(df["high"] - df["low"]), (df["high"] - prev_close).abs(), (df["low"] - prev_close).abs()], axis=1).max(axis=1)
    df["atr_24_pct"] = tr.rolling(24).mean() / df["close"]
    df["atr_72_pct"] = tr.rolling(72).mean() / df["close"]
    df["volume_ratio_24"] = df["volume"] / df["volume"].rolling(24).mean()
    df["rsi_14"] = rsi(df["close"])
    for col in ["atr_24_pct", "atr_72_pct", "vol_24", "vol_72", "volume_ratio_24", "rsi_14"]:
        df[f"{col}_pct"] = rolling_prior_percentile(df[col])
    return df


def add_mae_mfe(trades: pd.DataFrame, prices: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, row in trades.iterrows():
        path = prices[(prices["timestamp"] >= row["entry_timestamp"]) & (prices["timestamp"] <= row["exit_timestamp"])]
        entry = float(row["entry_open"])
        if path.empty:
            max_up = max_down = 0.0
        else:
            max_up = float(path["high"].max() / entry - 1.0)
            max_down = float(path["low"].min() / entry - 1.0)
        rows.append((max_up, max_down))
    out = trades.copy()
    out["path_max_up"] = [x[0] for x in rows]
    out["path_max_down"] = [x[1] for x in rows]
    return out


def enrich(asset: str, cfg: dict) -> pd.DataFrame:
    trades = pd.read_csv(cfg["trades"], parse_dates=["signal_timestamp", "entry_timestamp", "exit_timestamp"])
    prices = price_features(cfg["csv"])
    features = [
        "timestamp", "ret_24", "ret_72", "ret_168", "ret_288", "atr_24_pct", "atr_72_pct",
        "vol_24", "vol_72", "volume_ratio_24", "rsi_14", "above_sma_72", "above_sma_168",
        "above_sma_288", "atr_24_pct_pct", "atr_72_pct_pct", "vol_24_pct", "vol_72_pct",
        "volume_ratio_24_pct", "rsi_14_pct",
    ]
    out = trades.merge(prices[features], left_on="signal_timestamp", right_on="timestamp", how="left").drop(columns=["timestamp"])
    out = add_mae_mfe(out, prices)
    out["asset"] = asset
    n = len(out)
    out["split"] = np.where(np.arange(n) < int(n * .6), "train", np.where(np.arange(n) < int(n * .8), "validation", "test"))
    return out


def mask_for(df: pd.DataFrame, rule: str) -> tuple[pd.Series, int]:
    p = df["upside_probability"]
    masks = {
        "long_p100": (p.eq(1.0), 1),
        "long_p80": (p.ge(0.8), 1),
        "long_p60": (p.ge(0.6), 1),
        "short_p0": (p.eq(0.0), -1),
        "short_p20": (p.le(0.2), -1),
        "short_p40": (p.le(0.4), -1),
    }
    return masks[rule]


def filter_for(df: pd.DataFrame, filt: str) -> pd.Series:
    rules = {
        "all": pd.Series(True, index=df.index),
        "high_atr": df["atr_24_pct_pct"].ge(.8),
        "high_vol24": df["vol_24_pct"].ge(.8),
        "high_vol72": df["vol_72_pct"].ge(.8),
        "low_volume": df["volume_ratio_24_pct"].le(.2),
        "high_volume": df["volume_ratio_24_pct"].ge(.8),
        "rsi_mid": df["rsi_14"].between(40, 60),
        "rsi_cold": df["rsi_14"].le(40),
        "rsi_hot": df["rsi_14"].ge(60),
        "mom24_down": df["ret_24"].lt(0),
        "mom72_down": df["ret_72"].lt(0),
        "mom24_up": df["ret_24"].gt(0),
        "trend72_down": df["above_sma_72"].eq(False),
        "trend168_down": df["above_sma_168"].eq(False),
        "trend72_up": df["above_sma_72"].eq(True),
    }
    return rules[filt].fillna(False)


def candidate_signal_filters(asset: str) -> list[tuple[str, str]]:
    common = [("long_p100", "high_vol24"), ("long_p100", "high_atr"), ("long_p80", "high_vol24"), ("long_p80", "high_atr")]
    if asset == "btc_1h":
        return common + [
            ("long_p100", "high_vol72"), ("long_p100", "mom24_down"), ("long_p100", "trend168_down"),
            ("long_p100", "rsi_cold"), ("long_p80", "high_vol72"),
        ]
    if asset == "btc_5m":
        return [("short_p0", "rsi_mid"), ("long_p80", "high_vol72"), ("long_p80", "high_vol24"), ("long_p80", "high_atr"), ("short_p0", "low_volume")]
    if asset == "sol_5m":
        return [("short_p0", "low_volume"), ("short_p0", "high_vol72"), ("short_p0", "mom72_down"), ("short_p20", "rsi_cold"), ("short_p0", "rsi_hot")]
    if asset == "zec_5m":
        return [("long_p80", "high_atr"), ("long_p80", "high_vol24"), ("long_p80", "high_vol72"), ("long_p60", "high_atr"), ("short_p20", "rsi_cold")]
    return common


def trade_returns(subset: pd.DataFrame, side: int, leverage: float, stop: float | None, take: float | None) -> pd.Series:
    time_ret = side * (subset["exit_close"] / subset["entry_open"] - 1.0)
    result = time_ret.copy()
    if stop is not None:
        stop_move = max((stop - leverage * COST_1X) / leverage, 0.0)
        if side == 1:
            stop_hit = subset["path_max_down"] <= -stop_move
        else:
            stop_hit = subset["path_max_up"] >= stop_move
        result = result.mask(stop_hit, -stop_move)
    if take is not None:
        take_move = take / leverage
        if side == 1:
            take_hit = subset["path_max_up"] >= take_move
        else:
            take_hit = subset["path_max_down"] <= -take_move
        # Conservative ordering: stops override takes when both happen.
        if stop is not None:
            result = result.mask(take_hit & ~stop_hit, take_move)
        else:
            result = result.mask(take_hit, take_move)
    return leverage * result - leverage * COST_1X


def stats(net: pd.Series) -> dict:
    if len(net) == 0:
        return {"trades": 0, "win_rate": 0.0, "avg_net": 0.0, "total": 0.0, "max_dd": 0.0}
    equity = (1.0 + net).cumprod()
    dd = equity / equity.cummax() - 1.0
    return {
        "trades": int(len(net)),
        "win_rate": float((net > 0).mean()),
        "avg_net": float(net.mean()),
        "total": float(equity.iloc[-1] - 1.0),
        "max_dd": float(dd.min()),
    }


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    enriched_by_asset = {asset: enrich(asset, cfg) for asset, cfg in ASSETS.items()}
    for asset, df in enriched_by_asset.items():
        df.to_csv(OUT_DIR / f"{asset}_enriched.csv", index=False)
        print(f"asset={asset}", flush=True)
        for signal, filt in candidate_signal_filters(asset):
            base, side = mask_for(df, signal)
            mask = base & filter_for(df, filt)
            for leverage, stop, take in product([1.0, 2.0, 3.0, 5.0, 10.0], [None, .10, .15, .30, .60], [None, .10, .15, .30, .60, 1.20]):
                if stop is None and take is not None:
                    continue
                row = {"asset": asset, "signal": signal, "filter": filt, "side": side, "leverage": leverage, "stop": stop, "take": take}
                for split in ["train", "validation", "test"]:
                    sub = df[mask & df["split"].eq(split)]
                    net = trade_returns(sub, side, leverage, stop, take)
                    s = stats(net)
                    for key, val in s.items():
                        row[f"{split}_{key}"] = val
                rows.append(row)

    results = pd.DataFrame(rows)
    results.to_csv(RESULTS_PATH, index=False)
    stable = results[
        (results["train_trades"] >= 10)
        & (results["validation_trades"] >= 5)
        & (results["test_trades"] >= 5)
        & (results["train_avg_net"] > 0)
        & (results["validation_avg_net"] > 0)
        & (results["test_avg_net"] > 0)
        & (results["test_max_dd"] > -0.80)
    ].copy()
    if stable.empty:
        stable = results[(results["validation_trades"] >= 5) & (results["test_trades"] >= 5)].copy()
    stable = stable.sort_values(["test_avg_net", "validation_avg_net", "test_trades"], ascending=[False, False, False])

    print("Top focused lab results")
    cols = [
        "asset", "signal", "filter", "leverage", "stop", "take",
        "train_trades", "train_win_rate", "train_avg_net", "train_total", "train_max_dd",
        "validation_trades", "validation_win_rate", "validation_avg_net", "validation_total", "validation_max_dd",
        "test_trades", "test_win_rate", "test_avg_net", "test_total", "test_max_dd",
    ]
    print(stable[cols].head(40).to_string(index=False, formatters={
        c: "{:.2%}".format
        for c in cols
        if c.endswith("win_rate") or c.endswith("avg_net") or c.endswith("total") or c.endswith("max_dd")
    }))

    best = stable.iloc[0]
    df = enriched_by_asset[best["asset"]]
    base, side = mask_for(df, best["signal"])
    mask = base & filter_for(df, best["filter"]) & df["split"].eq("test")
    best_test = df[mask].copy()
    best_test["strategy_net"] = trade_returns(
        best_test,
        side,
        float(best["leverage"]),
        None if pd.isna(best["stop"]) else float(best["stop"]),
        None if pd.isna(best["take"]) else float(best["take"]),
    )
    best_test["strategy_equity"] = (1.0 + best_test["strategy_net"]).cumprod()
    best_test.to_csv(BEST_TRADES_PATH, index=False)
    plt.figure(figsize=(10, 5))
    plt.plot(pd.to_datetime(best_test["signal_timestamp"]), best_test["strategy_equity"])
    plt.title("Focused Lab Best Test Equity")
    plt.xlabel("Signal time")
    plt.ylabel("Equity")
    plt.grid(True, alpha=.3)
    plt.tight_layout()
    plt.savefig(BEST_EQUITY_PATH, dpi=150)
    plt.close()

    print()
    print(f"saved_results: {RESULTS_PATH}")
    print(f"saved_best_test_trades: {BEST_TRADES_PATH}")
    print(f"saved_best_test_equity: {BEST_EQUITY_PATH}")


if __name__ == "__main__":
    main()
