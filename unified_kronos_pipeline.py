#!/usr/bin/env python3
# /// script
# requires-python = ">=3.14"
# dependencies = [
#   "einops",
#   "huggingface-hub",
#   "matplotlib",
#   "numpy",
#   "pandas",
#   "requests",
#   "scikit-learn",
#   "safetensors",
#   "torch",
#   "tqdm",
# ]
# ///

from __future__ import annotations

import argparse
import json
import shutil
import time
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import requests
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import backtest_kronos as core
from model import Kronos, KronosPredictor, KronosTokenizer

ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT / "outputs" / "unified_kronos_pipeline"
RAW_DIR = OUT_DIR / "raw"
SUMMARY_PATH = OUT_DIR / "summary.csv"
LEADERBOARD_PATH = OUT_DIR / "leaderboard.csv"
SKIPPED_PATH = OUT_DIR / "skipped.csv"

LOOKBACK = 360
PRED_LEN = 24
STEP = 24
SAMPLE_COUNT = 5
TOP_P = 0.95
TEMP = 1.0
LONG_THRESHOLD = 0.60
SHORT_THRESHOLD = 0.40
SEED = 42


@dataclass(frozen=True)
class Experiment:
    asset: str
    symbol: str
    timeframe: str
    source_path: Path | None
    source_interval: str
    lookback: int = LOOKBACK
    market: str = "spot"
    legacy_trades_path: Path | None = None

    @property
    def slug(self) -> str:
        return f"{self.asset}_{self.timeframe}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Unified Kronos backtest + accuracy pipeline.")
    parser.add_argument(
        "--assets",
        nargs="*",
        default=["btc", "sol", "zec"],
        help="Assets to include, e.g. btc sol zec",
    )
    parser.add_argument(
        "--timeframes",
        nargs="*",
        default=["5m", "1h", "12h", "1d", "2d", "1w"],
        help="Timeframes to include, e.g. 5m 1h 12h 1d 2d 1w",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Ignore cached raw trades and recompute.",
    )
    parser.add_argument(
        "--max-experiments",
        type=int,
        default=None,
        help="Limit the number of experiments, useful for smoke tests.",
    )
    parser.add_argument(
        "--max-signals",
        type=int,
        default=None,
        help="Only evaluate the last N signals per experiment.",
    )
    parser.add_argument(
        "--use-binance",
        action="store_true",
        help="Fetch Binance history for all experiments instead of using local CSV sources.",
    )
    parser.add_argument(
        "--sample-count",
        type=int,
        default=None,
        help="Override the Monte Carlo sample count used for each forecast.",
    )
    return parser.parse_args()


def interval_to_timedelta(interval: str) -> pd.Timedelta:
    value = interval.strip().lower()
    if value.endswith("m"):
        return pd.Timedelta(minutes=int(value[:-1]))
    if value.endswith("h"):
        return pd.Timedelta(hours=int(value[:-1]))
    if value.endswith("d"):
        return pd.Timedelta(days=int(value[:-1]))
    if value.endswith("w"):
        return pd.Timedelta(weeks=int(value[:-1]))
    raise ValueError(f"Unsupported interval: {interval}")


def interval_to_resample_frequency(interval: str) -> str:
    value = interval.strip().lower()
    if value.endswith("m"):
        return f"{int(value[:-1])}min"
    if value.endswith("h"):
        return f"{int(value[:-1])}h"
    if value.endswith("d"):
        return f"{int(value[:-1])}D"
    if value.endswith("w"):
        return f"{int(value[:-1])}W"
    raise ValueError(f"Unsupported interval: {interval}")


def interval_ratio(target_interval: str, source_interval: str) -> int:
    ratio = interval_to_timedelta(target_interval) / interval_to_timedelta(source_interval)
    rounded = int(round(ratio))
    if rounded <= 0 or abs(ratio - rounded) > 1e-9:
        raise ValueError(f"Cannot align {target_interval} from {source_interval}")
    return rounded


def interval_to_binance(interval: str) -> str:
    value = interval.strip()
    if value.isdigit():
        minutes = int(value)
        if minutes % 60 == 0:
            return f"{minutes // 60}h"
        return f"{minutes}m"
    return value.lower()


def choose_binance_fetch_interval(interval: str) -> str:
    normalized = interval_to_binance(interval)
    if normalized in {"1m", "3m", "5m", "15m", "30m", "1h", "2h", "4h", "6h", "8h", "12h", "1d", "3d", "1w", "1m"}:
        return normalized
    if normalized == "2d":
        return "1d"
    raise ValueError(f"Unsupported Binance interval: {interval}")


def resample_candles(df: pd.DataFrame, target_interval: str) -> pd.DataFrame:
    if df.empty:
        return df.copy()
    if target_interval == "5m" and len(df) and pd.infer_freq(df["timestamp"].tail(min(len(df), 20))) == "5min":
        return df.copy()

    aggregated = (
        df.sort_values("timestamp")
        .set_index("timestamp")
        .resample(interval_to_resample_frequency(target_interval), label="left", closed="left")
        .agg(
            {
                "open": "first",
                "high": "max",
                "low": "min",
                "close": "last",
                "volume": "sum",
                "amount": "sum",
            }
        )
        .dropna(subset=["open", "high", "low", "close"])
        .reset_index()
    )
    return aggregated


def build_experiments(assets: list[str], timeframes: list[str]) -> list[Experiment]:
    catalog = [
        Experiment(
            asset="btc",
            symbol="BTCUSDT",
            timeframe="5m",
            source_path=ROOT / "data" / "MEXC_BTCUSDT.P, 5.csv",
            source_interval="5m",
            legacy_trades_path=ROOT / "outputs" / "multi_asset" / "btc_5m_trades.csv",
        ),
        Experiment(
            asset="btc",
            symbol="BTCUSDT",
            timeframe="1h",
            source_path=ROOT / "data" / "BINANCE_BTCUSDT.P, 60.csv",
            source_interval="1h",
            legacy_trades_path=ROOT / "outputs" / "trades.csv",
        ),
        Experiment(
            asset="btc",
            symbol="BTCUSDT",
            timeframe="12h",
            source_path=ROOT / "data" / "BINANCE_BTCUSDT.P, 60.csv",
            source_interval="1h",
        ),
        Experiment(
            asset="btc",
            symbol="BTCUSDT",
            timeframe="1d",
            source_path=ROOT / "data" / "BINANCE_BTCUSDT.P, 60.csv",
            source_interval="1h",
        ),
        Experiment(
            asset="btc",
            symbol="BTCUSDT",
            timeframe="2d",
            source_path=ROOT / "data" / "BINANCE_BTCUSDT.P, 60.csv",
            source_interval="1h",
        ),
        Experiment(
            asset="btc",
            symbol="BTCUSDT",
            timeframe="1w",
            source_path=ROOT / "data" / "BINANCE_BTCUSDT.P, 60.csv",
            source_interval="1h",
            lookback=260,
        ),
        Experiment(
            asset="sol",
            symbol="SOLUSDT",
            timeframe="5m",
            source_path=ROOT / "data" / "MEXC_SOLUSDT.P, 5 (1).csv",
            source_interval="5m",
            legacy_trades_path=ROOT / "outputs" / "multi_asset" / "sol_5m_trades.csv",
        ),
        Experiment(
            asset="sol",
            symbol="SOLUSDT",
            timeframe="1h",
            source_path=ROOT / "data" / "MEXC_SOLUSDT.P, 5 (1).csv",
            source_interval="5m",
        ),
        Experiment(
            asset="sol",
            symbol="SOLUSDT",
            timeframe="12h",
            source_path=ROOT / "data" / "MEXC_SOLUSDT.P, 5 (1).csv",
            source_interval="5m",
        ),
        Experiment(
            asset="sol",
            symbol="SOLUSDT",
            timeframe="1d",
            source_path=ROOT / "data" / "MEXC_SOLUSDT.P, 5 (1).csv",
            source_interval="5m",
        ),
        Experiment(
            asset="sol",
            symbol="SOLUSDT",
            timeframe="2d",
            source_path=ROOT / "data" / "MEXC_SOLUSDT.P, 5 (1).csv",
            source_interval="5m",
        ),
        Experiment(
            asset="sol",
            symbol="SOLUSDT",
            timeframe="1w",
            source_path=ROOT / "data" / "MEXC_SOLUSDT.P, 5 (1).csv",
            source_interval="5m",
            lookback=260,
        ),
        Experiment(
            asset="zec",
            symbol="ZECUSDT",
            timeframe="5m",
            source_path=ROOT / "data" / "BINANCE_ZECUSDT, 5.csv",
            source_interval="5m",
            legacy_trades_path=ROOT / "outputs" / "multi_asset" / "zec_5m_trades.csv",
        ),
        Experiment(
            asset="zec",
            symbol="ZECUSDT",
            timeframe="1h",
            source_path=ROOT / "data" / "BINANCE_ZECUSDT, 5.csv",
            source_interval="5m",
        ),
        Experiment(
            asset="zec",
            symbol="ZECUSDT",
            timeframe="12h",
            source_path=ROOT / "data" / "BINANCE_ZECUSDT, 5.csv",
            source_interval="5m",
        ),
        Experiment(
            asset="zec",
            symbol="ZECUSDT",
            timeframe="1d",
            source_path=ROOT / "data" / "BINANCE_ZECUSDT, 5.csv",
            source_interval="5m",
        ),
        Experiment(
            asset="zec",
            symbol="ZECUSDT",
            timeframe="2d",
            source_path=ROOT / "data" / "BINANCE_ZECUSDT, 5.csv",
            source_interval="5m",
        ),
        Experiment(
            asset="zec",
            symbol="ZECUSDT",
            timeframe="1w",
            source_path=ROOT / "data" / "BINANCE_ZECUSDT, 5.csv",
            source_interval="5m",
            lookback=260,
        ),
    ]
    allowed_assets = {asset.lower() for asset in assets}
    allowed_timeframes = {timeframe.lower() for timeframe in timeframes}
    return [
        experiment
        for experiment in catalog
        if experiment.asset in allowed_assets and experiment.timeframe.lower() in allowed_timeframes
    ]


def configure_core() -> None:
    core.LOOKBACK = LOOKBACK
    core.PRED_LEN = PRED_LEN
    core.STEP = STEP
    core.SAMPLE_COUNT = SAMPLE_COUNT
    core.TOP_P = TOP_P
    core.TEMP = TEMP
    core.LONG_THRESHOLD = LONG_THRESHOLD
    core.SHORT_THRESHOLD = SHORT_THRESHOLD


def load_predictor() -> tuple[str, KronosPredictor]:
    device = core.choose_device()
    tokenizer = KronosTokenizer.from_pretrained("NeoQuasar/Kronos-Tokenizer-2k")
    model = Kronos.from_pretrained("NeoQuasar/Kronos-mini")
    predictor = KronosPredictor(model, tokenizer, device=device, max_context=2048)
    return device, predictor


def target_history_bars(experiment: Experiment) -> int | None:
    minimum = experiment.lookback + PRED_LEN + 2 + STEP * 120
    caps = {
        "5m": 20_000,
        "1h": 10_000,
        "12h": None,
        "1d": None,
        "2d": None,
        "1w": None,
    }
    cap = caps.get(experiment.timeframe)
    if cap is None:
        return None
    return max(cap, minimum)


def fetch_json(url: str, params: dict[str, Any]) -> Any:
    response = requests.get(url, params=params, timeout=30)
    response.raise_for_status()
    payload = response.json()
    if isinstance(payload, dict) and payload.get("code") is not None:
        raise RuntimeError(f"Binance error: {payload}")
    return payload


def fetch_binance_history(symbol: str, interval: str, market: str, max_bars: int | None) -> pd.DataFrame:
    endpoint = "https://data-api.binance.vision/api/v3/klines"
    if market in {"future", "futures", "linear", "perp", "perpetual", "usdm"}:
        endpoint = "https://fapi.binance.com/fapi/v1/klines"

    limit = 1000
    interval_ms = int(interval_to_timedelta(interval) / pd.Timedelta(milliseconds=1))
    now_ms = int(pd.Timestamp.now(tz="UTC").timestamp() * 1000)
    if max_bars is None:
        start_time_ms = int(pd.Timestamp("2017-01-01", tz="UTC").timestamp() * 1000)
    else:
        start_time_ms = max(
            int(pd.Timestamp("2017-01-01", tz="UTC").timestamp() * 1000),
            now_ms - (max_bars + 8) * interval_ms,
        )

    rows: list[dict[str, Any]] = []
    while True:
        remaining = None if max_bars is None else max_bars - len(rows)
        if remaining is not None and remaining <= 0:
            break
        batch_limit = limit if remaining is None else min(limit, max(1, remaining))
        payload = fetch_json(
            endpoint,
            {
                "symbol": symbol,
                "interval": interval,
                "startTime": start_time_ms,
                "limit": batch_limit,
            },
        )
        if not payload:
            break

        batch_rows = []
        for item in payload:
            if int(item[6]) > now_ms:
                continue
            batch_rows.append(
                {
                    "timestamp": pd.to_datetime(int(item[0]), unit="ms", utc=True).tz_convert(None),
                    "open": float(item[1]),
                    "high": float(item[2]),
                    "low": float(item[3]),
                    "close": float(item[4]),
                    "volume": float(item[5]),
                    "amount": float(item[7]),
                }
            )
        if not batch_rows:
            break
        rows.extend(batch_rows)

        last_open_ms = int(payload[-1][0])
        next_start_ms = last_open_ms + interval_ms
        if next_start_ms <= start_time_ms or len(payload) < batch_limit:
            break
        start_time_ms = next_start_ms
        time.sleep(0.05)

    frame = pd.DataFrame(rows)
    if frame.empty:
        raise ValueError(f"No Binance candles returned for {symbol} {interval}")
    frame = frame.sort_values("timestamp").drop_duplicates("timestamp", keep="last").reset_index(drop=True)
    return frame


def fetch_binance_prices(experiment: Experiment) -> pd.DataFrame:
    fetch_interval = choose_binance_fetch_interval(experiment.timeframe)
    target_bars = target_history_bars(experiment)
    raw_target_bars = (
        None
        if target_bars is None
        else target_bars * interval_ratio(experiment.timeframe, fetch_interval) + 8
    )
    frame = fetch_binance_history(
        symbol=experiment.symbol,
        interval=fetch_interval,
        market=experiment.market,
        max_bars=raw_target_bars,
    )
    if fetch_interval != experiment.timeframe:
        frame = resample_candles(frame, experiment.timeframe)
    if target_bars is not None:
        frame = frame.sort_values("timestamp").tail(target_bars).reset_index(drop=True)
    return frame


def load_prices(experiment: Experiment, use_binance: bool) -> pd.DataFrame:
    if use_binance:
        return fetch_binance_prices(experiment)
    if experiment.source_path is None:
        raise ValueError(f"No local source configured for {experiment.slug}")
    base = core.normalize_columns(pd.read_csv(experiment.source_path))
    if experiment.timeframe == experiment.source_interval:
        return base
    return resample_candles(base, experiment.timeframe)


def rolling_prior_percentile(series: pd.Series, window: int = 365, min_periods: int = 50) -> pd.Series:
    values = series.to_numpy(dtype=float)
    out = np.full(len(values), np.nan)
    for idx, value in enumerate(values):
        history = values[max(0, idx - window) : idx]
        history = history[~np.isnan(history)]
        if len(history) >= min_periods and not np.isnan(value):
            out[idx] = float((history <= value).mean())
    return pd.Series(out, index=series.index)


def rsi(close: pd.Series, window: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(window).mean()
    loss = (-delta.clip(upper=0)).rolling(window).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - 100 / (1 + rs)


def compute_price_features(prices: pd.DataFrame) -> pd.DataFrame:
    df = prices.copy().sort_values("timestamp").reset_index(drop=True)
    df["log_ret"] = np.log(df["close"]).diff()
    df["ret_24"] = df["close"] / df["close"].shift(24) - 1.0
    df["ret_72"] = df["close"] / df["close"].shift(72) - 1.0
    df["vol_24"] = df["log_ret"].rolling(24).std()
    df["vol_72"] = df["log_ret"].rolling(72).std()
    df["sma_24"] = df["close"].rolling(24).mean()
    df["sma_72"] = df["close"].rolling(72).mean()
    df["above_sma_24"] = df["close"] > df["sma_24"]
    df["above_sma_72"] = df["close"] > df["sma_72"]
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
    for column in ["ret_24", "ret_72", "vol_24", "vol_72", "atr_24_pct", "volume_ratio_24", "rsi_14"]:
        df[f"{column}_live_pct"] = rolling_prior_percentile(df[column])
    return df


def read_trades(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, parse_dates=["signal_timestamp", "entry_timestamp", "exit_timestamp"])


def write_config(path: Path, config: dict[str, Any]) -> None:
    path.write_text(json.dumps(config, indent=2, sort_keys=True), encoding="utf-8")


def read_config(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def raw_config(experiment: Experiment, max_signals: int | None, use_binance: bool) -> dict[str, Any]:
    return {
        "asset": experiment.asset,
        "symbol": experiment.symbol,
        "timeframe": experiment.timeframe,
        "source_path": str(experiment.source_path.relative_to(ROOT)) if experiment.source_path else None,
        "source_interval": experiment.source_interval,
        "lookback": experiment.lookback,
        "pred_len": PRED_LEN,
        "step": STEP,
        "sample_count": SAMPLE_COUNT,
        "top_p": TOP_P,
        "temperature": TEMP,
        "long_threshold": LONG_THRESHOLD,
        "short_threshold": SHORT_THRESHOLD,
        "max_signals": max_signals,
        "use_binance": use_binance,
    }


def select_signal_indices(df: pd.DataFrame, experiment: Experiment, max_signals: int | None) -> list[int]:
    last_end_idx = len(df) - PRED_LEN - 1
    indices = list(range(experiment.lookback - 1, last_end_idx + 1, STEP))
    if max_signals is not None:
        return indices[-max_signals:]
    return indices


def compute_raw_trades(
    experiment: Experiment,
    prices: pd.DataFrame,
    predictor: KronosPredictor,
    max_signals: int | None,
) -> pd.DataFrame:
    signal_indices = select_signal_indices(prices, experiment, max_signals)
    rows: list[dict[str, Any]] = []
    print(
        f"{experiment.slug}: rows={len(prices)} range={prices['timestamp'].iloc[0]}->{prices['timestamp'].iloc[-1]} "
        f"signals={len(signal_indices)}",
        flush=True,
    )
    for signal_number, end_idx in enumerate(signal_indices, start=1):
        entry_idx = end_idx + 1
        exit_idx = end_idx + PRED_LEN
        window = prices.iloc[end_idx - experiment.lookback + 1 : end_idx + 1].copy()
        x_timestamp = window["timestamp"].reset_index(drop=True)
        y_timestamp = core.future_timestamps(prices, end_idx)

        current_close = float(prices.loc[end_idx, "close"])
        entry_open = float(prices.loc[entry_idx, "open"])
        exit_close = float(prices.loc[exit_idx, "close"])

        close_samples = core.predict_close_samples(predictor, window, x_timestamp, y_timestamp)
        final_pred_close = close_samples[:, -1]
        upside_probability = float(np.mean(final_pred_close > current_close))
        forecast_return = float(np.mean(final_pred_close / current_close - 1.0))

        if upside_probability >= LONG_THRESHOLD:
            position = 1
        elif upside_probability <= SHORT_THRESHOLD:
            position = -1
        else:
            position = 0

        gross_return = position * (exit_close / entry_open - 1.0) if position else 0.0
        net_return = gross_return - core.ROUND_TRIP_COST if position else 0.0
        rows.append(
            {
                "signal_number": signal_number,
                "signal_timestamp": prices.loc[end_idx, "timestamp"],
                "entry_timestamp": prices.loc[entry_idx, "timestamp"],
                "exit_timestamp": prices.loc[exit_idx, "timestamp"],
                "current_close": current_close,
                "entry_open": entry_open,
                "exit_close": exit_close,
                "mean_final_pred_close": float(np.mean(final_pred_close)),
                "forecast_return": forecast_return,
                "upside_probability": upside_probability,
                "position": position,
                "gross_return": gross_return,
                "net_return": net_return,
            }
        )
        if signal_number == 1 or signal_number % 25 == 0 or signal_number == len(signal_indices):
            print(
                f"{experiment.slug}: {signal_number}/{len(signal_indices)} "
                f"{prices.loc[end_idx, 'timestamp']} p_up={upside_probability:.2f} "
                f"pos={position:+d} net={net_return:.4%}",
                flush=True,
            )
    return pd.DataFrame(rows)


def ensure_raw_trades(
    experiment: Experiment,
    predictor: KronosPredictor,
    force: bool,
    max_signals: int | None,
    use_binance: bool,
) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    run_dir = RAW_DIR / experiment.slug
    run_dir.mkdir(parents=True, exist_ok=True)
    trades_path = run_dir / "trades.csv"
    prices_path = run_dir / "prices.csv"
    config_path = run_dir / "config.json"
    expected_config = raw_config(experiment, max_signals, use_binance)

    cached_config = read_config(config_path)
    if not force and trades_path.exists() and prices_path.exists() and cached_config == expected_config:
        source = "binance_cache" if use_binance else "cache"
        return read_trades(trades_path), pd.read_csv(prices_path, parse_dates=["timestamp"]), source

    prices = load_prices(experiment, use_binance)
    prices.to_csv(prices_path, index=False)

    if (
        not use_binance
        and not force
        and experiment.legacy_trades_path
        and experiment.legacy_trades_path.exists()
        and max_signals is None
    ):
        shutil.copyfile(experiment.legacy_trades_path, trades_path)
        write_config(config_path, expected_config)
        return read_trades(trades_path), prices, "legacy"

    trades = compute_raw_trades(experiment, prices, predictor, max_signals)
    trades.to_csv(trades_path, index=False)
    write_config(config_path, expected_config)
    return trades, prices, "binance_fresh" if use_binance else "fresh"


def enrich_trades(experiment: Experiment, trades: pd.DataFrame, prices: pd.DataFrame) -> pd.DataFrame:
    features = compute_price_features(prices)
    feature_cols = [
        "timestamp",
        "ret_24",
        "ret_72",
        "vol_24",
        "vol_72",
        "atr_24_pct",
        "volume_ratio_24",
        "rsi_14",
        "above_sma_24",
        "above_sma_72",
        "ret_24_live_pct",
        "ret_72_live_pct",
        "vol_24_live_pct",
        "vol_72_live_pct",
        "atr_24_pct_live_pct",
        "volume_ratio_24_live_pct",
        "rsi_14_live_pct",
    ]
    out = trades.merge(
        features[feature_cols],
        left_on="signal_timestamp",
        right_on="timestamp",
        how="left",
    ).drop(columns=["timestamp"])
    out["asset"] = experiment.asset
    out["timeframe"] = experiment.timeframe
    out["raw_p"] = out["upside_probability"].astype(float)
    out["pred_ret"] = out["forecast_return"].astype(float)
    out["abs_pred_ret"] = out["pred_ret"].abs()
    out["actual_up_current"] = out["exit_close"] > out["current_close"]
    out["actual_up_entry"] = out["exit_close"] > out["entry_open"]
    out["forecast_direction_correct"] = (out["pred_ret"] >= 0.0) == out["actual_up_current"]
    out["signal_direction_correct"] = np.where(
        out["position"] == 1,
        out["actual_up_entry"],
        np.where(out["position"] == -1, ~out["actual_up_entry"], False),
    )
    signal_time = pd.to_datetime(out["signal_timestamp"])
    out["hour"] = signal_time.dt.hour.astype(float)
    out["weekday_num"] = signal_time.dt.weekday.astype(float)
    out = out.sort_values("signal_timestamp").reset_index(drop=True)
    return out


def add_splits(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    n = len(out)
    out["split"] = np.where(
        np.arange(n) < int(n * 0.60),
        "train",
        np.where(np.arange(n) < int(n * 0.80), "validation", "test"),
    )
    return out


def fit_scores(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    scored = df.copy()
    score_cols = ["score_raw"]
    scored["score_raw"] = scored["raw_p"]

    train = scored[scored["split"] == "train"].copy()
    y_train = train["actual_up_entry"].astype(int)

    try:
        iso = IsotonicRegression(out_of_bounds="clip")
        iso.fit(train["raw_p"], y_train)
        scored["score_iso"] = iso.predict(scored["raw_p"])
        score_cols.append("score_iso")
    except Exception:
        pass

    feature_cols = [
        "raw_p",
        "pred_ret",
        "abs_pred_ret",
        "hour",
        "weekday_num",
        "ret_24",
        "vol_24",
        "atr_24_pct",
        "volume_ratio_24",
        "rsi_14",
        "above_sma_24",
        "above_sma_72",
        "ret_24_live_pct",
        "vol_24_live_pct",
        "atr_24_pct_live_pct",
    ]
    if len(train) >= 25 and y_train.nunique() >= 2:
        train_x = train[feature_cols].copy()
        fill_values = train_x.median(numeric_only=True)
        full_x = scored[feature_cols].copy().fillna(fill_values)
        try:
            logit = make_pipeline(
                StandardScaler(),
                LogisticRegression(C=0.25, solver="lbfgs", max_iter=2000),
            )
            logit.fit(train_x.fillna(fill_values), y_train)
            scored["score_logit"] = logit.predict_proba(full_x)[:, 1]
            score_cols.append("score_logit")
        except Exception:
            pass

    return scored, score_cols


def evaluate(
    split_df: pd.DataFrame,
    score_col: str,
    side: str,
    long_threshold: float,
    short_threshold: float,
    min_abs_pred_ret: float,
) -> dict[str, float | int]:
    eligible = split_df[split_df["abs_pred_ret"] >= min_abs_pred_ret].copy()
    if eligible.empty:
        return {
            "trades": 0,
            "accuracy": 0.0,
            "avg_directional_edge": 0.0,
            "avg_original_net_return": 0.0,
        }

    score = eligible[score_col]
    position = np.zeros(len(eligible), dtype=int)
    if side in {"both", "long"}:
        position = np.where(score >= long_threshold, 1, position)
    if side in {"both", "short"}:
        position = np.where(score <= short_threshold, -1, position)
    eligible["position"] = position
    taken = eligible[eligible["position"] != 0].copy()
    if taken.empty:
        return {
            "trades": 0,
            "accuracy": 0.0,
            "avg_directional_edge": 0.0,
            "avg_original_net_return": 0.0,
        }

    actual_up = taken["actual_up_entry"].to_numpy(dtype=bool)
    direction_correct = np.where(taken["position"] == 1, actual_up, ~actual_up)
    edge = np.where(
        taken["position"] == 1,
        taken["exit_close"] / taken["entry_open"] - 1.0,
        1.0 - taken["exit_close"] / taken["entry_open"],
    )
    return {
        "trades": int(len(taken)),
        "accuracy": float(np.mean(direction_correct)),
        "avg_directional_edge": float(np.mean(edge)),
        "avg_original_net_return": float(taken["net_return"].mean()),
    }


def search_best(df: pd.DataFrame) -> pd.Series:
    scored, score_cols = fit_scores(add_splits(df))
    train = scored[scored["split"] == "train"].copy()
    validation = scored[scored["split"] == "validation"].copy()
    test = scored[scored["split"] == "test"].copy()

    rows = []
    for score_col, side, long_t, short_t, min_abs_pred_ret in product(
        score_cols,
        ["both", "long", "short"],
        [0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85],
        [0.45, 0.40, 0.35, 0.30, 0.25, 0.20, 0.15],
        [0.0, 0.001, 0.002, 0.005, 0.01],
    ):
        train_result = evaluate(train, score_col, side, long_t, short_t, min_abs_pred_ret)
        validation_result = evaluate(validation, score_col, side, long_t, short_t, min_abs_pred_ret)
        test_result = evaluate(test, score_col, side, long_t, short_t, min_abs_pred_ret)
        rows.append(
            {
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
        candidates = results[
            (results["validation_trades"] >= 5)
            & (results["test_trades"] >= 5)
        ].copy()
    if candidates.empty:
        candidates = results.copy()
    return candidates.sort_values(
        ["validation_accuracy", "test_accuracy", "validation_trades", "train_accuracy"],
        ascending=[False, False, False, False],
    ).iloc[0]


def summarize_experiment(
    experiment: Experiment,
    trades: pd.DataFrame,
    prices: pd.DataFrame,
    raw_source: str,
) -> dict[str, Any]:
    traded = trades[trades["position"] != 0].copy()
    horizon = interval_to_timedelta(experiment.timeframe) * PRED_LEN
    if traded.empty:
        total_return = 0.0
        win_rate = 0.0
        max_drawdown = 0.0
        threshold_accuracy = 0.0
    else:
        equity = (1.0 + traded["net_return"]).cumprod()
        total_return = float(equity.iloc[-1] - 1.0)
        win_rate = float((traded["net_return"] > 0).mean())
        max_drawdown = core.max_drawdown(equity)
        actual_up_entry = traded["exit_close"] > traded["entry_open"]
        threshold_accuracy = float(
            np.where(traded["position"] == 1, actual_up_entry, ~actual_up_entry).mean()
        )
    return {
        "slug": experiment.slug,
        "asset": experiment.asset,
        "timeframe": experiment.timeframe,
        "source_interval": experiment.source_interval,
        "source_path": str(experiment.source_path.relative_to(ROOT)) if experiment.source_path else "binance",
        "raw_source": raw_source,
        "sample_count": SAMPLE_COUNT,
        "rows": len(prices),
        "signals": len(trades),
        "traded_signals": int(len(traded)),
        "coverage": float(len(traded) / len(trades)) if len(trades) else 0.0,
        "forecast_direction_accuracy": float(
            ((trades["forecast_return"] >= 0.0) == (trades["exit_close"] > trades["current_close"])).mean()
        )
        if len(trades)
        else 0.0,
        "threshold_signal_accuracy": threshold_accuracy,
        "net_win_rate": win_rate,
        "total_return": total_return,
        "max_drawdown": max_drawdown,
        "pred_len_bars": PRED_LEN,
        "approx_horizon_hours": float(horizon / pd.Timedelta(hours=1)),
    }


def main() -> None:
    global SAMPLE_COUNT
    args = parse_args()
    if args.sample_count is not None:
        if args.sample_count < 1:
            raise SystemExit("--sample-count must be >= 1")
        SAMPLE_COUNT = int(args.sample_count)
    configure_core()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    experiments = build_experiments(args.assets, args.timeframes)
    if args.max_experiments is not None:
        experiments = experiments[: args.max_experiments]
    if not experiments:
        raise SystemExit("No experiments matched the requested assets/timeframes.")

    core.set_seed(SEED)
    device, predictor = load_predictor()
    print(f"device: {device}", flush=True)
    print(
        f"use_binance={args.use_binance} lookback={LOOKBACK} pred_len={PRED_LEN} step={STEP} sample_count={SAMPLE_COUNT} "
        f"long_threshold={LONG_THRESHOLD:.2f} short_threshold={SHORT_THRESHOLD:.2f}",
        flush=True,
    )

    summary_rows: list[dict[str, Any]] = []
    leaderboard_rows: list[dict[str, Any]] = []
    skipped_rows: list[dict[str, Any]] = []

    for experiment in experiments:
        try:
            trades, prices, raw_source = ensure_raw_trades(
                experiment=experiment,
                predictor=predictor,
                force=args.force,
                max_signals=args.max_signals,
                use_binance=args.use_binance,
            )
            min_rows = experiment.lookback + PRED_LEN + 2
            if len(prices) < min_rows or len(trades) < 20:
                skipped_rows.append(
                    {
                        "slug": experiment.slug,
                        "asset": experiment.asset,
                        "timeframe": experiment.timeframe,
                        "reason": f"not_enough_data rows={len(prices)} trades={len(trades)} min_rows={min_rows}",
                    }
                )
                continue

            enriched = enrich_trades(experiment, trades, prices)
            summary_rows.append(summarize_experiment(experiment, trades, prices, raw_source))
            best = search_best(enriched)
            leaderboard_rows.append(
                {
                    "slug": experiment.slug,
                    "asset": experiment.asset,
                    "timeframe": experiment.timeframe,
                    "sample_count": SAMPLE_COUNT,
                    "score_col": best["score_col"],
                    "side": best["side"],
                    "long_threshold": best["long_threshold"],
                    "short_threshold": best["short_threshold"],
                    "min_abs_pred_ret": best["min_abs_pred_ret"],
                    "train_trades": best["train_trades"],
                    "train_accuracy": best["train_accuracy"],
                    "validation_trades": best["validation_trades"],
                    "validation_accuracy": best["validation_accuracy"],
                    "test_trades": best["test_trades"],
                    "test_accuracy": best["test_accuracy"],
                    "test_avg_directional_edge": best["test_avg_directional_edge"],
                    "test_avg_original_net_return": best["test_avg_original_net_return"],
                    "raw_forecast_direction_accuracy": summary_rows[-1]["forecast_direction_accuracy"],
                    "raw_threshold_signal_accuracy": summary_rows[-1]["threshold_signal_accuracy"],
                    "raw_total_return": summary_rows[-1]["total_return"],
                    "raw_max_drawdown": summary_rows[-1]["max_drawdown"],
                    "approx_horizon_hours": summary_rows[-1]["approx_horizon_hours"],
                }
            )
        except Exception as exc:
            skipped_rows.append(
                {
                    "slug": experiment.slug,
                    "asset": experiment.asset,
                    "timeframe": experiment.timeframe,
                    "reason": f"{type(exc).__name__}: {exc}",
                }
            )

    summary_df = pd.DataFrame(summary_rows)
    if not summary_df.empty:
        summary_df = summary_df.sort_values(["asset", "timeframe"]).reset_index(drop=True)

    leaderboard_df = pd.DataFrame(leaderboard_rows)
    if not leaderboard_df.empty:
        leaderboard_df = leaderboard_df.sort_values(
            ["test_accuracy", "validation_accuracy", "test_trades"],
            ascending=[False, False, False],
        ).reset_index(drop=True)

    skipped_df = pd.DataFrame(skipped_rows).reset_index(drop=True)

    summary_df.to_csv(SUMMARY_PATH, index=False)
    leaderboard_df.to_csv(LEADERBOARD_PATH, index=False)
    skipped_df.to_csv(SKIPPED_PATH, index=False)

    if not summary_df.empty:
        print()
        print("Raw summary")
        print(
            summary_df.to_string(
                index=False,
                formatters={
                    "coverage": "{:.2%}".format,
                    "forecast_direction_accuracy": "{:.2%}".format,
                    "threshold_signal_accuracy": "{:.2%}".format,
                    "net_win_rate": "{:.2%}".format,
                    "total_return": "{:.2%}".format,
                    "max_drawdown": "{:.2%}".format,
                    "approx_horizon_hours": "{:.1f}".format,
                },
            )
        )

    if not leaderboard_df.empty:
        print()
        print("Best calibrated accuracy by asset/timeframe")
        print(
            leaderboard_df.to_string(
                index=False,
                formatters={
                    "train_accuracy": "{:.2%}".format,
                    "validation_accuracy": "{:.2%}".format,
                    "test_accuracy": "{:.2%}".format,
                    "test_avg_directional_edge": "{:.4%}".format,
                    "test_avg_original_net_return": "{:.4%}".format,
                    "raw_forecast_direction_accuracy": "{:.2%}".format,
                    "raw_threshold_signal_accuracy": "{:.2%}".format,
                    "raw_total_return": "{:.2%}".format,
                    "raw_max_drawdown": "{:.2%}".format,
                    "approx_horizon_hours": "{:.1f}".format,
                },
            )
        )

    if not skipped_df.empty:
        print()
        print("Skipped")
        print(skipped_df.to_string(index=False))

    print()
    print(f"saved_summary: {SUMMARY_PATH}")
    print(f"saved_leaderboard: {LEADERBOARD_PATH}")
    print(f"saved_skipped: {SKIPPED_PATH}")


if __name__ == "__main__":
    main()
