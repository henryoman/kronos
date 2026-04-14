# /// script
# requires-python = ">=3.14"
# dependencies = [
#   "einops",
#   "flask",
#   "huggingface_hub",
#   "numpy",
#   "pandas",
#   "requests",
#   "safetensors",
#   "torch",
#   "tqdm",
# ]
# ///

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import requests
import torch
from flask import Flask, jsonify, render_template, request

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from model import Kronos, KronosPredictor, KronosTokenizer  # noqa: E402

APP_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG = APP_DIR / "config.json"

HTTP_TIMEOUT = 20
ALPHA_VANTAGE_URL = "https://www.alphavantage.co/query"
PRICE_COLUMNS = ["open", "high", "low", "close"]
FEATURE_COLUMNS = ["open", "high", "low", "close", "volume", "amount"]
BINANCE_NATIVE_INTERVALS = {
    "1s",
    "1m",
    "3m",
    "5m",
    "15m",
    "30m",
    "1h",
    "2h",
    "4h",
    "6h",
    "8h",
    "12h",
    "1d",
    "3d",
    "1w",
    "1M",
}

_MODEL_CACHE: dict[str, Any] = {}
_ALPHA_VANTAGE_CACHE: dict[tuple[str, str, str, str], Any] = {}
_ALPHA_VANTAGE_LAST_REQUEST_AT = 0.0


@dataclass(frozen=True)
class Demo:
    id: str
    label: str
    provider: str
    market: str
    symbol: str
    interval: str
    lookback: int
    pred_len: int
    sample_count: int
    top_p: float
    temperature: float
    long_threshold: float
    short_threshold: float
    strategy: dict[str, Any]
    source_path: str | None
    source_interval: str


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        config = json.load(handle)
    if not config.get("demos"):
        raise ValueError("Config must include at least one demo.")
    return config


def merge_demo(raw: dict[str, Any], defaults: dict[str, Any]) -> Demo:
    merged = {**defaults, **raw}
    return Demo(
        id=str(merged["id"]),
        label=str(merged.get("label") or merged["id"]),
        provider=str(merged["provider"]).lower(),
        market=str(merged.get("market", "spot")).lower(),
        symbol=str(merged["symbol"]).upper(),
        interval=str(merged["interval"]),
        lookback=int(merged["lookback"]),
        pred_len=int(merged["pred_len"]),
        sample_count=int(merged["sample_count"]),
        top_p=float(merged["top_p"]),
        temperature=float(merged["temperature"]),
        long_threshold=float(merged["long_threshold"]),
        short_threshold=float(merged["short_threshold"]),
        strategy=dict(merged.get("strategy", {})),
        source_path=str(merged["source_path"]) if merged.get("source_path") else None,
        source_interval=str(merged.get("source_interval") or merged["interval"]),
    )


def get_demo(config: dict[str, Any], demo_id: str) -> Demo:
    defaults = config.get("defaults", {})
    for raw in config["demos"]:
        if raw["id"] == demo_id:
            return merge_demo(raw, defaults)
    raise KeyError(f"Unknown demo id: {demo_id}")


def interval_to_timedelta(interval: str) -> pd.Timedelta:
    value = interval.strip()
    lower = value.lower()
    if lower.endswith("m"):
        return pd.Timedelta(minutes=int(lower[:-1]))
    if lower.endswith("h"):
        return pd.Timedelta(hours=int(lower[:-1]))
    if lower.endswith("d"):
        return pd.Timedelta(days=int(lower[:-1]))
    if lower.endswith("w"):
        return pd.Timedelta(weeks=int(lower[:-1]))
    if value.isdigit():
        return pd.Timedelta(minutes=int(value))
    if value == "D":
        return pd.Timedelta(days=1)
    if value == "W":
        return pd.Timedelta(weeks=1)
    raise ValueError(f"Unsupported interval: {interval}")


def interval_to_binance(interval: str) -> str:
    value = interval.strip()
    if value.isdigit():
        minutes = int(value)
        if minutes % 60 == 0:
            return f"{minutes // 60}h"
        return f"{minutes}m"
    return value.lower()


def normalize_interval_value(interval: str) -> tuple[int, str]:
    value = interval.strip()
    lower = value.lower()
    if lower.endswith(("m", "h", "d", "w")):
        return int(lower[:-1]), lower[-1]
    if value == "D":
        return 1, "d"
    if value == "W":
        return 1, "w"
    raise ValueError(f"Unsupported interval: {interval}")


def choose_binance_fetch_interval(interval: str) -> str:
    normalized = interval_to_binance(interval)
    if normalized in BINANCE_NATIVE_INTERVALS:
        return normalized

    value, unit = normalize_interval_value(normalized)
    if unit == "d" and value == 2:
        return "1d"

    raise ValueError(f"Unsupported Binance interval: {interval}")


def interval_ratio(target_interval: str, source_interval: str) -> int:
    target_delta = interval_to_timedelta(target_interval)
    source_delta = interval_to_timedelta(source_interval)
    ratio = target_delta / source_delta
    rounded = int(round(ratio))
    if rounded <= 0 or abs(ratio - rounded) > 1e-9:
        raise ValueError(f"Cannot align {target_interval} from {source_interval}")
    return rounded


def resample_candles(df: pd.DataFrame, target_interval: str) -> pd.DataFrame:
    if df.empty:
        return df

    value, unit = normalize_interval_value(target_interval)
    if unit == "m":
        freq = f"{value}min"
    elif unit == "h":
        freq = f"{value}h"
    elif unit == "d":
        freq = f"{value}D"
    elif unit == "w":
        freq = f"{value}W"
    else:
        raise ValueError(f"Unsupported resample interval: {target_interval}")
    aggregated = (
        df.sort_values("timestamp")
        .set_index("timestamp")
        .resample(freq, label="left", closed="left")
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
        .dropna(subset=PRICE_COLUMNS)
        .reset_index()
    )
    return aggregated


def interval_to_bybit(interval: str) -> str:
    value = interval.strip()
    lower = value.lower()
    if lower.endswith("m"):
        return lower[:-1]
    if lower.endswith("h"):
        return str(int(lower[:-1]) * 60)
    if lower.endswith("d"):
        days = int(lower[:-1])
        return "D" if days == 1 else str(days * 1440)
    if lower.endswith("w"):
        weeks = int(lower[:-1])
        return "W" if weeks == 1 else str(weeks * 10080)
    if value.isdigit() or value in {"D", "W", "M"}:
        return value
    raise ValueError(f"Unsupported Bybit interval: {interval}")


def request_json(url: str, params: dict[str, Any]) -> Any:
    response = requests.get(url, params=params, timeout=HTTP_TIMEOUT)
    response.raise_for_status()
    return response.json()


def get_alpha_vantage_api_key() -> str:
    env_key = os.getenv("ALPHA_VANTAGE_API_KEY")
    if env_key:
        return env_key

    env_path = ROOT / ".env"
    if env_path.exists():
        for line in env_path.read_text(encoding="utf-8").splitlines():
            if line.startswith("ALPHA_VANTAGE_API_KEY="):
                value = line.split("=", 1)[1].strip()
                if value:
                    return value
    raise ValueError("ALPHA_VANTAGE_API_KEY is not configured.")


def request_alpha_vantage(params: dict[str, Any]) -> dict[str, Any]:
    global _ALPHA_VANTAGE_LAST_REQUEST_AT

    function = str(params.get("function", ""))
    symbol = str(params.get("symbol", ""))
    interval = str(params.get("interval", ""))
    outputsize = str(params.get("outputsize", ""))
    cache_key = (function, symbol, interval, outputsize)
    cached = _ALPHA_VANTAGE_CACHE.get(cache_key)
    if cached is not None:
        return cached

    elapsed = time.time() - _ALPHA_VANTAGE_LAST_REQUEST_AT
    if elapsed < 1.05:
        time.sleep(1.05 - elapsed)

    payload = request_json(
        ALPHA_VANTAGE_URL,
        {
            **params,
            "apikey": get_alpha_vantage_api_key(),
            "datatype": "json",
        },
    )
    _ALPHA_VANTAGE_LAST_REQUEST_AT = time.time()

    if not payload:
        raise ValueError("Alpha Vantage returned an empty response.")
    if "Error Message" in payload:
        raise ValueError(f"Alpha Vantage error: {payload['Error Message']}")
    if "Information" in payload:
        raise ValueError(f"Alpha Vantage info: {payload['Information']}")

    _ALPHA_VANTAGE_CACHE[cache_key] = payload
    return payload


def alpha_vantage_series_to_frame(payload: dict[str, Any], series_key: str) -> pd.DataFrame:
    series = payload.get(series_key)
    if not isinstance(series, dict) or not series:
        raise ValueError(f"Alpha Vantage response missing {series_key}.")

    rows = []
    for timestamp, values in series.items():
        rows.append(
            {
                "timestamp": pd.to_datetime(timestamp, utc=True),
                "open": float(values["1. open"]),
                "high": float(values["2. high"]),
                "low": float(values["3. low"]),
                "close": float(values["4. close"]),
                "volume": float(values.get("5. volume", 0.0)),
                "amount": 0.0,
            }
        )
    return pd.DataFrame(rows)


def resolve_source_path(path_value: str) -> Path:
    path = Path(path_value).expanduser()
    if path.is_absolute():
        return path
    return ROOT / path


def matching_column(columns: pd.Index, candidates: list[str]) -> str | None:
    lowered = {str(column).lower(): str(column) for column in columns}
    for candidate in candidates:
        match = lowered.get(candidate.lower())
        if match:
            return match
    return None


def required_column(columns: pd.Index, candidates: list[str], label: str) -> str:
    match = matching_column(columns, candidates)
    if match is None:
        joined = ", ".join(candidates)
        raise ValueError(f"CSV source must include {label}. Tried: {joined}")
    return match


def fetch_csv(demo: Demo, limit: int) -> pd.DataFrame:
    if not demo.source_path:
        raise ValueError(f"CSV demo {demo.id} is missing source_path.")

    source_path = resolve_source_path(demo.source_path)
    if not source_path.exists():
        raise ValueError(f"CSV source not found: {source_path}")

    frame = pd.read_csv(source_path)
    rename_map: dict[str, str] = {}
    timestamp_column = required_column(frame.columns, ["timestamp", "datetime", "date", "time"], "a timestamp column")
    rename_map[timestamp_column] = "timestamp"
    rename_map[required_column(frame.columns, ["open"], "an open column")] = "open"
    rename_map[required_column(frame.columns, ["high"], "a high column")] = "high"
    rename_map[required_column(frame.columns, ["low"], "a low column")] = "low"
    rename_map[required_column(frame.columns, ["close"], "a close column")] = "close"

    volume_column = matching_column(frame.columns, ["volume", "vol"])
    if volume_column:
        rename_map[volume_column] = "volume"

    amount_column = matching_column(frame.columns, ["amount", "quote_volume", "quotevolume", "turnover"])
    if amount_column:
        rename_map[amount_column] = "amount"

    symbol_column = matching_column(frame.columns, ["symbol", "ticker"])
    frame = frame.rename(columns=rename_map)
    if symbol_column:
        matches = frame[symbol_column].astype(str).str.upper() == demo.symbol.upper()
        if not matches.any():
            raise ValueError(f"CSV source {source_path} does not contain symbol {demo.symbol}.")
        frame = frame.loc[matches]

    if "volume" not in frame.columns:
        frame["volume"] = 0.0
    if "amount" not in frame.columns:
        frame["amount"] = 0.0

    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True, errors="coerce")
    source_interval = demo.source_interval or demo.interval
    if source_interval != demo.interval:
        raw_limit = limit * interval_ratio(demo.interval, source_interval) + 8
        frame = frame.sort_values("timestamp").tail(raw_limit).reset_index(drop=True)
        frame = resample_candles(frame, demo.interval)
    else:
        frame = frame.sort_values("timestamp").tail(limit).reset_index(drop=True)
    return frame


def fetch_binance(demo: Demo, limit: int) -> pd.DataFrame:
    endpoint = "https://data-api.binance.vision/api/v3/klines"
    if demo.market in {"future", "futures", "linear", "perp", "perpetual", "usdm"}:
        endpoint = "https://fapi.binance.com/fapi/v1/klines"

    requested_interval = interval_to_binance(demo.interval)
    fetch_interval = choose_binance_fetch_interval(demo.interval)
    raw_limit = limit
    if fetch_interval != requested_interval:
        raw_limit = min(max(limit * interval_ratio(demo.interval, fetch_interval) + 8, 1), 1000)

    payload = request_json(
        endpoint,
        {
            "symbol": demo.symbol,
            "interval": fetch_interval,
            "limit": min(max(raw_limit, 1), 1000),
        },
    )
    rows = []
    now_ms = int(time.time() * 1000)
    for item in payload:
        if int(item[6]) > now_ms:
            continue
        rows.append(
            {
                "timestamp": pd.to_datetime(int(item[0]), unit="ms", utc=True),
                "open": float(item[1]),
                "high": float(item[2]),
                "low": float(item[3]),
                "close": float(item[4]),
                "volume": float(item[5]),
                "amount": float(item[7]),
            }
        )
    frame = pd.DataFrame(rows)
    if fetch_interval != requested_interval:
        frame = resample_candles(frame, demo.interval)
    return frame


def fetch_bybit(demo: Demo, limit: int) -> pd.DataFrame:
    payload = request_json(
        "https://api.bybit.com/v5/market/kline",
        {
            "category": demo.market,
            "symbol": demo.symbol,
            "interval": interval_to_bybit(demo.interval),
            "limit": min(max(limit, 1), 1000),
        },
    )
    if payload.get("retCode") != 0:
        raise RuntimeError(f"Bybit error: {payload.get('retMsg', payload)}")

    rows = []
    for item in payload["result"]["list"]:
        rows.append(
            {
                "timestamp": pd.to_datetime(int(item[0]), unit="ms", utc=True),
                "open": float(item[1]),
                "high": float(item[2]),
                "low": float(item[3]),
                "close": float(item[4]),
                "volume": float(item[5]),
                "amount": float(item[6]),
            }
        )
    return pd.DataFrame(rows)


def fetch_alpha_vantage(demo: Demo, limit: int) -> pd.DataFrame:
    upper_interval = demo.interval.upper()
    if upper_interval in {"1D", "2D"}:
        payload = request_alpha_vantage(
            {
                "function": "TIME_SERIES_DAILY",
                "symbol": demo.symbol,
                "outputsize": "compact",
            }
        )
        frame = alpha_vantage_series_to_frame(payload, "Time Series (Daily)")
        if upper_interval == "2D":
            raw_limit = limit * interval_ratio(demo.interval, "1d") + 8
            frame = frame.sort_values("timestamp").tail(raw_limit).reset_index(drop=True)
            frame = resample_candles(frame, demo.interval)
        return frame

    if upper_interval == "1W":
        payload = request_alpha_vantage(
            {
                "function": "TIME_SERIES_WEEKLY",
                "symbol": demo.symbol,
            }
        )
        return alpha_vantage_series_to_frame(payload, "Weekly Time Series")

    raise ValueError(
        f"Unsupported Alpha Vantage interval: {demo.interval}. "
        "Current key supports daily, 2d, and weekly demo feeds."
    )


def normalize_candles(df: pd.DataFrame, lookback: int) -> pd.DataFrame:
    if df.empty:
        raise ValueError("No candles returned by provider.")

    out = df.copy()
    out["timestamp"] = pd.to_datetime(out["timestamp"], utc=True, errors="coerce")
    out = out.dropna(subset=["timestamp", *PRICE_COLUMNS])
    out = out.sort_values("timestamp").drop_duplicates("timestamp", keep="last")

    for column in FEATURE_COLUMNS:
        if column not in out.columns:
            out[column] = 0.0
        out[column] = pd.to_numeric(out[column], errors="coerce").fillna(0.0)

    out = out.tail(lookback).reset_index(drop=True)
    if len(out) < lookback:
        raise ValueError(f"Need {lookback} closed candles, got {len(out)}.")
    return out


def fetch_candles(demo: Demo, extra: int = 8) -> pd.DataFrame:
    limit = demo.lookback + extra
    if demo.provider == "binance":
        raw = fetch_binance(demo, limit)
    elif demo.provider == "bybit":
        raw = fetch_bybit(demo, limit)
    elif demo.provider == "alphavantage":
        raw = fetch_alpha_vantage(demo, limit)
    elif demo.provider == "csv":
        raw = fetch_csv(demo, limit)
    else:
        raise ValueError(f"Unsupported provider: {demo.provider}")
    return normalize_candles(raw, demo.lookback)


def choose_device(requested: str) -> str:
    if requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def get_predictor(config: dict[str, Any]) -> KronosPredictor:
    model_config = config.get("model", {})
    tokenizer_id = model_config.get("tokenizer_id", "NeoQuasar/Kronos-Tokenizer-2k")
    model_id = model_config.get("model_id", "NeoQuasar/Kronos-mini")
    device = choose_device(str(model_config.get("device", "auto")))
    max_context = int(model_config.get("max_context", 2048))
    key = f"{tokenizer_id}|{model_id}|{device}|{max_context}"

    if key not in _MODEL_CACHE:
        tokenizer = KronosTokenizer.from_pretrained(tokenizer_id)
        model = Kronos.from_pretrained(model_id)
        model.eval()
        _MODEL_CACHE[key] = KronosPredictor(
            model=model,
            tokenizer=tokenizer,
            device=device,
            max_context=max_context,
        )
    return _MODEL_CACHE[key]


def future_timestamps(history: pd.DataFrame, pred_len: int, interval: str) -> pd.Series:
    timestamps = pd.to_datetime(history["timestamp"], utc=True)
    if len(timestamps) >= 2:
        delta = timestamps.diff().dropna().median()
    else:
        delta = interval_to_timedelta(interval)
    if pd.isna(delta) or delta <= pd.Timedelta(0):
        delta = interval_to_timedelta(interval)
    start = timestamps.iloc[-1] + delta
    future = pd.date_range(start=start, periods=pred_len, freq=delta)
    return pd.Series(future).dt.tz_convert(None)


def dataframe_for_model(candles: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    model_df = candles[FEATURE_COLUMNS].copy()
    timestamps = pd.to_datetime(candles["timestamp"], utc=True).dt.tz_convert(None)
    return model_df, timestamps


def run_forecast(config: dict[str, Any], demo: Demo) -> dict[str, Any]:
    candles = fetch_candles(demo)
    model_df, x_timestamps = dataframe_for_model(candles)
    y_timestamps = future_timestamps(candles, demo.pred_len, demo.interval)
    predictor = get_predictor(config)

    repeated_dfs = [model_df] * demo.sample_count
    repeated_x = [x_timestamps] * demo.sample_count
    repeated_y = [y_timestamps] * demo.sample_count
    predictions = predictor.predict_batch(
        repeated_dfs,
        repeated_x,
        repeated_y,
        pred_len=demo.pred_len,
        T=demo.temperature,
        top_k=0,
        top_p=demo.top_p,
        sample_count=1,
        verbose=False,
    )

    current_close = float(candles["close"].iloc[-1])
    final_closes = np.array([float(pred["close"].iloc[-1]) for pred in predictions], dtype=float)
    p_up = float(np.mean(final_closes > current_close))
    forecast_return = float(np.mean(final_closes / current_close - 1.0))

    if p_up >= demo.long_threshold:
        side = "long"
    elif p_up <= demo.short_threshold:
        side = "short"
    else:
        side = "flat"

    mean_prediction = pd.concat(predictions).groupby(level=0).mean(numeric_only=True)
    horizon_delta = interval_to_timedelta(demo.interval) * demo.pred_len

    return {
        "demo": {
            "id": demo.id,
            "label": demo.label,
            "provider": demo.provider,
            "market": demo.market,
            "symbol": demo.symbol,
            "interval": demo.interval,
        },
        "model": {
            "lookback": demo.lookback,
            "pred_len": demo.pred_len,
            "sample_count": demo.sample_count,
            "top_p": demo.top_p,
            "temperature": demo.temperature,
        },
        "strategy": {
            "mode": demo.strategy.get("mode", "threshold"),
            "long_threshold": demo.long_threshold,
            "short_threshold": demo.short_threshold,
            "side": side,
        },
        "forecast": {
            "current_close": current_close,
            "mean_final_close": float(np.mean(final_closes)),
            "sample_final_closes": [float(value) for value in final_closes],
            "p_up": p_up,
            "forecast_return": forecast_return,
            "horizon_label": format_timedelta(horizon_delta),
            "last_candle_at": iso_utc(candles["timestamp"].iloc[-1]),
            "forecast_until": iso_utc(y_timestamps.iloc[-1]),
        },
        "candles": candles_to_records(candles),
        "prediction": prediction_to_records(mean_prediction),
    }


def format_timedelta(delta: pd.Timedelta) -> str:
    total_seconds = int(delta.total_seconds())
    if total_seconds % 86400 == 0:
        days = total_seconds // 86400
        return f"{days}d"
    if total_seconds % 3600 == 0:
        hours = total_seconds // 3600
        return f"{hours}h"
    if total_seconds % 60 == 0:
        minutes = total_seconds // 60
        return f"{minutes}m"
    return f"{total_seconds}s"


def iso_utc(value: Any) -> str:
    ts = pd.Timestamp(value)
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    else:
        ts = ts.tz_convert("UTC")
    return ts.isoformat().replace("+00:00", "Z")


def unix_seconds(value: Any) -> int:
    ts = pd.Timestamp(value)
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    else:
        ts = ts.tz_convert("UTC")
    return math.floor(ts.timestamp())


def candles_to_records(candles: pd.DataFrame) -> list[dict[str, Any]]:
    records = []
    for row in candles.itertuples(index=False):
        records.append(
            {
                "time": unix_seconds(row.timestamp),
                "timestamp": iso_utc(row.timestamp),
                "open": float(row.open),
                "high": float(row.high),
                "low": float(row.low),
                "close": float(row.close),
                "volume": float(row.volume),
            }
        )
    return records


def prediction_to_records(prediction: pd.DataFrame) -> list[dict[str, Any]]:
    records = []
    for ts, row in prediction.iterrows():
        records.append(
            {
                "time": unix_seconds(ts),
                "timestamp": iso_utc(ts),
                "open": float(row["open"]),
                "high": float(row["high"]),
                "low": float(row["low"]),
                "close": float(row["close"]),
            }
        )
    return records


def build_app(config_path: Path) -> Flask:
    config = load_config(config_path)
    app = Flask(__name__, template_folder="templates", static_folder="static")

    @app.get("/")
    def index() -> str:
        return render_template("index.html")

    @app.get("/api/config")
    def api_config() -> Any:
        demos = [merge_demo(raw, config.get("defaults", {})).__dict__ for raw in config["demos"]]
        return jsonify({"model": config.get("model", {}), "demos": demos})

    @app.get("/api/candles")
    def api_candles() -> Any:
        try:
            demo = get_demo(config, request.args.get("id", config["demos"][0]["id"]))
            candles = fetch_candles(demo)
            return jsonify({"demo": demo.__dict__, "candles": candles_to_records(candles)})
        except Exception as exc:
            return jsonify({"error": str(exc)}), 500

    @app.post("/api/forecast")
    def api_forecast() -> Any:
        try:
            payload = request.get_json(silent=True) or {}
            demo = get_demo(config, str(payload.get("id") or config["demos"][0]["id"]))
            return jsonify(run_forecast(config, demo))
        except Exception as exc:
            return jsonify({"error": str(exc)}), 500

    @app.get("/api/health")
    def api_health() -> Any:
        return jsonify({"ok": True})

    return app


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the local Kronos demo app.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--host", default=None)
    parser.add_argument("--port", type=int, default=None)
    parser.add_argument("--list", action="store_true", help="List configured demos and exit.")
    args = parser.parse_args()

    config = load_config(args.config)
    if args.list:
        defaults = config.get("defaults", {})
        for raw in config["demos"]:
            demo = merge_demo(raw, defaults)
            print(f"{demo.id}: {demo.provider}/{demo.market} {demo.symbol} {demo.interval}")
        return

    app = build_app(args.config)
    server = config.get("server", {})
    host = args.host or server.get("host", "127.0.0.1")
    port = args.port or int(server.get("port", 7071))
    app.run(host=host, port=port, debug=False)


if __name__ == "__main__":
    main()
