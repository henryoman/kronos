# /// script
# requires-python = ">=3.14"
# dependencies = [
#   "einops",
#   "huggingface-hub",
#   "matplotlib",
#   "numpy",
#   "pandas",
#   "safetensors",
#   "torch",
#   "tqdm",
# ]
# ///

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from model import Kronos, KronosPredictor, KronosTokenizer


CSV_PATH = Path("data/BINANCE_BTCUSDT.P, 60.csv")
TRADES_PATH = Path("outputs/trades.csv")
EQUITY_PATH = Path("outputs/equity.png")

LOOKBACK = 360
PRED_LEN = 24
SAMPLE_COUNT = 5
STEP = 24
TOP_P = 0.95
TEMP = 1.0
SEED = 42

LONG_THRESHOLD = 0.60
SHORT_THRESHOLD = 0.40

FEE_BPS_PER_SIDE = 5
SLIPPAGE_BPS_PER_SIDE = 2
ROUND_TRIP_COST = 2 * ((FEE_BPS_PER_SIDE + SLIPPAGE_BPS_PER_SIDE) / 10_000)


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [
        str(col).strip().lower().replace(" ", "_").replace("-", "_")
        for col in df.columns
    ]

    rename = {}
    for col in df.columns:
        compact = col.replace("_", "").replace(".", "")
        if compact in {"time", "date", "datetime"}:
            rename[col] = "timestamp"
        elif compact == "vol":
            rename[col] = "volume"
    df = df.rename(columns=rename)

    if "volume" not in df.columns:
        volume_candidates = [
            col
            for col in df.columns
            if col.startswith("volume") or col.replace(".", "") == "vol"
        ]
        if volume_candidates:
            df = df.rename(columns={volume_candidates[0]: "volume"})

    required = {"timestamp", "open", "high", "low", "close"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Missing required CSV columns: {missing}")

    if "volume" not in df.columns:
        df["volume"] = 0.0
    df["amount"] = 0.0

    df = df[["timestamp", "open", "high", "low", "close", "volume", "amount"]]

    for col in ["open", "high", "low", "close", "volume", "amount"]:
        cleaned = (
            df[col]
            .astype(str)
            .str.replace(",", "", regex=False)
            .replace({"": np.nan, "nan": np.nan, "None": np.nan})
        )
        df[col] = pd.to_numeric(cleaned, errors="coerce")

    df["timestamp"] = parse_timestamps(df["timestamp"])
    df = df.dropna(subset=["timestamp", "open", "high", "low", "close"])
    df = df.sort_values("timestamp").drop_duplicates("timestamp").reset_index(drop=True)

    min_rows = LOOKBACK + PRED_LEN + 2
    if len(df) < min_rows:
        raise ValueError(f"Need at least {min_rows} rows, found {len(df)}.")

    return df


def parse_timestamps(values: pd.Series) -> pd.Series:
    raw = values.copy()
    numeric = pd.to_numeric(raw, errors="coerce")

    if numeric.notna().mean() > 0.8:
        median = float(numeric.dropna().median())
        if median > 1e17:
            parsed = pd.to_datetime(numeric, unit="ns", errors="coerce", utc=True)
        elif median > 1e14:
            parsed = pd.to_datetime(numeric, unit="us", errors="coerce", utc=True)
        elif median > 1e11:
            parsed = pd.to_datetime(numeric, unit="ms", errors="coerce", utc=True)
        elif median > 1e9:
            parsed = pd.to_datetime(numeric, unit="s", errors="coerce", utc=True)
        else:
            parsed = pd.to_datetime(raw, errors="coerce", utc=True)
    else:
        parsed = pd.to_datetime(raw, errors="coerce", utc=True)

    return parsed.dt.tz_convert(None)


def choose_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def future_timestamps(df: pd.DataFrame, end_idx: int) -> pd.Series:
    known = df.loc[end_idx + 1 : end_idx + PRED_LEN, "timestamp"]
    if len(known) == PRED_LEN:
        return known.reset_index(drop=True)

    history = df.loc[:end_idx, "timestamp"]
    freq = pd.infer_freq(history.tail(min(len(history), 20)))
    if freq is not None:
        offset = pd.tseries.frequencies.to_offset(freq)
    else:
        delta = history.diff().dropna().median()
        offset = delta if pd.notna(delta) else pd.Timedelta(hours=1)

    start = df.loc[end_idx, "timestamp"]
    return pd.Series([start + offset * i for i in range(1, PRED_LEN + 1)])


def extract_close_samples(prediction: Any) -> np.ndarray | None:
    if isinstance(prediction, pd.DataFrame):
        if "close" not in prediction.columns:
            raise ValueError("Predictor returned a DataFrame without a close column.")
        close = prediction["close"].to_numpy(dtype=float)
        if close.shape[0] < PRED_LEN:
            raise ValueError(f"Expected at least {PRED_LEN} close predictions, got {close.shape[0]}.")
        return close[-PRED_LEN:].reshape(1, PRED_LEN)

    if not isinstance(prediction, tuple):
        raise TypeError(f"Unsupported predictor return type: {type(prediction)!r}")

    arrays = []
    for item in prediction:
        try:
            array = np.asarray(item, dtype=float)
        except (TypeError, ValueError):
            continue
        if array.size:
            arrays.append(np.squeeze(array))

    for array in arrays:
        if array.ndim == 1 and array.shape[0] >= PRED_LEN:
            return array[-PRED_LEN:].reshape(1, PRED_LEN)

        if array.ndim == 2:
            if array.shape[1] >= PRED_LEN:
                return array[:, -PRED_LEN:]
            if array.shape[0] >= PRED_LEN:
                return array[-PRED_LEN:, :].T

        if array.ndim == 3:
            if array.shape[-1] >= 4 and array.shape[-2] >= PRED_LEN:
                close = array[..., -PRED_LEN:, 3]
                return close.reshape(-1, PRED_LEN)
            if array.shape[-1] >= PRED_LEN:
                return array[..., -PRED_LEN:].reshape(-1, PRED_LEN)

    raise TypeError("Predictor returned a tuple, but no close prediction samples were usable.")


def predict_close_samples(
    predictor: KronosPredictor,
    window: pd.DataFrame,
    x_timestamp: pd.Series,
    y_timestamp: pd.Series,
) -> np.ndarray:
    frame = window[["open", "high", "low", "close", "volume", "amount"]]

    if hasattr(predictor, "predict_batch"):
        batch_prediction = predictor.predict_batch(
            df_list=[frame] * SAMPLE_COUNT,
            x_timestamp_list=[x_timestamp] * SAMPLE_COUNT,
            y_timestamp_list=[y_timestamp] * SAMPLE_COUNT,
            pred_len=PRED_LEN,
            T=TEMP,
            top_k=0,
            top_p=TOP_P,
            sample_count=1,
            verbose=False,
        )
        if isinstance(batch_prediction, list):
            return np.vstack(
                [extract_close_samples(prediction)[0] for prediction in batch_prediction]
            )
        batch_samples = extract_close_samples(batch_prediction)
        if batch_samples is not None:
            return batch_samples

    samples = []
    for _ in range(SAMPLE_COUNT):
        prediction = predictor.predict(
            df=frame,
            x_timestamp=x_timestamp,
            y_timestamp=y_timestamp,
            pred_len=PRED_LEN,
            T=TEMP,
            top_k=0,
            top_p=TOP_P,
            sample_count=1,
            verbose=False,
        )
        close_sample = extract_close_samples(prediction)
        if close_sample.shape[0] > 1:
            return close_sample
        samples.append(close_sample[0])

    return np.vstack(samples)


def max_drawdown(equity: pd.Series) -> float:
    if equity.empty:
        return 0.0
    peak = equity.cummax()
    drawdown = equity / peak - 1.0
    return float(drawdown.min())


def main() -> None:
    if not CSV_PATH.exists():
        raise FileNotFoundError(f"Place your TradingView CSV at {CSV_PATH}")

    set_seed(SEED)
    Path("outputs").mkdir(exist_ok=True)

    df = normalize_columns(pd.read_csv(CSV_PATH))
    device = choose_device()
    print(f"csv: {CSV_PATH}", flush=True)
    print(f"rows: {len(df)}", flush=True)
    print(f"range: {df['timestamp'].iloc[0]} -> {df['timestamp'].iloc[-1]}", flush=True)
    print(f"device: {device}", flush=True)
    print(f"seed: {SEED}", flush=True)

    tokenizer = KronosTokenizer.from_pretrained("NeoQuasar/Kronos-Tokenizer-2k")
    model = Kronos.from_pretrained("NeoQuasar/Kronos-mini")
    predictor = KronosPredictor(model, tokenizer, device=device, max_context=2048)

    rows = []
    last_end_idx = len(df) - PRED_LEN - 1
    signal_indices = list(range(LOOKBACK - 1, last_end_idx + 1, STEP))
    print(f"signals: {len(signal_indices)}", flush=True)

    for signal_number, end_idx in enumerate(signal_indices, start=1):
        entry_idx = end_idx + 1
        exit_idx = end_idx + PRED_LEN

        window = df.iloc[end_idx - LOOKBACK + 1 : end_idx + 1].copy()
        x_timestamp = window["timestamp"].reset_index(drop=True)
        y_timestamp = future_timestamps(df, end_idx)

        current_close = float(df.loc[end_idx, "close"])
        entry_open = float(df.loc[entry_idx, "open"])
        exit_close = float(df.loc[exit_idx, "close"])

        close_samples = predict_close_samples(predictor, window, x_timestamp, y_timestamp)
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
        net_return = gross_return - ROUND_TRIP_COST if position else 0.0

        rows.append(
            {
                "signal_number": signal_number,
                "signal_timestamp": df.loc[end_idx, "timestamp"],
                "entry_timestamp": df.loc[entry_idx, "timestamp"],
                "exit_timestamp": df.loc[exit_idx, "timestamp"],
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
        pd.DataFrame(rows).to_csv(TRADES_PATH, index=False)

        print(
            f"{signal_number}/{len(signal_indices)} "
            f"{df.loc[end_idx, 'timestamp']} "
            f"p_up={upside_probability:.2f} "
            f"pos={position:+d} "
            f"net={net_return:.4%}",
            flush=True,
        )

    trades = pd.DataFrame(rows)
    trades.to_csv(TRADES_PATH, index=False)

    traded = trades[trades["position"] != 0].copy()
    if traded.empty:
        equity = pd.Series([1.0])
        total_return = 0.0
        win_rate = 0.0
        mdd = 0.0
    else:
        equity = (1.0 + traded["net_return"]).cumprod()
        total_return = float(equity.iloc[-1] - 1.0)
        win_rate = float((traded["net_return"] > 0).mean())
        mdd = max_drawdown(equity)

    plt.figure(figsize=(10, 5))
    if traded.empty:
        plt.plot([0], [1.0])
    else:
        plt.plot(pd.to_datetime(traded["exit_timestamp"]), equity)
    plt.title("Kronos Walk-Forward Equity")
    plt.xlabel("Exit time")
    plt.ylabel("Equity")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(EQUITY_PATH, dpi=150)
    plt.close()

    print("", flush=True)
    print(f"trades: {len(traded)}", flush=True)
    print(f"total_return: {total_return:.2%}", flush=True)
    print(f"win_rate: {win_rate:.2%}", flush=True)
    print(f"max_drawdown: {mdd:.2%}", flush=True)
    print(f"saved_trades: {TRADES_PATH}", flush=True)
    print(f"saved_equity: {EQUITY_PATH}", flush=True)


if __name__ == "__main__":
    main()
