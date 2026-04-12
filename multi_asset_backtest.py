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

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from backtest_kronos import (
    LOOKBACK,
    PRED_LEN,
    ROUND_TRIP_COST,
    STEP,
    choose_device,
    future_timestamps,
    max_drawdown,
    normalize_columns,
    predict_close_samples,
    set_seed,
)
from model import Kronos, KronosPredictor, KronosTokenizer


ASSETS = [
    ("btc_5m", Path("data/MEXC_BTCUSDT.P, 5.csv")),
    ("sol_5m", Path("data/MEXC_SOLUSDT.P, 5 (1).csv")),
    ("zec_5m", Path("data/BINANCE_ZECUSDT, 5.csv")),
]

OUT_DIR = Path("outputs/multi_asset")
SEED = 42


def run_asset(slug: str, csv_path: Path, predictor: KronosPredictor) -> dict:
    set_seed(SEED)
    df = normalize_columns(pd.read_csv(csv_path))
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    trades_path = OUT_DIR / f"{slug}_trades.csv"
    equity_path = OUT_DIR / f"{slug}_equity.png"

    rows = []
    last_end_idx = len(df) - PRED_LEN - 1
    signal_indices = list(range(LOOKBACK - 1, last_end_idx + 1, STEP))

    print(
        f"{slug}: rows={len(df)} range={df['timestamp'].iloc[0]}->{df['timestamp'].iloc[-1]} "
        f"signals={len(signal_indices)}",
        flush=True,
    )

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

        if upside_probability >= 0.60:
            position = 1
        elif upside_probability <= 0.40:
            position = -1
        else:
            position = 0

        gross_return = position * (exit_close / entry_open - 1.0) if position else 0.0
        net_return = gross_return - ROUND_TRIP_COST if position else 0.0

        rows.append(
            {
                "asset": slug,
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

        if signal_number == 1 or signal_number % 50 == 0 or signal_number == len(signal_indices):
            pd.DataFrame(rows).to_csv(trades_path, index=False)
            print(
                f"{slug}: {signal_number}/{len(signal_indices)} "
                f"{df.loc[end_idx, 'timestamp']} p_up={upside_probability:.2f} "
                f"pos={position:+d} net={net_return:.4%}",
                flush=True,
            )

    trades = pd.DataFrame(rows)
    trades.to_csv(trades_path, index=False)

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
    plt.title(f"{slug} Kronos Equity")
    plt.xlabel("Exit time")
    plt.ylabel("Equity")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(equity_path, dpi=150)
    plt.close()

    summary = {
        "asset": slug,
        "csv_path": str(csv_path),
        "rows": len(df),
        "start": df["timestamp"].iloc[0],
        "end": df["timestamp"].iloc[-1],
        "signals": len(signal_indices),
        "trades": len(traded),
        "total_return": total_return,
        "win_rate": win_rate,
        "max_drawdown": mdd,
        "trades_path": str(trades_path),
        "equity_path": str(equity_path),
    }
    print(
        f"{slug}: DONE trades={len(traded)} total={total_return:.2%} "
        f"win={win_rate:.2%} mdd={mdd:.2%}",
        flush=True,
    )
    return summary


def main() -> None:
    device = choose_device()
    print(f"device: {device}", flush=True)
    tokenizer = KronosTokenizer.from_pretrained("NeoQuasar/Kronos-Tokenizer-2k")
    model = Kronos.from_pretrained("NeoQuasar/Kronos-mini")
    predictor = KronosPredictor(model, tokenizer, device=device, max_context=2048)

    summaries = []
    for slug, path in ASSETS:
        summaries.append(run_asset(slug, path, predictor))

    summary_df = pd.DataFrame(summaries)
    summary_path = OUT_DIR / "summary.csv"
    summary_df.to_csv(summary_path, index=False)
    print()
    print(summary_df.to_string(index=False, formatters={
        "total_return": "{:.2%}".format,
        "win_rate": "{:.2%}".format,
        "max_drawdown": "{:.2%}".format,
    }))
    print(f"saved_summary: {summary_path}")


if __name__ == "__main__":
    main()
