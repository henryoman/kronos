#!/usr/bin/env python3
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

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import backtest_kronos as core
from model import Kronos, KronosPredictor, KronosTokenizer


LOOKBACK = 360
PRED_LEN = 24
SAMPLE_COUNT = 5
TOP_P = 0.95
TEMP = 1.0
LONG_THRESHOLD = 0.60
SHORT_THRESHOLD = 0.40
FEE_BPS_PER_SIDE = 5
SLIPPAGE_BPS_PER_SIDE = 2
ROUND_TRIP_COST = 2 * ((FEE_BPS_PER_SIDE + SLIPPAGE_BPS_PER_SIDE) / 10_000)

ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT / "outputs" / "nasdaq_backtests"


@dataclass(frozen=True)
class Experiment:
    slug: str
    csv_path: Path
    mode: str
    seed: int
    sample_size: int | None = None
    tail_signals: int | None = None


EXPERIMENTS = [
    Experiment(
        slug="nq_5m_random_seed7",
        csv_path=ROOT / "data" / "demo_app" / "NQ_in_5_minute.csv",
        mode="random",
        seed=7,
        sample_size=64,
    ),
    Experiment(
        slug="nq_1h_random_seed42",
        csv_path=ROOT / "data" / "demo_app" / "NQ_in_1_hour.csv",
        mode="random",
        seed=42,
        sample_size=64,
    ),
    Experiment(
        slug="nq_1d_random_seed99",
        csv_path=ROOT / "data" / "demo_app" / "NQ_in_daily.csv",
        mode="random",
        seed=99,
        sample_size=64,
    ),
    Experiment(
        slug="nq_1d_every_bar_last160",
        csv_path=ROOT / "data" / "demo_app" / "NQ_in_daily.csv",
        mode="every_bar",
        seed=123,
        tail_signals=160,
    ),
]


def configure_core() -> None:
    core.LOOKBACK = LOOKBACK
    core.PRED_LEN = PRED_LEN
    core.SAMPLE_COUNT = SAMPLE_COUNT
    core.TOP_P = TOP_P
    core.TEMP = TEMP
    core.LONG_THRESHOLD = LONG_THRESHOLD
    core.SHORT_THRESHOLD = SHORT_THRESHOLD
    core.ROUND_TRIP_COST = ROUND_TRIP_COST


def load_predictor() -> tuple[str, KronosPredictor]:
    device = core.choose_device()
    tokenizer = KronosTokenizer.from_pretrained("NeoQuasar/Kronos-Tokenizer-2k")
    model = Kronos.from_pretrained("NeoQuasar/Kronos-mini")
    predictor = KronosPredictor(model, tokenizer, device=device, max_context=2048)
    return device, predictor


def eligible_signal_indices(df: pd.DataFrame) -> list[int]:
    last_end_idx = len(df) - PRED_LEN - 1
    return list(range(LOOKBACK - 1, last_end_idx + 1))


def select_signal_indices(experiment: Experiment, df: pd.DataFrame) -> list[int]:
    indices = eligible_signal_indices(df)
    if experiment.mode == "every_bar":
        if experiment.tail_signals is not None:
            indices = indices[-experiment.tail_signals :]
        return indices

    if experiment.mode == "random":
        if experiment.sample_size is None:
            raise ValueError(f"{experiment.slug} is missing sample_size.")
        sample_size = min(experiment.sample_size, len(indices))
        rng = np.random.default_rng(experiment.seed)
        chosen = rng.choice(indices, size=sample_size, replace=False)
        return sorted(int(value) for value in chosen)

    raise ValueError(f"Unsupported experiment mode: {experiment.mode}")


def render_equity_curve(traded: pd.DataFrame, path: Path, title: str) -> None:
    plt.figure(figsize=(10, 5))
    if traded.empty:
        plt.plot([0], [1.0])
    else:
        equity = (1.0 + traded["net_return"]).cumprod()
        plt.plot(pd.to_datetime(traded["exit_timestamp"]), equity)
    plt.title(title)
    plt.xlabel("Exit time")
    plt.ylabel("Equity")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def summarize_results(experiment: Experiment, df: pd.DataFrame, trades: pd.DataFrame) -> dict[str, object]:
    traded = trades[trades["position"] != 0].copy()
    total_compounded_return = float((1.0 + traded["net_return"]).prod() - 1.0) if not traded.empty else 0.0
    max_drawdown = (
        core.max_drawdown((1.0 + traded["net_return"]).cumprod()) if not traded.empty else 0.0
    )
    threshold_accuracy = float(traded["prob_signal_correct"].mean()) if not traded.empty else 0.0
    return {
        "slug": experiment.slug,
        "mode": experiment.mode,
        "csv_path": str(experiment.csv_path.relative_to(ROOT)),
        "rows": len(df),
        "start": df["timestamp"].iloc[0],
        "end": df["timestamp"].iloc[-1],
        "signals": len(trades),
        "traded_signals": len(traded),
        "coverage": float(len(traded) / len(trades)) if len(trades) else 0.0,
        "forecast_direction_accuracy": float(trades["forecast_direction_correct"].mean()) if len(trades) else 0.0,
        "threshold_signal_accuracy": threshold_accuracy,
        "gross_win_rate": float((traded["gross_return"] > 0).mean()) if not traded.empty else 0.0,
        "net_win_rate": float((traded["net_return"] > 0).mean()) if not traded.empty else 0.0,
        "avg_forecast_return": float(trades["forecast_return"].mean()) if len(trades) else 0.0,
        "avg_actual_return_from_entry": float(trades["actual_return_from_entry"].mean()) if len(trades) else 0.0,
        "avg_net_return": float(traded["net_return"].mean()) if not traded.empty else 0.0,
        "total_compounded_return": total_compounded_return,
        "max_drawdown": max_drawdown,
    }


def run_experiment(experiment: Experiment, predictor: KronosPredictor) -> dict[str, object]:
    core.set_seed(experiment.seed)
    df = core.normalize_columns(pd.read_csv(experiment.csv_path))
    signal_indices = select_signal_indices(experiment, df)
    run_dir = OUT_DIR / experiment.slug
    run_dir.mkdir(parents=True, exist_ok=True)
    trades_path = run_dir / "trades.csv"
    equity_path = run_dir / "equity.png"
    summary_path = run_dir / "summary.csv"

    rows: list[dict[str, object]] = []
    print(
        f"{experiment.slug}: rows={len(df)} range={df['timestamp'].iloc[0]}->{df['timestamp'].iloc[-1]} "
        f"signals={len(signal_indices)} mode={experiment.mode}",
        flush=True,
    )

    for signal_number, end_idx in enumerate(signal_indices, start=1):
        entry_idx = end_idx + 1
        exit_idx = end_idx + PRED_LEN
        window = df.iloc[end_idx - LOOKBACK + 1 : end_idx + 1].copy()
        x_timestamp = window["timestamp"].reset_index(drop=True)
        y_timestamp = core.future_timestamps(df, end_idx)

        current_close = float(df.loc[end_idx, "close"])
        entry_open = float(df.loc[entry_idx, "open"])
        exit_close = float(df.loc[exit_idx, "close"])

        close_samples = core.predict_close_samples(predictor, window, x_timestamp, y_timestamp)
        final_pred_close = close_samples[:, -1]
        mean_final_pred_close = float(np.mean(final_pred_close))
        upside_probability = float(np.mean(final_pred_close > current_close))
        forecast_return = float(np.mean(final_pred_close / current_close - 1.0))

        if upside_probability >= LONG_THRESHOLD:
            position = 1
        elif upside_probability <= SHORT_THRESHOLD:
            position = -1
        else:
            position = 0

        actual_return_from_current = float(exit_close / current_close - 1.0)
        actual_return_from_entry = float(exit_close / entry_open - 1.0)
        actual_up_from_current = bool(exit_close > current_close)
        actual_dir_from_entry = 1 if exit_close > entry_open else -1
        mean_pred_up = bool(mean_final_pred_close > current_close)
        forecast_direction_correct = bool(mean_pred_up == actual_up_from_current)
        prob_signal_correct = bool(position == actual_dir_from_entry) if position else False
        directional_edge = float(position * actual_return_from_entry) if position else 0.0
        gross_return = float(position * actual_return_from_entry) if position else 0.0
        net_return = float(gross_return - ROUND_TRIP_COST) if position else 0.0

        rows.append(
            {
                "slug": experiment.slug,
                "mode": experiment.mode,
                "signal_number": signal_number,
                "signal_timestamp": df.loc[end_idx, "timestamp"],
                "entry_timestamp": df.loc[entry_idx, "timestamp"],
                "exit_timestamp": df.loc[exit_idx, "timestamp"],
                "current_close": current_close,
                "entry_open": entry_open,
                "exit_close": exit_close,
                "mean_final_pred_close": mean_final_pred_close,
                "forecast_return": forecast_return,
                "upside_probability": upside_probability,
                "position": position,
                "actual_return_from_current": actual_return_from_current,
                "actual_return_from_entry": actual_return_from_entry,
                "actual_up_from_current": actual_up_from_current,
                "actual_dir_from_entry": actual_dir_from_entry,
                "mean_pred_up": mean_pred_up,
                "forecast_direction_correct": forecast_direction_correct,
                "prob_signal_correct": prob_signal_correct,
                "directional_edge": directional_edge,
                "gross_return": gross_return,
                "net_return": net_return,
            }
        )

        if signal_number == 1 or signal_number % 10 == 0 or signal_number == len(signal_indices):
            pd.DataFrame(rows).to_csv(trades_path, index=False)
            print(
                f"{experiment.slug}: {signal_number}/{len(signal_indices)} "
                f"{df.loc[end_idx, 'timestamp']} p_up={upside_probability:.2f} "
                f"pos={position:+d} dir_ok={forecast_direction_correct} net={net_return:.4%}",
                flush=True,
            )

    trades = pd.DataFrame(rows)
    trades.to_csv(trades_path, index=False)
    traded = trades[trades["position"] != 0].copy()
    render_equity_curve(traded, equity_path, f"{experiment.slug} Kronos Equity")

    summary = summarize_results(experiment, df, trades)
    pd.DataFrame([summary]).to_csv(summary_path, index=False)
    print(
        f"{experiment.slug}: DONE direction={summary['forecast_direction_accuracy']:.2%} "
        f"threshold={summary['threshold_signal_accuracy']:.2%} "
        f"coverage={summary['coverage']:.2%} total={summary['total_compounded_return']:.2%}",
        flush=True,
    )
    return summary


def main() -> None:
    configure_core()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    device, predictor = load_predictor()
    print(f"device: {device}", flush=True)
    print(
        f"lookback={LOOKBACK} pred_len={PRED_LEN} sample_count={SAMPLE_COUNT} "
        f"long_threshold={LONG_THRESHOLD:.2f} short_threshold={SHORT_THRESHOLD:.2f}",
        flush=True,
    )

    summaries = [run_experiment(experiment, predictor) for experiment in EXPERIMENTS]
    summary_df = pd.DataFrame(summaries)
    summary_path = OUT_DIR / "summary.csv"
    summary_df.to_csv(summary_path, index=False)

    print()
    print(
        summary_df.to_string(
            index=False,
            formatters={
                "coverage": "{:.2%}".format,
                "forecast_direction_accuracy": "{:.2%}".format,
                "threshold_signal_accuracy": "{:.2%}".format,
                "gross_win_rate": "{:.2%}".format,
                "net_win_rate": "{:.2%}".format,
                "avg_forecast_return": "{:.4%}".format,
                "avg_actual_return_from_entry": "{:.4%}".format,
                "avg_net_return": "{:.4%}".format,
                "total_compounded_return": "{:.2%}".format,
                "max_drawdown": "{:.2%}".format,
            },
        )
    )
    print(f"saved_summary: {summary_path}")


if __name__ == "__main__":
    main()
