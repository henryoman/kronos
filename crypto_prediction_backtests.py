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

import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from backtest_kronos import (
    LOOKBACK,
    PRED_LEN,
    ROUND_TRIP_COST,
    SAMPLE_COUNT,
    STEP,
    TEMP,
    TOP_P,
    choose_device,
    future_timestamps,
    max_drawdown,
    normalize_columns,
    set_seed,
)
from model import Kronos, KronosPredictor, KronosTokenizer


ASSETS = [
    ("btc_5m", Path("data/MEXC_BTCUSDT.P, 5.csv")),
    ("sol_5m", Path("data/MEXC_SOLUSDT.P, 5 (1).csv")),
    ("zec_5m", Path("data/BINANCE_ZECUSDT, 5.csv")),
]

OUT_DIR = Path("outputs/crypto_prediction_reviews")
VISIBLE_HISTORY_BARS = 120
REVIEW_STRIDE = 4
SEED = 42


def draw_candles(
    ax: plt.Axes,
    df: pd.DataFrame,
    *,
    x_offset: int = 0,
    width: float = 0.62,
    alpha: float = 1.0,
    up_color: str = "#177a59",
    down_color: str = "#b23b3b",
) -> None:
    for idx, row in enumerate(df.itertuples(index=False), start=x_offset):
        color = up_color if row.close >= row.open else down_color
        ax.vlines(idx, row.low, row.high, color=color, linewidth=1.0, alpha=alpha)
        low_body = min(row.open, row.close)
        body_height = abs(row.close - row.open)
        if body_height == 0:
            body_height = max(row.close * 0.0002, 0.0001)
        ax.add_patch(
            plt.Rectangle(
                (idx - width / 2, low_body),
                width,
                body_height,
                facecolor=color,
                edgecolor=color,
                linewidth=0.8,
                alpha=alpha,
            )
        )


def style_axis(ax: plt.Axes) -> None:
    ax.set_facecolor("#f5f2e8")
    ax.grid(True, axis="y", color="#1f231d", alpha=0.13, linewidth=0.8)
    ax.grid(False, axis="x")
    for spine in ax.spines.values():
        spine.set_color("#1f231d")
        spine.set_linewidth(1.2)
    ax.tick_params(colors="#3a4135", labelsize=9)


def render_contact_sheet(image_paths: list[Path], output_path: Path) -> None:
    if not image_paths:
        return
    images = [plt.imread(path) for path in image_paths]
    cols = min(3, max(1, math.ceil(math.sqrt(len(images)))))
    rows = math.ceil(len(images) / cols)
    fig, axes = plt.subplots(
        rows,
        cols,
        figsize=(cols * 6.2, rows * 3.9),
        dpi=140,
        facecolor="#f5f2e8",
        layout="constrained",
    )
    flat_axes = np.atleast_1d(axes).ravel()
    for ax, image, path in zip(flat_axes, images, image_paths, strict=False):
        ax.imshow(image)
        ax.set_title(path.stem, fontsize=12, color="#1f231d")
        ax.axis("off")
    for ax in flat_axes[len(images):]:
        ax.axis("off")
    fig.savefig(output_path, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)


def evenly_spaced_paths(paths: list[Path], count: int = 9) -> list[Path]:
    if len(paths) <= count:
        return paths
    indices = np.linspace(0, len(paths) - 1, num=count, dtype=int)
    return [paths[index] for index in indices]


def prediction_frames(
    predictor: KronosPredictor,
    window: pd.DataFrame,
    x_timestamp: pd.Series,
    y_timestamp: pd.Series,
) -> list[pd.DataFrame]:
    frame = window[["open", "high", "low", "close", "volume", "amount"]]
    predictions = predictor.predict_batch(
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
    return [prediction.copy() for prediction in predictions]


def mean_prediction_frame(predictions: list[pd.DataFrame]) -> pd.DataFrame:
    mean_prediction = pd.concat(predictions).groupby(level=0).mean(numeric_only=True)
    return mean_prediction.reset_index().rename(columns={"index": "timestamp"})


def render_signal_chart(
    *,
    slug: str,
    history: pd.DataFrame,
    actual_future: pd.DataFrame,
    predicted_future: pd.DataFrame,
    signal_timestamp: pd.Timestamp,
    upside_probability: float,
    forecast_return: float,
    actual_return: float,
    position: int,
    direction_correct: bool,
    path: Path,
) -> None:
    history = history.reset_index(drop=True)
    actual_future = actual_future.reset_index(drop=True)
    predicted_future = predicted_future.reset_index(drop=True)

    combined = pd.concat([history, actual_future, predicted_future], ignore_index=True)
    y_min = float(combined["low"].min())
    y_max = float(combined["high"].max())
    y_pad = (y_max - y_min) * 0.08 or y_max * 0.01 or 1.0

    fig = plt.figure(figsize=(15, 8.5), dpi=150, facecolor="#f5f2e8")
    ax = fig.add_axes((0.055, 0.16, 0.89, 0.66))
    style_axis(ax)

    draw_candles(ax, history)
    split = len(history) - 0.5
    ax.axvline(split, color="#806a00", linestyle=(0, (4, 5)), linewidth=1.2)

    draw_candles(
        ax,
        actual_future,
        x_offset=len(history),
        alpha=0.30,
        up_color="#6a717c",
        down_color="#6a717c",
    )
    draw_candles(
        ax,
        predicted_future,
        x_offset=len(history),
        alpha=0.72,
        up_color="#b88500",
        down_color="#b88500",
    )

    actual_x = np.arange(len(history), len(history) + len(actual_future))
    predicted_x = np.arange(len(history), len(history) + len(predicted_future))
    ax.plot(actual_x, actual_future["close"], color="#1f231d", linewidth=1.6, label="Actual close")
    ax.plot(predicted_x, predicted_future["close"], color="#b88500", linewidth=2.0, label="Predicted close")

    ax.set_ylim(y_min - y_pad, y_max + y_pad)
    ax.set_xlim(-1, len(history) + len(actual_future))
    ax.set_ylabel("Price", color="#1f231d")
    ax.set_xticks(
        [
            0,
            max(0, len(history) // 2),
            len(history) - 1,
            len(history) + len(actual_future) - 1,
        ]
    )
    ax.set_xticklabels(
        [
            history["timestamp"].iloc[0].strftime("%m-%d %H:%M"),
            history["timestamp"].iloc[max(0, len(history) // 2)].strftime("%m-%d %H:%M"),
            history["timestamp"].iloc[-1].strftime("%m-%d %H:%M"),
            actual_future["timestamp"].iloc[-1].strftime("%m-%d %H:%M"),
        ]
    )
    ax.legend(loc="upper left", frameon=False, fontsize=9)

    symbol = slug.split("_", maxsplit=1)[0].upper()
    side = "LONG" if position > 0 else "SHORT" if position < 0 else "FLAT"
    verdict = "YES" if direction_correct else "NO"
    verdict_color = "#177a59" if direction_correct else "#b23b3b"

    fig.text(0.055, 0.925, f"{symbol} 5m Backtest Forecast Review", fontsize=28, color="#1f231d", fontweight="bold")
    fig.text(
        0.057,
        0.875,
        f"Prediction made at {signal_timestamp.strftime('%Y-%m-%d %H:%M')} and overlaid on realized candles",
        fontsize=12,
        color="#64705d",
    )

    stats = [
        ("Signal", side),
        ("Upside Probability", f"{upside_probability * 100:.1f}%"),
        ("Forecast Return", f"{forecast_return * 100:+.2f}%"),
        ("Actual Return", f"{actual_return * 100:+.2f}%"),
    ]
    x0 = 0.055
    for idx, (label, value) in enumerate(stats):
        fig.text(x0 + idx * 0.215, 0.06, label.upper(), fontsize=9, color="#64705d")
        fig.text(x0 + idx * 0.215, 0.025, value, fontsize=18, color="#1f231d", fontweight="bold")
    fig.text(0.865, 0.06, "DIR CORRECT", fontsize=9, color="#64705d")
    fig.text(0.865, 0.025, verdict, fontsize=18, color=verdict_color, fontweight="bold")

    fig.savefig(path, facecolor=fig.get_facecolor())
    plt.close(fig)


def run_asset(slug: str, csv_path: Path, predictor: KronosPredictor) -> dict:
    set_seed(SEED)
    df = normalize_columns(pd.read_csv(csv_path))

    asset_dir = OUT_DIR / slug
    chart_dir = asset_dir / "signals"
    asset_dir.mkdir(parents=True, exist_ok=True)
    chart_dir.mkdir(parents=True, exist_ok=True)

    trades_path = asset_dir / "trades.csv"
    equity_path = asset_dir / "equity.png"
    grid_path = asset_dir / "review_grid.png"

    rows = []
    chart_paths: list[Path] = []
    last_end_idx = len(df) - PRED_LEN - 1
    signal_indices = list(range(LOOKBACK - 1, last_end_idx + 1, STEP * REVIEW_STRIDE))

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

        predictions = prediction_frames(predictor, window, x_timestamp, y_timestamp)
        mean_prediction = mean_prediction_frame(predictions)
        final_pred_close = np.array(
            [float(prediction["close"].iloc[-1]) for prediction in predictions],
            dtype=float,
        )

        upside_probability = float(np.mean(final_pred_close > current_close))
        forecast_return = float(np.mean(final_pred_close / current_close - 1.0))
        actual_return = float(exit_close / current_close - 1.0)

        if upside_probability >= 0.60:
            position = 1
        elif upside_probability <= 0.40:
            position = -1
        else:
            position = 0

        gross_return = position * (exit_close / entry_open - 1.0) if position else 0.0
        net_return = gross_return - ROUND_TRIP_COST if position else 0.0
        direction_correct = (forecast_return >= 0.0) == (actual_return >= 0.0)

        history = df.iloc[max(0, end_idx - VISIBLE_HISTORY_BARS + 1) : end_idx + 1][
            ["timestamp", "open", "high", "low", "close"]
        ].copy()
        actual_future = df.iloc[entry_idx : exit_idx + 1][["timestamp", "open", "high", "low", "close"]].copy()

        signal_timestamp = pd.Timestamp(df.loc[end_idx, "timestamp"])
        chart_name = f"{signal_number:04d}_{signal_timestamp.strftime('%Y%m%d_%H%M%S')}.png"
        chart_path = chart_dir / chart_name
        render_signal_chart(
            slug=slug,
            history=history,
            actual_future=actual_future,
            predicted_future=mean_prediction[["timestamp", "open", "high", "low", "close"]],
            signal_timestamp=signal_timestamp,
            upside_probability=upside_probability,
            forecast_return=forecast_return,
            actual_return=actual_return,
            position=position,
            direction_correct=direction_correct,
            path=chart_path,
        )
        chart_paths.append(chart_path)

        rows.append(
            {
                "asset": slug,
                "signal_number": signal_number,
                "signal_timestamp": signal_timestamp,
                "entry_timestamp": df.loc[entry_idx, "timestamp"],
                "exit_timestamp": df.loc[exit_idx, "timestamp"],
                "current_close": current_close,
                "entry_open": entry_open,
                "exit_close": exit_close,
                "mean_final_pred_close": float(np.mean(final_pred_close)),
                "forecast_return": forecast_return,
                "actual_return": actual_return,
                "upside_probability": upside_probability,
                "position": position,
                "direction_correct": direction_correct,
                "gross_return": gross_return,
                "net_return": net_return,
                "final_close_error_pct": float(abs(np.mean(final_pred_close) / exit_close - 1.0)),
                "chart_path": str(chart_path),
            }
        )

        if signal_number == 1 or signal_number % 25 == 0 or signal_number == len(signal_indices):
            pd.DataFrame(rows).to_csv(trades_path, index=False)
            print(
                f"{slug}: {signal_number}/{len(signal_indices)} "
                f"{signal_timestamp} p_up={upside_probability:.2f} "
                f"actual={actual_return:+.2%} net={net_return:+.2%}",
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

    render_contact_sheet(evenly_spaced_paths(chart_paths), grid_path)

    summary = {
        "asset": slug,
        "csv_path": str(csv_path),
        "rows": len(df),
        "start": df["timestamp"].iloc[0],
        "end": df["timestamp"].iloc[-1],
        "signals": len(signal_indices),
        "review_stride_bars": STEP * REVIEW_STRIDE,
        "trades": len(traded),
        "direction_accuracy": float(trades["direction_correct"].mean()),
        "avg_final_close_error_pct": float(trades["final_close_error_pct"].mean()),
        "total_return": total_return,
        "win_rate": win_rate,
        "max_drawdown": mdd,
        "trades_path": str(trades_path),
        "equity_path": str(equity_path),
        "grid_path": str(grid_path),
        "signals_dir": str(chart_dir),
    }
    print(
        f"{slug}: DONE accuracy={summary['direction_accuracy']:.2%} "
        f"avg_error={summary['avg_final_close_error_pct']:.2%} "
        f"trades={len(traded)} total={total_return:.2%}",
        flush=True,
    )
    return summary


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
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
    print(
        summary_df.to_string(
            index=False,
            formatters={
                "direction_accuracy": "{:.2%}".format,
                "avg_final_close_error_pct": "{:.2%}".format,
                "total_return": "{:.2%}".format,
                "win_rate": "{:.2%}".format,
                "max_drawdown": "{:.2%}".format,
            },
        )
    )
    print(f"saved_summary: {summary_path}")


if __name__ == "__main__":
    main()
