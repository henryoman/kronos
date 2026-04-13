# /// script
# requires-python = ">=3.14"
# dependencies = [
#   "einops",
#   "flask",
#   "huggingface_hub",
#   "matplotlib",
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
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from app import DEFAULT_CONFIG, get_demo, load_config, run_forecast

APP_DIR = Path(__file__).resolve().parent
ROOT = APP_DIR.parents[0]
DEFAULT_OUTPUT_DIR = ROOT / "outputs" / "demo_app"
DEFAULT_RENDER_IDS = [
    "btc_1h_binance",
    "btc_5m_binance",
    "sol_1h_binance",
    "sol_5m_binance",
]


def as_frame(records: list[dict[str, Any]]) -> pd.DataFrame:
    df = pd.DataFrame(records)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    return df


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


def render_chart(payload: dict[str, Any], path: Path, *, visible_bars: int = 160) -> None:
    demo = payload["demo"]
    forecast = payload["forecast"]
    strategy = payload["strategy"]
    candles = as_frame(payload["candles"]).tail(visible_bars)
    prediction = as_frame(payload["prediction"])

    combined = pd.concat([candles, prediction], ignore_index=True)
    y_min = float(combined["low"].min())
    y_max = float(combined["high"].max())
    y_pad = (y_max - y_min) * 0.08 or y_max * 0.01 or 1.0

    fig = plt.figure(figsize=(15, 8.5), dpi=150, facecolor="#f5f2e8")
    ax = fig.add_axes((0.055, 0.16, 0.89, 0.66))
    style_axis(ax)

    draw_candles(ax, candles.reset_index(drop=True))
    split = len(candles) - 0.5
    ax.axvline(split, color="#806a00", linestyle=(0, (4, 5)), linewidth=1.2)
    draw_candles(
        ax,
        prediction.reset_index(drop=True),
        x_offset=len(candles),
        alpha=0.75,
        up_color="#806a00",
        down_color="#806a00",
    )

    ax.set_ylim(y_min - y_pad, y_max + y_pad)
    ax.set_xlim(-1, len(candles) + len(prediction))
    ax.set_ylabel("Price", color="#1f231d")
    ax.set_xticks(
        [
            0,
            max(0, len(candles) // 2),
            len(candles) - 1,
            len(candles) + len(prediction) - 1,
        ]
    )
    labels = [
        candles["timestamp"].iloc[0].strftime("%m-%d %H:%M"),
        candles["timestamp"].iloc[max(0, len(candles) // 2)].strftime("%m-%d %H:%M"),
        candles["timestamp"].iloc[-1].strftime("%m-%d %H:%M"),
        prediction["timestamp"].iloc[-1].strftime("%m-%d %H:%M"),
    ]
    ax.set_xticklabels(labels)

    title = f"{demo['symbol']} {demo['interval']} | {demo['provider']}/{demo['market']}"
    fig.text(0.055, 0.925, title, fontsize=30, color="#1f231d", fontweight="bold")
    fig.text(
        0.057,
        0.875,
        f"Kronos-mini forecast until {forecast['forecast_until']} | {forecast['horizon_label']} horizon",
        fontsize=12,
        color="#64705d",
    )

    stats = [
        ("Signal", strategy["side"].upper()),
        ("Upside Probability", f"{forecast['p_up'] * 100:.1f}%"),
        ("Forecast Return", f"{forecast['forecast_return'] * 100:+.2f}%"),
        ("Current Close", f"{forecast['current_close']:.4f}"),
    ]
    x0 = 0.055
    for idx, (label, value) in enumerate(stats):
        fig.text(x0 + idx * 0.225, 0.06, label.upper(), fontsize=9, color="#64705d")
        fig.text(x0 + idx * 0.225, 0.025, value, fontsize=18, color="#1f231d", fontweight="bold")

    fig.savefig(path, facecolor=fig.get_facecolor())
    plt.close(fig)


def render_contact_sheet(image_paths: list[Path], output_path: Path) -> None:
    if not image_paths:
        return
    images = [plt.imread(path) for path in image_paths]
    fig, axes = plt.subplots(2, 2, figsize=(16, 10), dpi=140, facecolor="#f5f2e8")
    for ax, image, path in zip(axes.ravel(), images, image_paths, strict=False):
        ax.imshow(image)
        ax.set_title(path.stem, fontsize=12, color="#1f231d")
        ax.axis("off")
    for ax in axes.ravel()[len(images):]:
        ax.axis("off")
    fig.tight_layout(pad=1.0)
    fig.savefig(output_path, facecolor=fig.get_facecolor())
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Render Kronos demo chart PNGs from API candles and forecasts.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--ids", nargs="*", default=DEFAULT_RENDER_IDS)
    parser.add_argument("--visible-bars", type=int, default=160)
    args = parser.parse_args()

    config = load_config(args.config)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    image_paths: list[Path] = []
    for demo_id in args.ids:
        demo = get_demo(config, demo_id)
        print(f"Rendering {demo.id}: {demo.provider}/{demo.market} {demo.symbol} {demo.interval}")
        payload = run_forecast(config, demo)
        image_path = args.output_dir / f"{demo.id}.png"
        render_chart(payload, image_path, visible_bars=args.visible_bars)
        image_paths.append(image_path)
        print(f"Saved {image_path}")

    contact_sheet = args.output_dir / "btc_sol_1h_5m_grid.png"
    render_contact_sheet(image_paths, contact_sheet)
    print(f"Saved {contact_sheet}")


if __name__ == "__main__":
    main()
