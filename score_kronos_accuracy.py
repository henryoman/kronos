# /// script
# requires-python = ">=3.14"
# dependencies = [
#   "matplotlib",
#   "pandas",
# ]
# ///

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


TRADES_PATH = Path("outputs/trades.csv")
SUMMARY_PATH = Path("outputs/accuracy_summary.csv")
CALIBRATION_PATH = Path("outputs/calibration_by_probability.csv")
PLOT_PATH = Path("outputs/calibration.png")


def main() -> None:
    trades = pd.read_csv(TRADES_PATH)

    trades["actual_up_from_current"] = trades["exit_close"] > trades["current_close"]
    trades["mean_pred_up"] = trades["mean_final_pred_close"] > trades["current_close"]
    trades["forecast_direction_correct"] = trades["mean_pred_up"] == trades["actual_up_from_current"]
    trades["net_trade_win"] = trades["net_return"] > 0

    trades["actual_dir_from_entry"] = trades["exit_close"].gt(trades["entry_open"]).map({True: 1, False: -1})
    trades["prob_signal"] = trades["upside_probability"].map(lambda p: 1 if p >= 0.60 else -1 if p <= 0.40 else 0)
    trades["prob_signal_correct"] = trades["prob_signal"] == trades["actual_dir_from_entry"]

    summary = pd.DataFrame(
        [
            {
                "signals": len(trades),
                "mean_forecast_direction_accuracy": trades["forecast_direction_correct"].mean(),
                "probability_signal_accuracy": trades["prob_signal_correct"].mean(),
                "gross_trade_win_rate": (trades["gross_return"] > 0).mean(),
                "net_trade_win_rate": trades["net_trade_win"].mean(),
                "average_net_return": trades["net_return"].mean(),
                "total_compounded_return": (1.0 + trades["net_return"]).prod() - 1.0,
            }
        ]
    )

    calibration = (
        trades.groupby("upside_probability")
        .agg(
            n=("signal_number", "count"),
            actual_up_rate=("actual_up_from_current", "mean"),
            signal_correct_rate=("prob_signal_correct", "mean"),
            average_net_return=("net_return", "mean"),
        )
        .reset_index()
    )

    summary.to_csv(SUMMARY_PATH, index=False)
    calibration.to_csv(CALIBRATION_PATH, index=False)

    plt.figure(figsize=(8, 5))
    plt.plot(calibration["upside_probability"], calibration["actual_up_rate"], marker="o", label="Actual up rate")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Perfect calibration")
    for _, row in calibration.iterrows():
        plt.text(row["upside_probability"], row["actual_up_rate"], f"n={int(row['n'])}", fontsize=8)
    plt.title("Kronos Probability Calibration")
    plt.xlabel("Predicted upside probability")
    plt.ylabel("Actual 24h up rate")
    plt.ylim(0, 1)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(PLOT_PATH, dpi=150)
    plt.close()

    print(summary.to_string(index=False, formatters={
        "mean_forecast_direction_accuracy": "{:.2%}".format,
        "probability_signal_accuracy": "{:.2%}".format,
        "gross_trade_win_rate": "{:.2%}".format,
        "net_trade_win_rate": "{:.2%}".format,
        "average_net_return": "{:.4%}".format,
        "total_compounded_return": "{:.2%}".format,
    }))
    print()
    print(calibration.to_string(index=False, formatters={
        "actual_up_rate": "{:.2%}".format,
        "signal_correct_rate": "{:.2%}".format,
        "average_net_return": "{:.4%}".format,
    }))
    print(f"saved_summary: {SUMMARY_PATH}")
    print(f"saved_calibration: {CALIBRATION_PATH}")
    print(f"saved_plot: {PLOT_PATH}")


if __name__ == "__main__":
    main()
