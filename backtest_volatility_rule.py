# /// script
# requires-python = ">=3.14"
# dependencies = [
#   "matplotlib",
#   "numpy",
#   "pandas",
# ]
# ///

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ENRICHED_PATH = Path("outputs/volatility_enriched_trades.csv")
SUMMARY_PATH = Path("outputs/volatility_rule_summary.csv")
DETAIL_PATH = Path("outputs/volatility_rule_trades.csv")
PLOT_PATH = Path("outputs/volatility_rule_equity.png")

FEE_BPS_PER_SIDE = 5
SLIPPAGE_BPS_PER_SIDE = 2
ROUND_TRIP_COST_PER_1X = 2 * ((FEE_BPS_PER_SIDE + SLIPPAGE_BPS_PER_SIDE) / 10_000)


def max_drawdown(equity: pd.Series) -> float:
    if equity.empty:
        return 0.0
    return float((equity / equity.cummax() - 1.0).min())


def run_rule(df: pd.DataFrame, leverage: float, stop_margin_loss: float | None) -> tuple[dict, pd.DataFrame]:
    mask = (df["upside_probability"] == 1.0) & (df["pre_vol_24_live_percentile"] >= 0.80)
    trades = df[mask].copy()

    if stop_margin_loss is not None:
        max_underlying_loss = max((stop_margin_loss - leverage * ROUND_TRIP_COST_PER_1X) / leverage, 0.0)
        underlying = trades["exit_close"] / trades["entry_open"] - 1.0
        stopped = underlying < -max_underlying_loss
        trades["rule_exit_reason"] = np.where(stopped, "stop", "time")
        trades["rule_underlying_return"] = np.where(stopped, -max_underlying_loss, underlying)
    else:
        trades["rule_exit_reason"] = "time"
        trades["rule_underlying_return"] = trades["exit_close"] / trades["entry_open"] - 1.0

    trades["rule_net_return"] = leverage * trades["rule_underlying_return"] - leverage * ROUND_TRIP_COST_PER_1X
    trades["rule_equity"] = (1.0 + trades["rule_net_return"]).cumprod()

    summary = {
        "split": "all",
        "leverage": leverage,
        "stop_margin_loss": stop_margin_loss,
        "trades": len(trades),
        "win_rate": float((trades["rule_net_return"] > 0).mean()) if len(trades) else 0.0,
        "avg_net_return": float(trades["rule_net_return"].mean()) if len(trades) else 0.0,
        "total_return": float(trades["rule_equity"].iloc[-1] - 1.0) if len(trades) else 0.0,
        "max_drawdown": max_drawdown(trades["rule_equity"]) if len(trades) else 0.0,
        "stops": int((trades["rule_exit_reason"] == "stop").sum()) if len(trades) else 0,
    }
    return summary, trades


def split_name(signal_number: pd.Series, n_total: int) -> pd.Series:
    train_end = int(n_total * 0.60)
    val_end = int(n_total * 0.80)
    return pd.cut(
        signal_number,
        bins=[0, train_end, val_end, n_total],
        labels=["train", "validation", "test"],
        include_lowest=True,
    )


def summarize_split(trades: pd.DataFrame, leverage: float, stop_margin_loss: float | None, split: str) -> dict:
    subset = trades[trades["split"] == split]
    return {
        "split": split,
        "leverage": leverage,
        "stop_margin_loss": stop_margin_loss,
        "trades": len(subset),
        "win_rate": float((subset["rule_net_return"] > 0).mean()) if len(subset) else 0.0,
        "avg_net_return": float(subset["rule_net_return"].mean()) if len(subset) else 0.0,
        "total_return": float((1.0 + subset["rule_net_return"]).prod() - 1.0) if len(subset) else 0.0,
        "max_drawdown": max_drawdown((1.0 + subset["rule_net_return"]).cumprod()) if len(subset) else 0.0,
        "stops": int((subset["rule_exit_reason"] == "stop").sum()) if len(subset) else 0,
    }


def main() -> None:
    df = pd.read_csv(ENRICHED_PATH, parse_dates=["signal_timestamp", "entry_timestamp", "exit_timestamp"])
    n_total = int(df["signal_number"].max())
    summaries = []
    details = []

    for leverage in [1.0, 2.0, 3.0, 5.0, 10.0]:
        for stop_margin_loss in [None, 0.15, 0.30, 0.60]:
            summary, trades = run_rule(df, leverage, stop_margin_loss)
            trades["leverage"] = leverage
            trades["stop_margin_loss"] = stop_margin_loss
            trades["split"] = split_name(trades["signal_number"], n_total)
            summaries.append(summary)
            for split in ["train", "validation", "test"]:
                summaries.append(summarize_split(trades, leverage, stop_margin_loss, split))
            details.append(trades)

    summary_df = pd.DataFrame(summaries).sort_values(["split", "total_return"], ascending=[True, False])
    detail_df = pd.concat(details, ignore_index=True)

    summary_df.to_csv(SUMMARY_PATH, index=False)
    detail_df.to_csv(DETAIL_PATH, index=False)

    best_all = summary_df[summary_df["split"] == "all"].sort_values("total_return", ascending=False).iloc[0]
    best_detail = detail_df[
        (detail_df["leverage"] == best_all["leverage"])
        & (
            (detail_df["stop_margin_loss"].isna() & pd.isna(best_all["stop_margin_loss"]))
            | (detail_df["stop_margin_loss"] == best_all["stop_margin_loss"])
        )
    ].copy()

    plt.figure(figsize=(10, 5))
    plt.plot(pd.to_datetime(best_detail["signal_timestamp"]), best_detail["rule_equity"])
    plt.title("Best Volatility Rule Equity")
    plt.xlabel("Signal time")
    plt.ylabel("Equity")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(PLOT_PATH, dpi=150)
    plt.close()

    print("Rule: long only when p_up == 1.0 and prior 24h volatility percentile >= 80%")
    print()
    print(summary_df.to_string(index=False, formatters={
        "win_rate": "{:.2%}".format,
        "avg_net_return": "{:.4%}".format,
        "total_return": "{:.2%}".format,
        "max_drawdown": "{:.2%}".format,
    }))
    print()
    print(f"saved_summary: {SUMMARY_PATH}")
    print(f"saved_detail: {DETAIL_PATH}")
    print(f"saved_plot: {PLOT_PATH}")


if __name__ == "__main__":
    main()
