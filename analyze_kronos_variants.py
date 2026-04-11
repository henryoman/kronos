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


DATA_PATH = Path("data/BINANCE_BTCUSDT.P, 60.csv")
TRADES_PATH = Path("outputs/trades.csv")
SUMMARY_PATH = Path("outputs/variant_summary.csv")
EQUITY_PATH = Path("outputs/variant_equity.png")
BEST_TRADES_PATH = Path("outputs/variant_trades_best.csv")

LEVERAGE = 10.0
MAX_MARGIN_LOSS = 0.60

FEE_BPS_PER_SIDE = 5
SLIPPAGE_BPS_PER_SIDE = 2
ROUND_TRIP_COST = LEVERAGE * 2 * ((FEE_BPS_PER_SIDE + SLIPPAGE_BPS_PER_SIDE) / 10_000)
STOP_UNDERLYING_MOVE = (MAX_MARGIN_LOSS - ROUND_TRIP_COST) / LEVERAGE


VARIANTS = [
    {"name": "p60_40_both", "long_p": 0.60, "short_p": 0.40, "side": "both", "min_abs_fcst": 0.00},
    {"name": "p80_20_both", "long_p": 0.80, "short_p": 0.20, "side": "both", "min_abs_fcst": 0.00},
    {"name": "p100_0_both", "long_p": 1.00, "short_p": 0.00, "side": "both", "min_abs_fcst": 0.00},
    {"name": "p60_40_fcst_0p5", "long_p": 0.60, "short_p": 0.40, "side": "both", "min_abs_fcst": 0.005},
    {"name": "p60_40_fcst_1p0", "long_p": 0.60, "short_p": 0.40, "side": "both", "min_abs_fcst": 0.010},
    {"name": "p80_20_fcst_0p5", "long_p": 0.80, "short_p": 0.20, "side": "both", "min_abs_fcst": 0.005},
    {"name": "long_p60", "long_p": 0.60, "short_p": None, "side": "long", "min_abs_fcst": 0.00},
    {"name": "long_p80", "long_p": 0.80, "short_p": None, "side": "long", "min_abs_fcst": 0.00},
    {"name": "long_p100", "long_p": 1.00, "short_p": None, "side": "long", "min_abs_fcst": 0.00},
    {"name": "short_p40", "long_p": None, "short_p": 0.40, "side": "short", "min_abs_fcst": 0.00},
    {"name": "short_p20", "long_p": None, "short_p": 0.20, "side": "short", "min_abs_fcst": 0.00},
    {"name": "short_p0", "long_p": None, "short_p": 0.00, "side": "short", "min_abs_fcst": 0.00},
]


def load_price_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH)
    df.columns = [str(c).strip().lower() for c in df.columns]
    df = df.rename(columns={"time": "timestamp", "volume": "volume", "vol": "volume"})
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s", utc=True).dt.tz_convert(None)
    return df.sort_values("timestamp").reset_index(drop=True)


def max_drawdown(equity: pd.Series) -> float:
    if equity.empty:
        return 0.0
    return float((equity / equity.cummax() - 1.0).min())


def choose_position(row: pd.Series, variant: dict) -> int:
    p_up = float(row["upside_probability"])
    forecast_return = float(row["forecast_return"])
    min_abs_fcst = float(variant["min_abs_fcst"])

    if abs(forecast_return) < min_abs_fcst:
        return 0

    if variant["side"] in {"both", "long"} and variant["long_p"] is not None:
        if p_up >= variant["long_p"] and forecast_return > 0:
            return 1
        if p_up >= variant["long_p"] and min_abs_fcst == 0.0:
            return 1

    if variant["side"] in {"both", "short"} and variant["short_p"] is not None:
        if p_up <= variant["short_p"] and forecast_return < 0:
            return -1
        if p_up <= variant["short_p"] and min_abs_fcst == 0.0:
            return -1

    return 0


def simulate_trade(row: pd.Series, price_df: pd.DataFrame, position: int) -> dict:
    if position == 0:
        return {
            "exit_reason": "flat",
            "exit_price": np.nan,
            "underlying_return": 0.0,
            "net_return": 0.0,
            "stopped": False,
        }

    entry_time = pd.Timestamp(row["entry_timestamp"])
    exit_time = pd.Timestamp(row["exit_timestamp"])
    entry_open = float(row["entry_open"])

    path = price_df[(price_df["timestamp"] >= entry_time) & (price_df["timestamp"] <= exit_time)]
    if path.empty:
        raise ValueError(f"No price path for {entry_time} -> {exit_time}")

    if position == 1:
        stop_price = entry_open * (1.0 - STOP_UNDERLYING_MOVE)
        hit = path[path["low"] <= stop_price]
        if not hit.empty:
            exit_price = stop_price
            exit_reason = "stop"
            stopped = True
        else:
            exit_price = float(row["exit_close"])
            exit_reason = "time"
            stopped = False
        underlying_return = exit_price / entry_open - 1.0
    else:
        stop_price = entry_open * (1.0 + STOP_UNDERLYING_MOVE)
        hit = path[path["high"] >= stop_price]
        if not hit.empty:
            exit_price = stop_price
            exit_reason = "stop"
            stopped = True
        else:
            exit_price = float(row["exit_close"])
            exit_reason = "time"
            stopped = False
        underlying_return = 1.0 - exit_price / entry_open

    leveraged_return = underlying_return * LEVERAGE
    net_return = leveraged_return - ROUND_TRIP_COST

    return {
        "exit_reason": exit_reason,
        "exit_price": exit_price,
        "underlying_return": underlying_return,
        "net_return": net_return,
        "stopped": stopped,
    }


def run_variant(trades: pd.DataFrame, price_df: pd.DataFrame, variant: dict) -> tuple[dict, pd.DataFrame]:
    rows = []
    equity = 1.0
    ruined_at = None

    for _, row in trades.iterrows():
        position = choose_position(row, variant)
        result = simulate_trade(row, price_df, position)
        net_return = result["net_return"]

        if position != 0:
            equity *= 1.0 + net_return
            if equity <= 0 and ruined_at is None:
                ruined_at = row["signal_timestamp"]
                equity = 0.0

        rows.append(
            {
                "signal_number": row["signal_number"],
                "signal_timestamp": row["signal_timestamp"],
                "entry_timestamp": row["entry_timestamp"],
                "exit_timestamp": row["exit_timestamp"],
                "upside_probability": row["upside_probability"],
                "forecast_return": row["forecast_return"],
                "position": position,
                "exit_reason": result["exit_reason"],
                "exit_price": result["exit_price"],
                "underlying_return": result["underlying_return"],
                "net_return": net_return,
                "equity": equity,
            }
        )

        if ruined_at is not None:
            break

    detail = pd.DataFrame(rows)
    taken = detail[detail["position"] != 0]
    summary = {
        "variant": variant["name"],
        "trades": int(len(taken)),
        "longs": int((taken["position"] == 1).sum()),
        "shorts": int((taken["position"] == -1).sum()),
        "stops": int((taken["exit_reason"] == "stop").sum()),
        "stop_rate": float((taken["exit_reason"] == "stop").mean()) if len(taken) else 0.0,
        "total_return": float(detail["equity"].iloc[-1] - 1.0) if len(detail) else 0.0,
        "win_rate": float((taken["net_return"] > 0).mean()) if len(taken) else 0.0,
        "max_drawdown": max_drawdown(detail["equity"]) if len(detail) else 0.0,
        "ruined_at": ruined_at,
    }
    return summary, detail


def main() -> None:
    price_df = load_price_data()
    trades = pd.read_csv(TRADES_PATH, parse_dates=["signal_timestamp", "entry_timestamp", "exit_timestamp"])

    summaries = []
    details = {}
    for variant in VARIANTS:
        summary, detail = run_variant(trades, price_df, variant)
        summaries.append(summary)
        details[variant["name"]] = detail

    summary_df = pd.DataFrame(summaries).sort_values("total_return", ascending=False)
    summary_df.to_csv(SUMMARY_PATH, index=False)

    best_name = str(summary_df.iloc[0]["variant"])
    details[best_name].to_csv(BEST_TRADES_PATH, index=False)

    plt.figure(figsize=(12, 6))
    for name in summary_df.head(6)["variant"]:
        detail = details[str(name)]
        plt.plot(pd.to_datetime(detail["signal_timestamp"]), detail["equity"], label=str(name))
    plt.title("Kronos 10x Variant Equity")
    plt.xlabel("Signal time")
    plt.ylabel("Equity")
    plt.yscale("log")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(EQUITY_PATH, dpi=150)
    plt.close()

    print(summary_df.to_string(index=False, formatters={
        "total_return": "{:.2%}".format,
        "win_rate": "{:.2%}".format,
        "max_drawdown": "{:.2%}".format,
        "stop_rate": "{:.2%}".format,
    }))
    print(f"saved_summary: {SUMMARY_PATH}")
    print(f"saved_best_trades: {BEST_TRADES_PATH}")
    print(f"saved_equity: {EQUITY_PATH}")


if __name__ == "__main__":
    main()
