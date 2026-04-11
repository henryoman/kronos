# /// script
# requires-python = ">=3.14"
# dependencies = [
#   "matplotlib",
#   "numpy",
#   "pandas",
#   "scikit-learn",
# ]
# ///

from __future__ import annotations

from itertools import product
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


DATA_PATH = Path("data/BINANCE_BTCUSDT.P, 60.csv")
TRADES_PATH = Path("outputs/trades.csv")

GRID_PATH = Path("outputs/research_grid_results.csv")
BEST_TRADES_PATH = Path("outputs/research_best_test_trades.csv")
BEST_EQUITY_PATH = Path("outputs/research_best_test_equity.png")
CALIBRATION_PATH = Path("outputs/research_calibration.csv")

FEE_BPS_PER_SIDE = 5
SLIPPAGE_BPS_PER_SIDE = 2
ROUND_TRIP_COST_PER_1X = 2 * ((FEE_BPS_PER_SIDE + SLIPPAGE_BPS_PER_SIDE) / 10_000)


def load_price_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH)
    df.columns = [str(c).strip().lower() for c in df.columns]
    df = df.rename(columns={"time": "timestamp", "vol": "volume"})
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s", utc=True).dt.tz_convert(None)
    return df.sort_values("timestamp").reset_index(drop=True)


def load_trades() -> pd.DataFrame:
    df = pd.read_csv(TRADES_PATH, parse_dates=["signal_timestamp", "entry_timestamp", "exit_timestamp"])
    df["actual_up"] = df["exit_close"] > df["current_close"]
    df["entry_up"] = df["exit_close"] > df["entry_open"]
    df["raw_p"] = df["upside_probability"].astype(float)
    df["pred_ret"] = df["forecast_return"].astype(float)
    df["abs_pred_ret"] = df["pred_ret"].abs()
    df["pred_edge"] = df["raw_p"] - 0.5
    return df


def chronological_split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    n = len(df)
    train_end = int(n * 0.60)
    val_end = int(n * 0.80)
    return df.iloc[:train_end].copy(), df.iloc[train_end:val_end].copy(), df.iloc[val_end:].copy()


def fit_scores(train: pd.DataFrame, all_trades: pd.DataFrame) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    features = ["raw_p", "pred_ret", "abs_pred_ret", "pred_edge"]
    y_train = train["actual_up"].astype(int)

    scored = all_trades.copy()
    scored["score_raw"] = scored["raw_p"]

    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(train["raw_p"], y_train)
    scored["score_iso"] = iso.predict(scored["raw_p"])

    logit = make_pipeline(
        StandardScaler(),
        LogisticRegression(C=0.25, solver="lbfgs", max_iter=2000),
    )
    logit.fit(train[features], y_train)
    scored["score_logit"] = logit.predict_proba(scored[features])[:, 1]

    reports = []
    for split_name, split in [
        ("train", scored.iloc[: len(train)]),
        ("validation", scored.iloc[len(train) : int(len(scored) * 0.80)]),
        ("test", scored.iloc[int(len(scored) * 0.80) :]),
    ]:
        y = split["actual_up"].astype(int)
        for score_col in ["score_raw", "score_iso", "score_logit"]:
            p = split[score_col].clip(1e-6, 1 - 1e-6)
            auc = roc_auc_score(y, p) if y.nunique() > 1 else np.nan
            reports.append(
                {
                    "split": split_name,
                    "score": score_col,
                    "brier": brier_score_loss(y, p),
                    "log_loss": log_loss(y, p),
                    "auc": auc,
                    "mean_p": float(p.mean()),
                    "actual_up_rate": float(y.mean()),
                }
            )

    return scored, reports


def max_drawdown(equity: pd.Series) -> float:
    if equity.empty:
        return 0.0
    return float((equity / equity.cummax() - 1.0).min())


def choose_position(row: pd.Series, config: dict[str, Any]) -> int:
    score = float(row[config["score_col"]])
    pred_ret = float(row["pred_ret"])

    if abs(pred_ret) < config["min_abs_pred_ret"]:
        return 0

    if config["side"] in {"both", "long"} and score >= config["long_threshold"]:
        if not config["require_forecast_sign"] or pred_ret > 0:
            return 1

    if config["side"] in {"both", "short"} and score <= config["short_threshold"]:
        if not config["require_forecast_sign"] or pred_ret < 0:
            return -1

    return 0


def simulate_exit(row: pd.Series, price_df: pd.DataFrame, position: int, config: dict[str, Any]) -> tuple[str, float, float]:
    if position == 0:
        return "flat", np.nan, 0.0

    leverage = config["leverage"]
    entry_time = pd.Timestamp(row["entry_timestamp"])
    exit_time = pd.Timestamp(row["exit_timestamp"])
    entry = float(row["entry_open"])
    path = price_df[(price_df["timestamp"] >= entry_time) & (price_df["timestamp"] <= exit_time)]
    if path.empty:
        raise ValueError(f"No path for {entry_time} -> {exit_time}")

    stop_margin = config["stop_margin_loss"]
    take_margin = config["take_margin_gain"]
    stop_move = None if stop_margin is None else max((stop_margin - leverage * ROUND_TRIP_COST_PER_1X) / leverage, 0.0)
    take_move = None if take_margin is None else take_margin / leverage

    stop_price = None
    take_price = None
    if position == 1:
        if stop_move is not None:
            stop_price = entry * (1.0 - stop_move)
        if take_move is not None:
            take_price = entry * (1.0 + take_move)
    else:
        if stop_move is not None:
            stop_price = entry * (1.0 + stop_move)
        if take_move is not None:
            take_price = entry * (1.0 - take_move)

    for _, bar in path.iterrows():
        if position == 1:
            stop_hit = stop_price is not None and float(bar["low"]) <= stop_price
            take_hit = take_price is not None and float(bar["high"]) >= take_price
        else:
            stop_hit = stop_price is not None and float(bar["high"]) >= stop_price
            take_hit = take_price is not None and float(bar["low"]) <= take_price

        if stop_hit:
            exit_price = float(stop_price)
            underlying = position * (exit_price / entry - 1.0)
            return "stop", exit_price, underlying
        if take_hit:
            exit_price = float(take_price)
            underlying = position * (exit_price / entry - 1.0)
            return "take_profit", exit_price, underlying

    exit_price = float(row["exit_close"])
    underlying = position * (exit_price / entry - 1.0)
    return "time", exit_price, underlying


def run_config(trades: pd.DataFrame, price_df: pd.DataFrame, config: dict[str, Any]) -> tuple[dict[str, Any], pd.DataFrame]:
    rows = []
    equity = 1.0
    log_equity = 0.0
    ruined = False

    for _, row in trades.iterrows():
        position = choose_position(row, config)
        exit_reason, exit_price, underlying_return = simulate_exit(row, price_df, position, config)
        cost = config["leverage"] * ROUND_TRIP_COST_PER_1X if position != 0 else 0.0
        net_return = config["leverage"] * underlying_return - cost

        if position != 0 and not ruined:
            if 1.0 + net_return <= 0:
                equity = 0.0
                ruined = True
            else:
                equity *= 1.0 + net_return
                log_equity += np.log1p(net_return)

        rows.append(
            {
                "signal_number": row["signal_number"],
                "signal_timestamp": row["signal_timestamp"],
                "position": position,
                "score": row[config["score_col"]],
                "pred_ret": row["pred_ret"],
                "exit_reason": exit_reason,
                "exit_price": exit_price,
                "underlying_return": underlying_return,
                "net_return": net_return if position != 0 else 0.0,
                "equity": equity,
            }
        )

        if ruined:
            break

    detail = pd.DataFrame(rows)
    taken = detail[detail["position"] != 0]
    summary = {
        **config,
        "bars": len(detail),
        "trades": int(len(taken)),
        "longs": int((taken["position"] == 1).sum()),
        "shorts": int((taken["position"] == -1).sum()),
        "stops": int((taken["exit_reason"] == "stop").sum()),
        "takes": int((taken["exit_reason"] == "take_profit").sum()),
        "win_rate": float((taken["net_return"] > 0).mean()) if len(taken) else 0.0,
        "avg_net": float(taken["net_return"].mean()) if len(taken) else 0.0,
        "total_return": float(equity - 1.0),
        "log_return": float(log_equity),
        "max_drawdown": max_drawdown(detail["equity"]) if len(detail) else 0.0,
        "ruined": bool(ruined),
    }
    return summary, detail


def config_grid() -> list[dict[str, Any]]:
    configs = []
    score_cols = ["score_raw", "score_iso", "score_logit"]
    sides = ["both", "long", "short"]
    long_thresholds = [0.52, 0.55, 0.58, 0.60, 0.65, 0.70, 0.80]
    short_thresholds = [0.48, 0.45, 0.42, 0.40, 0.35, 0.30, 0.20]
    min_abs_pred_rets = [0.0, 0.0025, 0.005, 0.01, 0.015, 0.02]
    leverages = [1.0, 2.0, 3.0, 5.0, 10.0]
    stops = [None, 0.15, 0.30, 0.60]
    takes = [None, 0.15, 0.30, 0.60, 1.00]
    require_signs = [False, True]

    for score_col, side, min_abs, lev, stop, take, require_sign in product(
        score_cols,
        sides,
        min_abs_pred_rets,
        leverages,
        stops,
        takes,
        require_signs,
    ):
        for long_t, short_t in zip(long_thresholds, short_thresholds):
            configs.append(
                {
                    "score_col": score_col,
                    "side": side,
                    "long_threshold": long_t,
                    "short_threshold": short_t,
                    "min_abs_pred_ret": min_abs,
                    "leverage": lev,
                    "stop_margin_loss": stop,
                    "take_margin_gain": take,
                    "require_forecast_sign": require_sign,
                }
            )
    return configs


def select_best(validation_results: pd.DataFrame) -> pd.Series:
    candidates = validation_results[
        (validation_results["trades"] >= 20)
        & (~validation_results["ruined"])
        & (validation_results["max_drawdown"] > -0.70)
    ].copy()
    if candidates.empty:
        candidates = validation_results[(validation_results["trades"] >= 20) & (~validation_results["ruined"])].copy()
    if candidates.empty:
        candidates = validation_results.copy()
    return candidates.sort_values(
        ["log_return", "max_drawdown", "trades"],
        ascending=[False, False, False],
    ).iloc[0]


def main() -> None:
    price_df = load_price_data()
    trades = load_trades()
    train, validation, test = chronological_split(trades)
    scored, calibration_reports = fit_scores(train, trades)

    train_scored = scored.loc[train.index].copy()
    validation_scored = scored.loc[validation.index].copy()
    test_scored = scored.loc[test.index].copy()

    pd.DataFrame(calibration_reports).to_csv(CALIBRATION_PATH, index=False)

    summaries = []
    configs = config_grid()
    best_detail = None
    best_config = None

    for i, config in enumerate(configs, start=1):
        val_summary, _ = run_config(validation_scored, price_df, config)
        train_summary, _ = run_config(train_scored, price_df, config)
        row = {
            "config_id": i,
            **config,
            **{f"train_{k}": v for k, v in train_summary.items() if k not in config},
            **{f"val_{k}": v for k, v in val_summary.items() if k not in config},
        }
        summaries.append(row)

    grid = pd.DataFrame(summaries)
    best = select_best(grid.rename(columns={c: c[4:] for c in grid.columns if c.startswith("val_")}))
    config_keys = [
        "score_col",
        "side",
        "long_threshold",
        "short_threshold",
        "min_abs_pred_ret",
        "leverage",
        "stop_margin_loss",
        "take_margin_gain",
        "require_forecast_sign",
    ]
    best_config = {key: best[key] for key in config_keys}
    test_summary, best_detail = run_config(test_scored, price_df, best_config)

    for key, value in test_summary.items():
        if key not in best_config:
            grid.loc[grid["config_id"] == best["config_id"], f"selected_test_{key}"] = value

    grid = grid.sort_values("val_log_return", ascending=False)
    grid.to_csv(GRID_PATH, index=False)
    best_detail.to_csv(BEST_TRADES_PATH, index=False)

    plt.figure(figsize=(10, 5))
    plt.plot(pd.to_datetime(best_detail["signal_timestamp"]), best_detail["equity"])
    plt.title("Selected Strategy Test Equity")
    plt.xlabel("Signal time")
    plt.ylabel("Equity")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(BEST_EQUITY_PATH, dpi=150)
    plt.close()

    print("Calibration by split/score")
    print(pd.DataFrame(calibration_reports).to_string(index=False))
    print()
    print("Selected config from validation")
    for key, value in best_config.items():
        print(f"{key}: {value}")
    print()
    print("Untouched test result")
    for key in [
        "trades",
        "longs",
        "shorts",
        "stops",
        "takes",
        "win_rate",
        "avg_net",
        "total_return",
        "max_drawdown",
        "ruined",
    ]:
        value = test_summary[key]
        if isinstance(value, float):
            print(f"{key}: {value:.4%}")
        else:
            print(f"{key}: {value}")
    print()
    print(f"saved_grid: {GRID_PATH}")
    print(f"saved_best_test_trades: {BEST_TRADES_PATH}")
    print(f"saved_best_test_equity: {BEST_EQUITY_PATH}")
    print(f"saved_calibration: {CALIBRATION_PATH}")


if __name__ == "__main__":
    main()
