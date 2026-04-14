# /// script
# requires-python = ">=3.14"
# dependencies = [
#   "numpy",
#   "pandas",
#   "scikit-learn",
# ]
# ///

from __future__ import annotations

from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


INPUT_PATH = Path("outputs/multi_asset/enriched_5m_trades.csv")
OUT_DIR = Path("outputs/clean_walkforward_research")
GRID_PATH = OUT_DIR / "grid.csv"
BEST_PATH = OUT_DIR / "best_configs.csv"
TEST_SUMMARY_PATH = OUT_DIR / "test_summary.csv"
TEST_TRADES_PATH = OUT_DIR / "test_trades.csv"

ROUND_TRIP_COST = 0.0014
N_BLOCKS = 7
VALIDATION_BLOCKS = (3, 4, 5)
TEST_BLOCK = 6
SEED = 42

BASE_FEATURES = [
    "raw_p",
    "pred_ret",
    "abs_pred_ret",
    "ret_24",
    "ret_72",
    "ret_288",
    "atr_24_pct",
    "vol_24",
    "vol_72",
    "volume_ratio_24",
    "rsi_14",
    "above_sma_24",
    "above_sma_72",
    "above_sma_288",
    "atr_24_pct_live_pct",
    "vol_24_live_pct",
    "vol_72_live_pct",
    "volume_ratio_24_live_pct",
    "rsi_14_live_pct",
    "ret_24_live_pct",
    "ret_72_live_pct",
    "hour_sin",
    "hour_cos",
    "dow_sin",
    "dow_cos",
    "is_weekend",
]

MODEL_NAMES = ("score_raw", "score_iso", "score_logit", "score_hgb")
SIDES = ("both", "long", "short")
LONG_THRESHOLDS = (0.55, 0.60, 0.65, 0.70, 0.75, 0.80)
SHORT_THRESHOLDS = (0.45, 0.40, 0.35, 0.30, 0.25, 0.20)
MIN_ABS_PRED_RET = (0.0, 0.001, 0.002, 0.005, 0.01)


def assign_blocks(asset_df: pd.DataFrame, n_blocks: int = N_BLOCKS) -> pd.DataFrame:
    asset_df = asset_df.sort_values("signal_timestamp").copy()
    positions = np.arange(len(asset_df))
    asset_df["block"] = np.floor(positions * n_blocks / len(asset_df)).astype(int)
    return asset_df


def load_data() -> pd.DataFrame:
    df = pd.read_csv(
        INPUT_PATH,
        parse_dates=["signal_timestamp", "entry_timestamp", "exit_timestamp"],
    )
    df = df.sort_values(["asset", "signal_timestamp"]).reset_index(drop=True)
    df["raw_p"] = df["upside_probability"].astype(float)
    df["pred_ret"] = df["forecast_return"].astype(float)
    df["abs_pred_ret"] = df["pred_ret"].abs()
    df["target_up"] = df["actual_up_entry"].astype(int)

    hour = df["signal_timestamp"].dt.hour.astype(float)
    weekday = df["signal_timestamp"].dt.weekday.astype(float)
    df["hour_sin"] = np.sin(2 * np.pi * hour / 24.0)
    df["hour_cos"] = np.cos(2 * np.pi * hour / 24.0)
    df["dow_sin"] = np.sin(2 * np.pi * weekday / 7.0)
    df["dow_cos"] = np.cos(2 * np.pi * weekday / 7.0)
    df["is_weekend"] = (weekday >= 5).astype(int)

    bool_cols = ["above_sma_24", "above_sma_72", "above_sma_288"]
    for col in bool_cols:
        df[col] = df[col].astype(int)

    df = (
        pd.concat(
            [assign_blocks(asset_df) for _, asset_df in df.groupby("asset", sort=False)],
            ignore_index=True,
        )
        .sort_values(["asset", "signal_timestamp"])
        .reset_index(drop=True)
    )

    required_cols = BASE_FEATURES + ["entry_open", "exit_close", "target_up"]
    df = df.dropna(subset=required_cols).reset_index(drop=True)
    return df


def design_matrix(df: pd.DataFrame, include_asset: bool) -> pd.DataFrame:
    X = df[BASE_FEATURES].astype(float).copy()
    if include_asset:
        asset_dummies = pd.get_dummies(df["asset"], prefix="asset", dtype=float)
        X = pd.concat([X, asset_dummies], axis=1)
    return X


def score_models(train_df: pd.DataFrame, target_df: pd.DataFrame, include_asset: bool) -> pd.DataFrame:
    scored = target_df.copy()
    scored["score_raw"] = scored["raw_p"].astype(float)

    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(train_df["raw_p"], train_df["target_up"])
    scored["score_iso"] = iso.predict(target_df["raw_p"])

    X_train = design_matrix(train_df, include_asset)
    X_target = design_matrix(target_df, include_asset).reindex(columns=X_train.columns, fill_value=0.0)
    y_train = train_df["target_up"].astype(int)

    logit = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            C=0.25,
            solver="lbfgs",
            max_iter=2000,
            class_weight="balanced",
            random_state=SEED,
        ),
    )
    logit.fit(X_train, y_train)
    scored["score_logit"] = logit.predict_proba(X_target)[:, 1]

    hgb = HistGradientBoostingClassifier(
        learning_rate=0.05,
        max_depth=3,
        max_iter=250,
        min_samples_leaf=20,
        l2_regularization=1.0,
        random_state=SEED,
    )
    hgb.fit(X_train, y_train)
    scored["score_hgb"] = hgb.predict_proba(X_target)[:, 1]

    return scored


def apply_policy(
    df: pd.DataFrame,
    score_col: str,
    side: str,
    long_t: float,
    short_t: float,
    min_abs_pred_ret: float,
) -> tuple[dict[str, float | int], pd.DataFrame]:
    eligible = df[df["abs_pred_ret"] >= min_abs_pred_ret].copy()
    if eligible.empty:
        return {
            "trades": 0,
            "accuracy": 0.0,
            "avg_gross_return": 0.0,
            "avg_net_return": 0.0,
            "compounded_return": 0.0,
            "max_drawdown": 0.0,
            "profit_factor": 0.0,
        }, eligible

    score = eligible[score_col].astype(float)
    position = np.zeros(len(eligible), dtype=int)
    if side in {"both", "long"}:
        position = np.where(score >= long_t, 1, position)
    if side in {"both", "short"}:
        position = np.where(score <= short_t, -1, position)

    eligible["policy_score"] = score
    eligible["policy_position"] = position
    taken = eligible[eligible["policy_position"] != 0].copy()
    if taken.empty:
        return {
            "trades": 0,
            "accuracy": 0.0,
            "avg_gross_return": 0.0,
            "avg_net_return": 0.0,
            "compounded_return": 0.0,
            "max_drawdown": 0.0,
            "profit_factor": 0.0,
        }, taken

    gross_return = np.where(
        taken["policy_position"] == 1,
        taken["exit_close"] / taken["entry_open"] - 1.0,
        1.0 - taken["exit_close"] / taken["entry_open"],
    )
    correct = np.where(
        taken["policy_position"] == 1,
        taken["target_up"].astype(bool),
        ~taken["target_up"].astype(bool),
    )

    taken["policy_gross_return"] = gross_return
    taken["policy_net_return"] = taken["policy_gross_return"] - ROUND_TRIP_COST
    taken["policy_correct"] = correct

    step_returns = (
        taken.groupby("signal_timestamp", sort=True)["policy_net_return"]
        .mean()
        .astype(float)
    )
    equity = (1.0 + step_returns).cumprod()
    max_drawdown = 0.0 if equity.empty else float((equity / equity.cummax() - 1.0).min())

    gains = taken.loc[taken["policy_net_return"] > 0.0, "policy_net_return"]
    losses = taken.loc[taken["policy_net_return"] < 0.0, "policy_net_return"]
    if losses.empty:
        profit_factor = float("inf") if not gains.empty else 0.0
    else:
        profit_factor = float(gains.sum() / abs(losses.sum()))

    metrics = {
        "trades": int(len(taken)),
        "accuracy": float(taken["policy_correct"].mean()),
        "avg_gross_return": float(taken["policy_gross_return"].mean()),
        "avg_net_return": float(taken["policy_net_return"].mean()),
        "compounded_return": float(equity.iloc[-1] - 1.0) if not equity.empty else 0.0,
        "max_drawdown": max_drawdown,
        "profit_factor": profit_factor,
    }
    return metrics, taken


def validation_predictions(scope_df: pd.DataFrame, include_asset: bool) -> pd.DataFrame:
    fold_frames = []
    for val_block in VALIDATION_BLOCKS:
        train = scope_df[scope_df["block"] < val_block].copy()
        validation = scope_df[scope_df["block"] == val_block].copy()
        if train.empty or validation.empty:
            continue
        scored = score_models(train, validation, include_asset=include_asset)
        scored["fold_block"] = val_block
        fold_frames.append(scored)
    if not fold_frames:
        return pd.DataFrame()
    return pd.concat(fold_frames, ignore_index=True)


def candidate_rows(scope_name: str, validation_scored: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, float | int | str]] = []
    for score_col, side, long_t, short_t, min_abs_pred_ret in product(
        MODEL_NAMES,
        SIDES,
        LONG_THRESHOLDS,
        SHORT_THRESHOLDS,
        MIN_ABS_PRED_RET,
    ):
        validation_metrics, _ = apply_policy(
            validation_scored,
            score_col=score_col,
            side=side,
            long_t=long_t,
            short_t=short_t,
            min_abs_pred_ret=min_abs_pred_ret,
        )
        fold_net_returns = []
        for _, fold_df in validation_scored.groupby("fold_block", sort=True):
            fold_metrics, _ = apply_policy(
                fold_df,
                score_col=score_col,
                side=side,
                long_t=long_t,
                short_t=short_t,
                min_abs_pred_ret=min_abs_pred_ret,
            )
            if fold_metrics["trades"] > 0:
                fold_net_returns.append(float(fold_metrics["avg_net_return"]))

        positive_fold_rate = 0.0 if not fold_net_returns else float(np.mean(np.array(fold_net_returns) > 0.0))
        rows.append(
            {
                "scope": scope_name,
                "score_col": score_col,
                "side": side,
                "long_threshold": long_t,
                "short_threshold": short_t,
                "min_abs_pred_ret": min_abs_pred_ret,
                "validation_positive_fold_rate": positive_fold_rate,
                **{f"validation_{key}": value for key, value in validation_metrics.items()},
            }
        )
    return pd.DataFrame(rows)


def pick_best_candidate(grid: pd.DataFrame) -> pd.Series:
    candidates = grid[
        (grid["validation_trades"] >= 30)
        & (grid["validation_positive_fold_rate"] >= (2.0 / 3.0))
    ].copy()
    if candidates.empty:
        candidates = grid[grid["validation_trades"] >= 30].copy()
    if candidates.empty:
        candidates = grid.copy()
    return candidates.sort_values(
        [
            "validation_avg_net_return",
            "validation_compounded_return",
            "validation_profit_factor",
            "validation_positive_fold_rate",
            "validation_accuracy",
            "validation_trades",
        ],
        ascending=[False, False, False, False, False, False],
    ).iloc[0]


def evaluate_scope(scope_name: str, scope_df: pd.DataFrame, include_asset: bool) -> tuple[pd.DataFrame, dict, pd.DataFrame]:
    validation_scored = validation_predictions(scope_df, include_asset=include_asset)
    if validation_scored.empty:
        raise ValueError(f"No validation folds available for scope {scope_name}")

    grid = candidate_rows(scope_name, validation_scored)
    best = pick_best_candidate(grid)

    dev_df = scope_df[scope_df["block"] < TEST_BLOCK].copy()
    test_df = scope_df[scope_df["block"] == TEST_BLOCK].copy()
    test_scored = score_models(dev_df, test_df, include_asset=include_asset)

    test_metrics, test_trades = apply_policy(
        test_scored,
        score_col=str(best["score_col"]),
        side=str(best["side"]),
        long_t=float(best["long_threshold"]),
        short_t=float(best["short_threshold"]),
        min_abs_pred_ret=float(best["min_abs_pred_ret"]),
    )
    baseline_metrics, _ = apply_policy(
        test_scored,
        score_col="score_raw",
        side="both",
        long_t=0.60,
        short_t=0.40,
        min_abs_pred_ret=0.0,
    )

    summary = {
        "scope": scope_name,
        "include_asset_dummies": include_asset,
        "score_col": str(best["score_col"]),
        "side": str(best["side"]),
        "long_threshold": float(best["long_threshold"]),
        "short_threshold": float(best["short_threshold"]),
        "min_abs_pred_ret": float(best["min_abs_pred_ret"]),
        **{f"validation_{key}": best[f"validation_{key}"] for key in [
            "trades",
            "accuracy",
            "avg_gross_return",
            "avg_net_return",
            "compounded_return",
            "max_drawdown",
            "profit_factor",
        ]},
        "validation_positive_fold_rate": float(best["validation_positive_fold_rate"]),
        **{f"test_{key}": value for key, value in test_metrics.items()},
        **{f"baseline_{key}": value for key, value in baseline_metrics.items()},
        "delta_test_avg_net_return": float(test_metrics["avg_net_return"] - baseline_metrics["avg_net_return"]),
    }

    if not test_trades.empty:
        test_trades = test_trades.copy()
        test_trades["scope"] = scope_name
        test_trades["score_col"] = str(best["score_col"])
        test_trades["side"] = str(best["side"])
        test_trades["long_threshold"] = float(best["long_threshold"])
        test_trades["short_threshold"] = float(best["short_threshold"])
        test_trades["min_abs_pred_ret"] = float(best["min_abs_pred_ret"])

    return grid, summary, test_trades


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = load_data()

    scopes: list[tuple[str, pd.DataFrame, bool]] = [("all_assets_pooled", df.copy(), True)]
    for asset, asset_df in df.groupby("asset", sort=False):
        scopes.append((str(asset), asset_df.copy(), False))

    grids = []
    summaries = []
    test_trades = []

    for scope_name, scope_df, include_asset in scopes:
        grid, summary, trades = evaluate_scope(scope_name, scope_df, include_asset)
        grids.append(grid)
        summaries.append(summary)
        if not trades.empty:
            test_trades.append(trades)

    full_grid = pd.concat(grids, ignore_index=True)
    summary_df = pd.DataFrame(summaries).sort_values("test_avg_net_return", ascending=False).reset_index(drop=True)
    trades_df = pd.concat(test_trades, ignore_index=True) if test_trades else pd.DataFrame()

    full_grid.to_csv(GRID_PATH, index=False)
    summary_df.to_csv(BEST_PATH, index=False)
    summary_df.to_csv(TEST_SUMMARY_PATH, index=False)
    if not trades_df.empty:
        trades_df.to_csv(TEST_TRADES_PATH, index=False)

    display_cols = [
        "scope",
        "score_col",
        "side",
        "long_threshold",
        "short_threshold",
        "min_abs_pred_ret",
        "validation_trades",
        "validation_accuracy",
        "validation_avg_net_return",
        "test_trades",
        "test_accuracy",
        "test_avg_net_return",
        "baseline_avg_net_return",
        "delta_test_avg_net_return",
    ]
    print("Walk-forward best configs")
    print(
        summary_df[display_cols].to_string(
            index=False,
            formatters={
                "validation_accuracy": "{:.2%}".format,
                "validation_avg_net_return": "{:.4%}".format,
                "test_accuracy": "{:.2%}".format,
                "test_avg_net_return": "{:.4%}".format,
                "baseline_avg_net_return": "{:.4%}".format,
                "delta_test_avg_net_return": "{:.4%}".format,
            },
        )
    )
    print()
    print(f"saved_grid: {GRID_PATH}")
    print(f"saved_best: {BEST_PATH}")
    print(f"saved_test_summary: {TEST_SUMMARY_PATH}")
    if not trades_df.empty:
        print(f"saved_test_trades: {TEST_TRADES_PATH}")


if __name__ == "__main__":
    main()
