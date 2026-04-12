# /// script
# requires-python = ">=3.14"
# dependencies = [
#   "pandas",
# ]
# ///

from __future__ import annotations

from math import sqrt
from pathlib import Path

import pandas as pd


VOL_SUBSETS_PATH = Path("outputs/volatility_best_subsets.csv")
REGIME_RESULTS_PATH = Path("outputs/regime_search_results.csv")
REPORT_PATH = Path("outputs/confidence_report.csv")


def wilson_interval(wins: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return 0.0, 0.0
    p = wins / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    margin = z * sqrt((p * (1 - p) + z * z / (4 * n)) / n) / denom
    return center - margin, center + margin


def main() -> None:
    vol = pd.read_csv(VOL_SUBSETS_PATH)
    vol = vol[vol["leakage_safe"] == True].copy()
    vol["wins"] = (vol["direction_accuracy"] * vol["n"]).round().astype(int)
    intervals = vol.apply(lambda r: wilson_interval(int(r["wins"]), int(r["n"])), axis=1)
    vol["accuracy_ci_low"] = [x[0] for x in intervals]
    vol["accuracy_ci_high"] = [x[1] for x in intervals]
    vol["source"] = "volatility_subset"
    vol_report = vol[
        [
            "source",
            "rule",
            "side",
            "n",
            "direction_accuracy",
            "accuracy_ci_low",
            "accuracy_ci_high",
            "avg_unlevered_direction_return",
            "avg_original_net_return",
        ]
    ].head(20)

    regime = pd.read_csv(REGIME_RESULTS_PATH)
    stable = regime[
        (regime["train_trades"] >= 10)
        & (regime["val_trades"] >= 5)
        & (regime["test_trades"] >= 5)
        & (regime["train_avg_net"] > 0)
        & (regime["val_avg_net"] > 0)
        & (regime["test_avg_net"] > 0)
    ].copy()
    stable = stable.sort_values(["test_avg_net", "val_avg_net"], ascending=False).head(20)
    stable["test_wins"] = (stable["test_accuracy"] * stable["test_trades"]).round().astype(int)
    intervals = stable.apply(lambda r: wilson_interval(int(r["test_wins"]), int(r["test_trades"])), axis=1)
    stable["accuracy_ci_low"] = [x[0] for x in intervals]
    stable["accuracy_ci_high"] = [x[1] for x in intervals]
    stable["source"] = "regime_test"
    stable["rule"] = stable["p_rule"] + "+" + stable["filters"] + "+lev" + stable["leverage"].astype(str)
    regime_report = stable[
        [
            "source",
            "rule",
            "side",
            "test_trades",
            "test_accuracy",
            "accuracy_ci_low",
            "accuracy_ci_high",
            "test_avg_net",
            "test_total_return",
            "test_max_drawdown",
        ]
    ].rename(columns={"test_trades": "n", "test_accuracy": "direction_accuracy"})

    report = pd.concat([vol_report, regime_report], ignore_index=True, sort=False)
    report.to_csv(REPORT_PATH, index=False)

    print(report.to_string(index=False, formatters={
        "direction_accuracy": "{:.2%}".format,
        "accuracy_ci_low": "{:.2%}".format,
        "accuracy_ci_high": "{:.2%}".format,
        "avg_unlevered_direction_return": "{:.4%}".format,
        "avg_original_net_return": "{:.4%}".format,
        "test_avg_net": "{:.4%}".format,
        "test_total_return": "{:.2%}".format,
        "test_max_drawdown": "{:.2%}".format,
    }))
    print(f"saved_report: {REPORT_PATH}")


if __name__ == "__main__":
    main()
