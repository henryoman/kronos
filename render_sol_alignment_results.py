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

import sol_5m_btc_alignment_search as search


OUT_DIR = Path("outputs/sol_5m_btc_alignment")


def load_rule(path: Path) -> pd.Series:
    return pd.read_csv(path).iloc[0]


def evaluate_rule(rule: pd.Series) -> pd.DataFrame:
    df = search.load_data()
    test = df[df["split"] == "test"].copy()
    filters = tuple() if rule["filters"] == "none" else tuple(str(rule["filters"]).split("+"))
    _, detail = search.evaluate_subset(test, str(rule["signal_rule"]), filters)
    detail = detail.sort_values("signal_timestamp").reset_index(drop=True)
    detail["equity"] = (1.0 + detail["net_return"]).cumprod()
    detail["cum_accuracy"] = detail["direction_correct"].astype(int).cumsum() / np.arange(1, len(detail) + 1)
    return detail


def render_rule_image(rule: pd.Series, detail: pd.DataFrame, output_path: Path, title_suffix: str) -> None:
    fig, axes = plt.subplots(
        2,
        1,
        figsize=(13, 8),
        dpi=160,
        facecolor="#f5f2e8",
        layout="constrained",
    )

    x = pd.to_datetime(detail["signal_timestamp"])
    axes[0].plot(x, detail["equity"], color="#24577a", linewidth=2)
    axes[0].set_title("Equity Curve", fontsize=13)
    axes[0].grid(True, alpha=0.25)
    axes[0].set_ylabel("Equity")

    colors = np.where(detail["direction_correct"], "#177a59", "#b23b3b")
    axes[1].bar(x, detail["direction_correct"].astype(int), color=colors, width=0.08)
    axes[1].plot(x, detail["cum_accuracy"], color="#806a00", linewidth=2, label="Cumulative accuracy")
    axes[1].axhline(0.5, color="gray", linestyle="--", linewidth=1)
    axes[1].set_title("Trade Outcomes", fontsize=13)
    axes[1].set_ylabel("Correct")
    axes[1].set_ylim(0, 1.05)
    axes[1].grid(True, alpha=0.25)
    axes[1].legend(frameon=False)

    fig.suptitle(f"SOL 5m {title_suffix}", fontsize=20, fontweight="bold", color="#1f231d")
    summary = (
        f"{rule['signal_rule']} | {rule['filters']} | "
        f"test trades={int(rule['test_trades'])} | "
        f"test accuracy={rule['test_accuracy']:.2%} | "
        f"avg net={rule['test_avg_original_net_return']:.4%}"
    )
    fig.text(0.06, 0.94, summary, fontsize=10, color="#4e5a46")
    fig.savefig(output_path, facecolor=fig.get_facecolor())
    plt.close(fig)


def render_top_accuracy(results: pd.DataFrame, output_path: Path) -> None:
    ranked = results[results["test_trades"] >= 20].copy()
    ranked = ranked.sort_values(["test_accuracy", "validation_accuracy", "test_trades"], ascending=[False, False, False]).head(12)
    labels = [f"{row.signal_rule}\n{row.filters}" for row in ranked.itertuples(index=False)]
    colors = np.where(ranked["test_avg_original_net_return"] > 0, "#177a59", "#b23b3b")

    fig, ax = plt.subplots(figsize=(14, 8), dpi=160, facecolor="#f5f2e8", layout="constrained")
    y = np.arange(len(ranked))
    ax.barh(y, ranked["test_accuracy"], color=colors, alpha=0.85)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.set_xlabel("Test directional accuracy")
    ax.set_title("Top SOL 5m Rule Accuracy", fontsize=18, color="#1f231d")
    ax.grid(True, axis="x", alpha=0.25)

    for idx, row in enumerate(ranked.itertuples(index=False)):
        ax.text(
            min(row.test_accuracy + 0.01, 0.98),
            idx,
            f"{row.test_accuracy:.1%} | n={int(row.test_trades)} | val={row.validation_accuracy:.1%} | net={row.test_avg_original_net_return:.3%}",
            va="center",
            fontsize=8,
            color="#1f231d",
        )

    fig.savefig(output_path, facecolor=fig.get_facecolor())
    plt.close(fig)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    results = pd.read_csv(OUT_DIR / "all_results.csv")
    best_stable = load_rule(OUT_DIR / "best_stable.csv")
    best_test = load_rule(OUT_DIR / "best_test_n20.csv")
    best_positive = load_rule(OUT_DIR / "best_positive_test_net.csv")

    stable_detail = evaluate_rule(best_stable)
    test_detail = evaluate_rule(best_test)
    positive_detail = evaluate_rule(best_positive)

    stable_detail.to_csv(OUT_DIR / "best_stable_test_detail_rendered.csv", index=False)
    test_detail.to_csv(OUT_DIR / "best_test_n20_detail_rendered.csv", index=False)
    positive_detail.to_csv(OUT_DIR / "best_positive_test_net_detail_rendered.csv", index=False)

    render_rule_image(best_stable, stable_detail, OUT_DIR / "best_stable_equity.png", "Best Stable Rule")
    render_rule_image(best_test, test_detail, OUT_DIR / "best_test_n20_equity.png", "Best Test-Accuracy Rule")
    render_rule_image(best_positive, positive_detail, OUT_DIR / "best_positive_test_net_equity.png", "Best Positive-Net Rule")
    render_top_accuracy(results, OUT_DIR / "top_accuracy_rules.png")

    print(f"saved: {OUT_DIR / 'best_stable_equity.png'}")
    print(f"saved: {OUT_DIR / 'best_test_n20_equity.png'}")
    print(f"saved: {OUT_DIR / 'best_positive_test_net_equity.png'}")
    print(f"saved: {OUT_DIR / 'top_accuracy_rules.png'}")


if __name__ == "__main__":
    main()
