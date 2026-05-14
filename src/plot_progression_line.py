"""
plot_progression_line.py
------------------------
ensemble_progression_v4.csv 의 8단계 PR-AUC 추이를 막대 → 선그래프로 재시각화.

산출물
  figures/ensemble_progression_line.png
"""

from __future__ import annotations

import pandas as pd
import matplotlib.pyplot as plt

from io_utils import (
    FIG_DIR, OUT_DIR,
    setup_korean_font, apply_plot_style, save_fig, PALETTE,
)

setup_korean_font()
apply_plot_style()


def plot_progression_line(df: pd.DataFrame, fname: str = "ensemble_progression_line"):
    fig, ax = plt.subplots(figsize=(16.5, 6.5))
    fig.subplots_adjust(top=0.92, bottom=0.26, left=0.07, right=0.97)

    labels = df["stage"].tolist()
    vals = df["PR-AUC"].tolist()
    xs = list(range(len(vals)))
    base = vals[0]

    # baseline 가로 기준선
    ax.axhline(base, color=PALETTE["rule"], linestyle="--",
               linewidth=0.9, alpha=0.6, zorder=1)

    # 선 + 마커 (코랄 톤, baseline 만 회색)
    line_color = PALETTE["fraud"]
    ax.plot(xs, vals, color=line_color, linewidth=2.2,
            marker="o", markersize=9, markerfacecolor=line_color,
            markeredgecolor="white", markeredgewidth=1.6, zorder=3)
    # baseline 포인트만 톤 다르게 덮어쓰기
    ax.plot([xs[0]], [vals[0]], marker="o", markersize=9,
            markerfacecolor=PALETTE["normal"], markeredgecolor="white",
            markeredgewidth=1.6, zorder=4)

    # 값 레이블 (점 위)
    for x, v in zip(xs, vals):
        ax.text(x, v + 0.0018, f"{v:.4f}",
                ha="center", va="bottom", fontsize=10.5,
                color=PALETTE["text"], weight="bold")

    # baseline 대비 Δpp 레이블 (점 아래)
    for x, v in zip(xs[1:], vals[1:]):
        dlt = (v - base) * 100
        ax.text(x, v - 0.0022, f"Δ {dlt:+.2f}pp",
                ha="center", va="top", fontsize=9.5,
                color=PALETTE["subtext"], weight="bold")

    ax.set_xticks(xs)
    ax.set_xticklabels(labels)
    ax.tick_params(axis="x", rotation=8)
    ax.set_ylabel("PR-AUC")

    lo, hi = min(vals), max(vals)
    pad = max((hi - lo) * 0.35, 0.005)
    ax.set_ylim(lo - pad, hi + pad)
    ax.grid(axis="y", linestyle=":", color=PALETTE["grid"])

    save_fig(fname)
    plt.close(fig)


def main():
    src = OUT_DIR / "ensemble_progression_v4.csv"
    df = pd.read_csv(src)
    plot_progression_line(df)
    print(f"saved → {FIG_DIR / 'ensemble_progression_line.png'}")


if __name__ == "__main__":
    main()
