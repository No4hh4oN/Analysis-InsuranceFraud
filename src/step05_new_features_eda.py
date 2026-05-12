"""
step05_new_features_eda.py
--------------------------
LightGBM 중요도에서 새로 발견된 CUST 측 변수 3개를 짧게 EDA.

  AGE         — 연령대별 SIU율
  TOTALPREM   — 총 보험료 분위별 SIU율
  RESI_COST   — 주거 비용 분위별 SIU율

산출물:
  figures/eda_new_01_age.png
  figures/eda_new_02_premium.png
  figures/eda_new_03_resicost.png
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from io_utils import (
    load_cust, FIG_DIR,
    setup_korean_font, apply_plot_style, save_fig,
    PALETTE,
)

setup_korean_font()
apply_plot_style()

cust = load_cust()
train = cust[cust["SIU_CUST_YN"].isin(["Y", "N"])].copy()
train["is_fraud"] = (train["SIU_CUST_YN"] == "Y").astype(int)
base_rate = train["is_fraud"].mean()


# ──────────────────────────────────────────────────────────────────────
# 공통 — 분위별 SIU율 막대
# ──────────────────────────────────────────────────────────────────────

def plot_rate_by_bin(values, labels, fname, xlabel, title_for_log):
    """막대 + 평균선. 핵심 막대(평균 이상)는 코랄, 나머지는 옅은 블루그레이."""
    rate_pct = values * 100
    colors = [PALETTE["fraud"] if v > base_rate else PALETTE["normal"]
              for v in values]

    fig, ax = plt.subplots(figsize=(12, 5.5))
    fig.subplots_adjust(top=0.95, right=0.96, left=0.08, bottom=0.16)
    bars = ax.bar(labels, rate_pct, color=colors, edgecolor="white",
                  width=0.72)

    # 평균선
    ax.axhline(base_rate * 100, color=PALETTE["rule"], linestyle="--",
               linewidth=1, alpha=0.6)
    ax.text(len(labels) - 0.5, base_rate * 100 + 0.4,
            f"전체 평균 {base_rate*100:.1f}%",
            ha="right", fontsize=11, color=PALETTE["text"], weight="bold")

    # 막대 위 % 라벨
    for bar, v in zip(bars, rate_pct):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.3,
                f"{v:.1f}%", ha="center", fontsize=11,
                color=PALETTE["text"], weight="bold")

    ax.set_xlabel(xlabel)
    ax.set_ylabel("SIU율 (%)")
    ax.set_ylim(0, max(rate_pct) * 1.20)
    ax.grid(axis="y", linestyle=":", color=PALETTE["grid"])
    ax.spines["left"].set_visible(False)
    save_fig(fname)
    print(f"  [{title_for_log}] 분위별 SIU율: " +
          ", ".join(f"{lab} {r:.1f}%" for lab, r in zip(labels, rate_pct)))


# ──────────────────────────────────────────────────────────────────────
# Fig 1. AGE — 연령대별 SIU율 (10세 단위)
# ──────────────────────────────────────────────────────────────────────
bins = [0, 20, 30, 40, 50, 60, 70, 200]
labels = ["~19", "20대", "30대", "40대", "50대", "60대", "70+"]
train["AGE_BIN"] = pd.cut(train["AGE"], bins=bins, labels=labels, right=False)
age_rate = train.groupby("AGE_BIN", observed=True)["is_fraud"].mean()
plot_rate_by_bin(age_rate.values, age_rate.index.tolist(),
                 "eda_new_01_age", "연령대", "AGE")


# ──────────────────────────────────────────────────────────────────────
# Fig 2. TOTALPREM — 총 보험료 5분위
# ──────────────────────────────────────────────────────────────────────
prem = train.dropna(subset=["TOTALPREM"]).copy()
prem["PREM_BIN"] = pd.qcut(prem["TOTALPREM"], q=5,
                           labels=["Q1\n(최저)", "Q2", "Q3", "Q4", "Q5\n(최고)"])
prem_rate = prem.groupby("PREM_BIN", observed=True)["is_fraud"].mean()
plot_rate_by_bin(prem_rate.values, prem_rate.index.tolist(),
                 "eda_new_02_premium", "총 보험료 5분위 (TOTALPREM)", "PREMIUM")


# ──────────────────────────────────────────────────────────────────────
# Fig 3. RESI_COST — 주거비용 5분위
# ──────────────────────────────────────────────────────────────────────
resi = train.dropna(subset=["RESI_COST"]).copy()
# RESI_COST가 0이 많을 수 있어 qcut 실패 가능 — duplicates='drop' 안전
resi["RESI_BIN"] = pd.qcut(resi["RESI_COST"], q=5,
                           labels=["Q1\n(최저)", "Q2", "Q3", "Q4", "Q5\n(최고)"],
                           duplicates="drop")
resi_rate = resi.groupby("RESI_BIN", observed=True)["is_fraud"].mean()
plot_rate_by_bin(resi_rate.values, resi_rate.index.tolist(),
                 "eda_new_03_resicost", "주거비용 5분위 (RESI_COST)", "RESI_COST")


print("\n" + "=" * 70)
print("신규 변수 EDA 3장 — figures/eda_new_01~03 생성 완료.")
print("=" * 70)
