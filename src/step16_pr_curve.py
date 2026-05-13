"""
step16_pr_curve.py
------------------
step15 최종 모델 (Optuna LGBM + 새 피처 + 시간 + 미사용변수 + 3-모델 weighted voting) 의
PR curve 시각화. 모델별 단일 곡선 + weighted voting 곡선 + random baseline 비교.

산출물
  figures/pr_curve_final.png         4개 곡선 + best F1 점 마킹
"""

from __future__ import annotations

import json
import warnings
import time

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, average_precision_score, f1_score

from io_utils import (
    load_cust, load_claim, FIG_DIR, OUT_DIR,
    setup_korean_font, apply_plot_style, save_fig, PALETTE,
)
from features import build_features
from preprocess import split_labeled
from step15_unused_vars import get_oof, load_optuna_params, MODEL_NAMES

warnings.filterwarnings("ignore", message="X does not have valid feature names")
warnings.filterwarnings("ignore", category=UserWarning, module="xgboost")

setup_korean_font()
apply_plot_style()


def best_f1_point(y, proba):
    """그 모델 PR 곡선에서 F1 최댓점 — threshold + (precision, recall, F1)"""
    p, r, t = precision_recall_curve(y, proba)
    # threshold len = len(p) - 1
    f1s = 2 * p[:-1] * r[:-1] / (p[:-1] + r[:-1] + 1e-12)
    i = int(np.nanargmax(f1s))
    return float(t[i]), float(p[i]), float(r[i]), float(f1s[i])


def main():
    print("[1/3] 데이터 + 베이스 피처")
    claim = load_claim()
    cust = load_cust()
    X = build_features(claim, cust)
    X_lab, y_lab, _ = split_labeled(X)
    y = y_lab.to_numpy()
    lgbm_params = load_optuna_params()

    print(f"\n[2/3] 3-모델 OOF 재계산 (~3분)")
    probas = {}
    for name in MODEL_NAMES:
        t0 = time.time()
        probas[name] = get_oof(X_lab, y_lab, claim, name, lgbm_params)
        pr = average_precision_score(y, probas[name])
        print(f"      {name:>5s}  PR-AUC {pr:.4f}   ({time.time()-t0:.1f}s)")

    # 가중치 로드 + weighted voting
    with open(OUT_DIR / "ensemble_unused_best_weights.json") as f:
        weights = json.load(f)["best_weights"]
    weighted = sum(weights[n] * probas[n] for n in MODEL_NAMES)
    weighted_pr = average_precision_score(y, weighted)
    print(f"      weighted  PR-AUC {weighted_pr:.4f}   (weights {weights})")

    print(f"\n[3/3] PR curve 시각화")

    # 4개 모델 곡선
    fig, ax = plt.subplots(figsize=(10.5, 7.5))
    fig.subplots_adjust(top=0.94, right=0.96, left=0.10, bottom=0.10)

    curves = [
        ("LR",       probas["lr"],   PALETTE["normal"],   1.6, "--"),
        ("XGBoost",  probas["xgb"],  "#F5C18A",           1.8, "-"),
        ("LightGBM", probas["lgbm"], "#8FB4D8",           1.8, "-"),
        ("Weighted Voting (최종)", weighted, PALETTE["fraud"], 3.0, "-"),
    ]

    for label, proba, color, lw, ls in curves:
        p, r, _ = precision_recall_curve(y, proba)
        ap = average_precision_score(y, proba)
        ax.plot(r, p, color=color, linewidth=lw, linestyle=ls,
                label=f"{label}   PR-AUC {ap:.4f}")

    # 최종 모델 best F1 점 마킹
    best_t, best_p, best_r, best_f1 = best_f1_point(y, weighted)
    ax.scatter([best_r], [best_p], s=220, color=PALETTE["fraud"],
               edgecolor="white", linewidth=2.5, zorder=6)
    ax.annotate(
        f"F1 최대점\n  t = {best_t:.2f}\n  P {best_p:.3f}  R {best_r:.3f}  F1 {best_f1:.3f}",
        xy=(best_r, best_p), xytext=(best_r - 0.30, best_p + 0.10),
        fontsize=10.5, color=PALETTE["text"], weight="bold",
        bbox=dict(boxstyle="round,pad=0.55", facecolor="white",
                  edgecolor=PALETTE["fraud"], linewidth=1.3),
        arrowprops=dict(arrowstyle="->", color=PALETTE["fraud"], lw=1.2),
    )

    # Top10% / Top20% 운영점 표시
    for k_pct in [10, 20]:
        n_top = int(len(y) * k_pct / 100)
        idx = np.argsort(-weighted)[:n_top]
        recall_k = float(y[idx].sum() / y.sum())
        # 그 threshold에서의 precision
        threshold_k = float(np.sort(weighted)[-n_top])
        pred_k = (weighted >= threshold_k).astype(int)
        prec_k = float((pred_k & y).sum() / pred_k.sum())
        ax.scatter([recall_k], [prec_k], s=100, color="#1A2238",
                   edgecolor="white", linewidth=1.8, zorder=5, marker="s")
        ax.annotate(f"Top{k_pct}%\nR {recall_k:.3f}",
                    xy=(recall_k, prec_k), xytext=(recall_k + 0.02, prec_k + 0.04),
                    fontsize=9.5, color=PALETTE["text"], weight="bold")

    # Random baseline (= 사기율)
    base = float(y.mean())
    ax.axhline(base, color=PALETTE["rule"], linestyle=":", linewidth=1, alpha=0.7)
    ax.text(0.02, base + 0.012,
            f"Random P = {base*100:.1f}% (사기율)",
            fontsize=10, color=PALETTE["subtext"])

    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_xlim(0, 1.0); ax.set_ylim(0, 1.0)
    ax.legend(loc="upper right", fontsize=10.5)
    ax.grid(linestyle=":", color=PALETTE["grid"])
    save_fig("pr_curve_final")

    print("\n" + "=" * 70)
    print("step16 — PR curve 시각화 완료.")
    print("=" * 70)


if __name__ == "__main__":
    main()
