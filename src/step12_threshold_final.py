"""
step12_threshold_final.py
-------------------------
step11 의 최종 앙상블(LR+LGBM+XGB weighted voting × 새 피처)에 대한
운영 threshold tuning. step08 의 threshold(0.58)는 LGBM *단일* 기준이라
최종 모델의 확률 분포에는 더 이상 적합하지 않을 수 있음.

흐름:
  1. step11 과 동일한 방식으로 3-모델 5-fold OOF 확률 재계산
     (interaction + target encoding 포함)
  2. ensemble_advanced_best_weights.json 에서 가중치 로드
     → weighted_voting OOF 확률 만들기
  3. threshold 0~1 스캔 → F1 최댓점
  4. step08(LGBM 단일) 와 비교

산출물
  outputs/threshold_final_scan.csv      후보 t 별 P/R/F1
  outputs/threshold_final_best.json     최종 best threshold + 점수 + step08 비교
  figures/threshold_final_pr_curve.png  PR 곡선 + best 점 마킹
  figures/threshold_final_compare.png   step08(단일) vs step12(앙상블) 비교
"""

from __future__ import annotations

import json
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (
    precision_recall_curve, f1_score, precision_score, recall_score,
)

from io_utils import (
    load_cust, load_claim, FIG_DIR, OUT_DIR,
    setup_korean_font, apply_plot_style, save_fig, PALETTE,
)
from features import build_features
from preprocess import split_labeled
from step11_ensemble_advanced import get_oof_with_new_features, MODEL_NAMES

warnings.filterwarnings("ignore", message="X does not have valid feature names")
warnings.filterwarnings("ignore", category=UserWarning, module="xgboost")

setup_korean_font()
apply_plot_style()


def scan_thresholds(y_true, y_proba, n_points: int = 101) -> pd.DataFrame:
    """0~1 균등 스캔 — F1 최댓점 찾기 위함."""
    ts = np.linspace(0.0, 1.0, n_points)
    rows = []
    for t in ts:
        y_hat = (y_proba >= t).astype(int)
        if y_hat.sum() == 0:
            rows.append({"threshold": t, "precision": 1.0,
                         "recall": 0.0, "f1": 0.0})
            continue
        rows.append({
            "threshold": t,
            "precision": precision_score(y_true, y_hat, zero_division=0),
            "recall":    recall_score(y_true, y_hat, zero_division=0),
            "f1":        f1_score(y_true, y_hat, zero_division=0),
        })
    return pd.DataFrame(rows)


def plot_pr_curve(y_true, y_proba, best_t, best_f1):
    p, r, _ = precision_recall_curve(y_true, y_proba)
    y_hat_best = (y_proba >= best_t).astype(int)
    best_p = precision_score(y_true, y_hat_best, zero_division=0)
    best_r = recall_score(y_true, y_hat_best, zero_division=0)

    fig, ax = plt.subplots(figsize=(9, 6))
    fig.subplots_adjust(top=0.96, right=0.96, left=0.10, bottom=0.12)
    ax.plot(r, p, color=PALETTE["fraud"], linewidth=2.4)
    ax.axhline(y_true.mean(), color=PALETTE["rule"], linestyle="--",
               linewidth=0.9, alpha=0.6)
    ax.text(0.02, y_true.mean() + 0.012,
            f"Random {y_true.mean()*100:.1f}%",
            fontsize=10, color=PALETTE["subtext"])

    ax.scatter([best_r], [best_p], s=180, color=PALETTE["fraud"],
               edgecolor="white", linewidth=2.5, zorder=5)
    ax.annotate(
        f"F1 최대점\n  threshold {best_t:.2f}\n  P {best_p:.3f}  R {best_r:.3f}  F1 {best_f1:.3f}",
        xy=(best_r, best_p), xytext=(best_r - 0.20, best_p + 0.07),
        textcoords="data", fontsize=11, color=PALETTE["text"], weight="bold",
        bbox=dict(boxstyle="round,pad=0.6", facecolor="white",
                  edgecolor=PALETTE["fraud"], linewidth=1.3),
        arrowprops=dict(arrowstyle="->", color=PALETTE["fraud"], lw=1.2),
    )
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_xlim(0, 1.0); ax.set_ylim(0, 1.0)
    ax.grid(linestyle=":", color=PALETTE["grid"])
    save_fig("threshold_final_pr_curve")


def plot_step08_vs_step12(step08: dict, step12: dict, fname: str = "threshold_final_compare"):
    """단일 모델 threshold vs 최종 앙상블 threshold — F1/P/R 4-막대 그래프."""
    metrics = ["F1", "Precision", "Recall"]
    s08_vals = [step08["best_f1"], step08["best_precision"], step08["best_recall"]]
    s12_vals = [step12["best_f1"], step12["best_precision"], step12["best_recall"]]

    x = np.arange(len(metrics))
    width = 0.36

    fig, ax = plt.subplots(figsize=(10, 5.8))
    fig.subplots_adjust(top=0.92, right=0.96, left=0.10, bottom=0.14)

    b1 = ax.bar(x - width/2, s08_vals, width,
                label=f"step08 (LGBM 단일) t={step08['best_threshold']:.2f}",
                color=PALETTE["normal"], edgecolor="white")
    b2 = ax.bar(x + width/2, s12_vals, width,
                label=f"step12 (최종 앙상블) t={step12['best_threshold']:.2f}",
                color=PALETTE["fraud"], edgecolor="white")

    for bars, vals in [(b1, s08_vals), (b2, s12_vals)]:
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, v + 0.005,
                    f"{v:.3f}", ha="center", fontsize=10.5,
                    color=PALETTE["text"], weight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.set_ylabel("Score")
    ax.set_ylim(0, max(max(s08_vals), max(s12_vals)) * 1.18)
    ax.legend(loc="upper right")
    ax.grid(axis="y", linestyle=":", color=PALETTE["grid"])
    save_fig(fname)


def main():
    print("[1/5] 데이터 + 베이스 피처")
    claim = load_claim()
    cust = load_cust()
    X = build_features(claim, cust)
    X_lab, y_lab, _ = split_labeled(X)
    y = y_lab.to_numpy()

    print(f"\n[2/5] 3-모델 OOF 재계산 (interaction + target encoding 포함)")
    import time
    probas = {}
    for name in MODEL_NAMES:
        t0 = time.time()
        probas[name] = get_oof_with_new_features(X_lab, y_lab, claim, name)
        print(f"      {name:>5s}  완료 ({time.time()-t0:.1f}s)")

    print(f"\n[3/5] weighted voting 확률 생성")
    with open(OUT_DIR / "ensemble_advanced_best_weights.json") as f:
        meta = json.load(f)
    weights = meta["best_weights"]
    print(f"      weights = {weights}")
    weighted_proba = sum(weights[n] * probas[n] for n in MODEL_NAMES)

    # 비교용 — LGBM 단일도 같이 본다
    lgbm_only = probas["lgbm"]

    print(f"\n[4/5] threshold scan + best 찾기")
    scan_final = scan_thresholds(y, weighted_proba, n_points=101)
    scan_lgbm  = scan_thresholds(y, lgbm_only,      n_points=101)

    best_idx = scan_final["f1"].idxmax()
    best_t   = float(scan_final.loc[best_idx, "threshold"])
    best_f1  = float(scan_final.loc[best_idx, "f1"])
    best_p   = float(scan_final.loc[best_idx, "precision"])
    best_r   = float(scan_final.loc[best_idx, "recall"])

    half = scan_final.iloc[(scan_final["threshold"] - 0.5).abs().idxmin()]
    half_f1 = float(half["f1"]); half_p = float(half["precision"]); half_r = float(half["recall"])

    # step08 결과 로드 (LGBM 단일 기준)
    with open(OUT_DIR / "threshold_best.json") as f:
        step08 = json.load(f)

    print(f"\n=== step08 (LGBM 단일, 기본 피처) ===")
    print(f"  best t = {step08['best_threshold']:.2f}   "
          f"F1 {step08['best_f1']:.4f}   "
          f"P {step08['best_precision']:.4f}   "
          f"R {step08['best_recall']:.4f}")

    print(f"\n=== step12 (최종 앙상블, 새 피처) ===")
    print(f"  t = 0.50         F1 {half_f1:.4f}   "
          f"P {half_p:.4f}   R {half_r:.4f}")
    print(f"  best t = {best_t:.2f}    F1 {best_f1:.4f}   "
          f"P {best_p:.4f}   R {best_r:.4f}  ← F1 최대")

    print(f"\n  Δ best F1 vs step08:  {best_f1 - step08['best_f1']:+.4f}")
    print(f"  Δ best F1 vs t=0.5 :  {best_f1 - half_f1:+.4f}")

    print(f"\n[5/5] 산출물 저장")
    scan_final.to_csv(OUT_DIR / "threshold_final_scan.csv", index=False)
    step12 = {
        "model": "step11 weighted voting (LR/LGBM/XGB + new features)",
        "weights": weights,
        "best_threshold": best_t,
        "best_f1": best_f1,
        "best_precision": best_p,
        "best_recall": best_r,
        "default_t05": {"f1": half_f1, "precision": half_p, "recall": half_r},
        "step08_compare": {
            "best_threshold": step08["best_threshold"],
            "best_f1":        step08["best_f1"],
            "best_precision": step08["best_precision"],
            "best_recall":    step08["best_recall"],
            "delta_f1":       best_f1 - step08["best_f1"],
        },
    }
    with open(OUT_DIR / "threshold_final_best.json", "w") as f:
        json.dump(step12, f, indent=2)
    print(f"  saved → outputs/threshold_final_scan.csv + threshold_final_best.json")

    plot_pr_curve(y, weighted_proba, best_t, best_f1)
    plot_step08_vs_step12(step08, step12)

    print("\n" + "=" * 70)
    print("step12 최종 threshold tuning 완료.")
    print("=" * 70)


if __name__ == "__main__":
    main()
