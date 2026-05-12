"""
step08_threshold.py
-------------------
운영 시 의심 threshold 결정

지금까지는 threshold = 0.5 기준으로 F1 보고했지만, scale_pos_weight 로 학습된
LightGBM 의 확률 분포는 0.5 가 적정점이 아닐 수 있음. PR 곡선 위에서 F1
최대점을 찾아 fold 평균 best threshold 와 그 점에서의 metric 을 보고

산출물
  outputs/threshold_scan.csv          — 후보 threshold 별 평균 F1·Precision·Recall
  outputs/threshold_best.json         — fold 평균 best threshold + 점수
  figures/threshold_pr_curve.png      — PR 곡선 + best threshold 마킹
  figures/threshold_metric_scan.png   — threshold 스캔 (F1/Precision/Recall)
"""

from __future__ import annotations

import json
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.metrics import precision_recall_curve, f1_score, precision_score, recall_score

from io_utils import (
    load_cust, load_claim, FIG_DIR, OUT_DIR,
    setup_korean_font, apply_plot_style, save_fig, PALETTE,
)
from features import build_features, split_columns
from preprocess import build_preprocessor, split_labeled
from model import build_model

warnings.filterwarnings("ignore", message="X does not have valid feature names")

setup_korean_font()
apply_plot_style()

N_SPLITS = 5
SEED = 42
SCALER = "standard"


def get_oof_proba(X, y, num, cat):
    """5-fold out-of-fold 확률. 각 행은 valid fold 에서 예측된 확률 한 번"""
    kf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    oof = np.zeros(len(X))
    for tr, va in kf.split(X, y):
        X_tr, X_va = X.iloc[tr], X.iloc[va]
        y_tr = y.iloc[tr]
        spw = (y_tr == 0).sum() / (y_tr == 1).sum()
        pipe = Pipeline([
            ("pre", build_preprocessor(num, cat, scaler=SCALER)),
            ("clf", build_model("lgbm", scale_pos_weight=spw)),
        ])
        pipe.fit(X_tr, y_tr)
        oof[va] = pipe.predict_proba(X_va)[:, 1]
    return oof


def scan_thresholds(y_true, y_proba, n_points=101):
    """0~1 균등 threshold 스캔 → F1/Precision/Recall"""
    ts = np.linspace(0.0, 1.0, n_points)
    rows = []
    for t in ts:
        y_hat = (y_proba >= t).astype(int)
        if y_hat.sum() == 0:
            p = 1.0
            r = 0.0
            f1 = 0.0
        else:
            p = precision_score(y_true, y_hat, zero_division=0)
            r = recall_score(y_true, y_hat, zero_division=0)
            f1 = f1_score(y_true, y_hat, zero_division=0)
        rows.append({"threshold": t, "precision": p, "recall": r, "f1": f1})
    return pd.DataFrame(rows)


def plot_pr_curve(y_true, y_proba, best_t, best_f1):
    """PR 곡선 + F1 최대점 마킹"""
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

    # best 점 강조
    ax.scatter([best_r], [best_p], s=180, color=PALETTE["fraud"],
               edgecolor="white", linewidth=2.5, zorder=5)
    ax.annotate(
        f"F1 최대점\n  threshold {best_t:.2f}\n  P {best_p:.3f}  R {best_r:.3f}  F1 {best_f1:.3f}",
        xy=(best_r, best_p), xytext=(best_r - 0.18, best_p + 0.07),
        textcoords="data", fontsize=11, color=PALETTE["text"], weight="bold",
        bbox=dict(boxstyle="round,pad=0.6", facecolor="white",
                  edgecolor=PALETTE["fraud"], linewidth=1.3),
        arrowprops=dict(arrowstyle="->", color=PALETTE["fraud"], lw=1.2),
    )
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_xlim(0, 1.0); ax.set_ylim(0, 1.0)
    ax.grid(linestyle=":", color=PALETTE["grid"])
    save_fig("threshold_pr_curve")


def plot_metric_scan(df, best_t):
    """threshold 스캔 — F1/Precision/Recall 세 곡선"""
    fig, ax = plt.subplots(figsize=(11, 5.5))
    fig.subplots_adjust(top=0.95, right=0.92, left=0.08, bottom=0.13)

    ax.plot(df["threshold"], df["f1"],        color=PALETTE["fraud"],  linewidth=2.6, label="F1")
    ax.plot(df["threshold"], df["precision"], color=PALETTE["text"],   linewidth=2.0, label="Precision")
    ax.plot(df["threshold"], df["recall"],    color=PALETTE["normal"], linewidth=2.0, label="Recall")

    ax.axvline(best_t, color=PALETTE["rule"], linestyle="--",
               linewidth=1, alpha=0.6)
    ax.text(best_t + 0.012, 0.05, f"best t = {best_t:.2f}",
            fontsize=11, color=PALETTE["text"], weight="bold")

    ax.set_xlabel("Threshold")
    ax.set_ylabel("Score")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.legend(loc="lower left")
    ax.grid(linestyle=":", color=PALETTE["grid"])
    save_fig("threshold_metric_scan")


def main():
    print("[1/4] 데이터 + 피처")
    X = build_features(load_claim(), load_cust())
    X_lab, y_lab, _ = split_labeled(X)
    num, cat = split_columns(X)
    y = y_lab.to_numpy()

    print(f"\n[2/4] 5-fold out-of-fold 확률 계산")
    oof = get_oof_proba(X_lab, y_lab, num, cat)

    print(f"\n[3/4] threshold 스캔")
    scan = scan_thresholds(y, oof, n_points=101)
    best_idx = scan["f1"].idxmax()
    best_t   = float(scan.loc[best_idx, "threshold"])
    best_f1  = float(scan.loc[best_idx, "f1"])
    best_p   = float(scan.loc[best_idx, "precision"])
    best_r   = float(scan.loc[best_idx, "recall"])

    # 0.5 기준 비교
    half = scan.iloc[(scan["threshold"] - 0.5).abs().idxmin()]

    print(f"\n  [t=0.50]  F1 {half['f1']:.3f}  P {half['precision']:.3f}  R {half['recall']:.3f}")
    print(f"  [t={best_t:.2f}]  F1 {best_f1:.3f}  P {best_p:.3f}  R {best_r:.3f}  ← F1 최대")

    print(f"\n[4/4] 산출물 저장")
    scan.to_csv(OUT_DIR / "threshold_scan.csv", index=False)
    with open(OUT_DIR / "threshold_best.json", "w") as f:
        json.dump({
            "best_threshold": best_t,
            "best_f1":        best_f1,
            "best_precision": best_p,
            "best_recall":    best_r,
            "default_t05": {
                "f1":        float(half["f1"]),
                "precision": float(half["precision"]),
                "recall":    float(half["recall"]),
            },
        }, f, indent=2)
    print(f"  saved → outputs/threshold_scan.csv + threshold_best.json")

    plot_pr_curve(y, oof, best_t, best_f1)
    plot_metric_scan(scan, best_t)

    print("\n" + "=" * 70)
    print("step08 threshold tuning 완료.")
    print("=" * 70)


if __name__ == "__main__":
    main()
