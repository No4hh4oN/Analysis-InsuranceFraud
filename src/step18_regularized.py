"""
step18_regularized.py
---------------------
step17 진단 결과 train fit 0.999 / valid 0.700 / gap 0.30 의 과적합 시그널.
LGBM 만 정규화 강화하여 train-valid gap 줄이는 실험.

정규화 강화 정책:
  Optuna best (step17)        → 보수적 (step18)
  n_estimators       796      → 500
  learning_rate      0.018    → 0.03 (살짝 ↑, n_est 줄어든 만큼 보상)
  num_leaves         90       → 31  (default 수준)
  min_child_samples  63       → 100 (leaf 최소 샘플 ↑)
  subsample          0.77     → 0.7
  colsample_bytree   0.88     → 0.7
  reg_alpha          ~0       → 0.1
  reg_lambda         8.04     → 20.0

목표:
  - train gap 0.30 → 0.10 ~ 0.15
  - valid PR-AUC 0.70 거의 유지 (살짝 낮아져도 OK — 일반화가 더 중요)
"""

from __future__ import annotations

import json
import warnings
import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier

from io_utils import (
    load_cust, load_claim, FIG_DIR, OUT_DIR,
    setup_korean_font, apply_plot_style, save_fig, PALETTE,
)
from features import (
    build_features, split_columns,
    add_interactions, compute_chapter_fraud_rate, apply_chapter_fraud_score,
    add_time_features, add_unused_features,
)
from preprocess import build_preprocessor, split_labeled
from model import evaluate

warnings.filterwarnings("ignore", message="X does not have valid feature names")
warnings.filterwarnings("ignore", category=UserWarning, module="xgboost")

setup_korean_font()
apply_plot_style()

N_SPLITS = 5
SEED = 42
SCALER = "hybrid"
SMOOTHING_ALPHA = 20.0
MODEL_NAMES = ("lr", "lgbm", "xgb")

# 보수적 LGBM params — 과적합 억제
LGBM_REGULARIZED = {
    "n_estimators":     500,
    "learning_rate":    0.03,
    "num_leaves":       31,
    "min_child_samples": 100,
    "subsample":        0.7,
    "colsample_bytree": 0.7,
    "reg_alpha":        0.1,
    "reg_lambda":       20.0,
}


def make_model(name: str, spw: float):
    if name == "lr":
        return LogisticRegression(
            class_weight="balanced", max_iter=5000,
            solver="lbfgs", random_state=SEED,
        )
    if name == "lgbm":
        # 정규화된 LGBM
        return LGBMClassifier(
            **LGBM_REGULARIZED,
            subsample_freq=1,    # subsample 활성화 위해 필요
            scale_pos_weight=spw,
            random_state=SEED, n_jobs=-1, verbosity=-1,
        )
    if name == "xgb":
        # XGB 도 살짝 보수화 (max_depth 6→5, subsample/colsample 추가)
        return XGBClassifier(
            scale_pos_weight=spw,
            n_estimators=500, learning_rate=0.05, max_depth=5,
            subsample=0.8, colsample_bytree=0.8,
            reg_lambda=10.0,
            random_state=SEED, n_jobs=-1, verbosity=0,
            tree_method="hist", eval_metric="aucpr",
        )
    raise ValueError(name)


def get_oof_with_train(X_lab_base, y_lab, claim, name):
    """5-fold OOF + 각 fold의 train 점수도 같이 — gap 측정."""
    kf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    oof = np.zeros(len(X_lab_base))
    train_scores = []
    valid_scores = []

    for fold, (tr, va) in enumerate(kf.split(X_lab_base, y_lab), 1):
        X_tr_base = X_lab_base.iloc[tr]
        X_va_base = X_lab_base.iloc[va]
        y_tr = y_lab.iloc[tr]
        y_va = y_lab.iloc[va]

        X_tr = add_interactions(X_tr_base); X_va = add_interactions(X_va_base)
        train_ids = X_tr_base.index
        tl = y_tr.copy(); tl.index = train_ids
        chap_rate = compute_chapter_fraud_rate(claim, train_ids, tl, alpha=SMOOTHING_ALPHA)
        prior = float(tl.mean())
        X_tr = apply_chapter_fraud_score(X_tr, claim, chap_rate, prior)
        X_va = apply_chapter_fraud_score(X_va, claim, chap_rate, prior)
        X_tr = add_time_features(X_tr, claim); X_va = add_time_features(X_va, claim)
        X_tr = add_unused_features(X_tr, claim); X_va = add_unused_features(X_va, claim)

        num, cat = split_columns(X_tr)
        spw = (y_tr == 0).sum() / (y_tr == 1).sum()
        pipe = Pipeline([
            ("pre", build_preprocessor(num, cat, scaler=SCALER)),
            ("clf", make_model(name, spw)),
        ])
        pipe.fit(X_tr, y_tr)

        p_tr = pipe.predict_proba(X_tr)[:, 1]
        p_va = pipe.predict_proba(X_va)[:, 1]
        train_scores.append(average_precision_score(y_tr, p_tr))
        valid_scores.append(average_precision_score(y_va, p_va))
        oof[va] = p_va

    return oof, train_scores, valid_scores


def grid_search_weights(y, probas, step=0.05):
    names = list(probas.keys())
    P = np.stack([probas[n] for n in names])
    grid = np.round(np.arange(0, 1 + step / 2, step), 4)
    best_pr = -1.0; best_w = None
    for w1 in grid:
        for w2 in grid:
            w3 = 1.0 - w1 - w2
            if w3 < -1e-9 or w3 > 1 + 1e-9:
                continue
            w = np.array([w1, w2, round(w3, 4)])
            pr = average_precision_score(y, (w[:, None] * P).sum(axis=0))
            if pr > best_pr:
                best_pr = pr; best_w = w.copy()
    return best_w, float(best_pr)


def plot_overfit_compare(step17_gap, step18_gap, step17_valid, step18_valid,
                          fname="overfit_compare"):
    """train-valid gap 비교 + valid 점수 변동."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    fig.subplots_adjust(top=0.92, bottom=0.13, left=0.08, right=0.97, wspace=0.25)

    labels = ["step17\n(Optuna LGBM)", "step18\n(정규화 강화)"]

    # Gap 비교
    ax = axes[0]
    bars = ax.bar(labels, [step17_gap, step18_gap],
                  color=[PALETTE["fraud"], PALETTE["normal"]],
                  edgecolor="white", width=0.55)
    for bar, v in zip(bars, [step17_gap, step18_gap]):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.005,
                f"{v:+.3f}", ha="center", fontsize=12, weight="bold")
    ax.set_ylabel("Train - Valid PR-AUC gap")
    ax.set_title("과적합 정도 (낮을수록 좋음)", fontsize=12, color=PALETTE["text"])
    ax.grid(axis="y", linestyle=":", color=PALETTE["grid"])

    # Valid 점수 변동
    ax = axes[1]
    bars = ax.bar(labels, [step17_valid, step18_valid],
                  color=[PALETTE["fraud"], PALETTE["normal"]],
                  edgecolor="white", width=0.55)
    for bar, v in zip(bars, [step17_valid, step18_valid]):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.001,
                f"{v:.4f}", ha="center", fontsize=12, weight="bold")
    ax.set_ylabel("Valid PR-AUC (일반화 성능)")
    ax.set_title("일반화 성능 (높을수록 좋음)", fontsize=12, color=PALETTE["text"])
    ax.set_ylim(min(step17_valid, step18_valid) * 0.99,
                max(step17_valid, step18_valid) * 1.01)
    ax.grid(axis="y", linestyle=":", color=PALETTE["grid"])

    save_fig(fname)


def main():
    print("[1/4] 데이터 + 베이스 피처")
    claim = load_claim()
    cust = load_cust()
    X = build_features(claim, cust)
    X_lab, y_lab, _ = split_labeled(X)
    y = y_lab.to_numpy()
    print(f"      labeled {X_lab.shape}")
    print(f"      LGBM_REGULARIZED: {LGBM_REGULARIZED}")

    print(f"\n[2/4] 3-모델 OOF (정규화 강화)")
    probas = {}; trains = {}; valids = {}
    for name in MODEL_NAMES:
        t0 = time.time()
        oof, tr_sc, va_sc = get_oof_with_train(X_lab, y_lab, claim, name)
        probas[name] = oof
        trains[name] = np.mean(tr_sc); valids[name] = np.mean(va_sc)
        pr = average_precision_score(y, oof)
        print(f"      {name:>5s}  train {np.mean(tr_sc):.4f}  valid {np.mean(va_sc):.4f}"
              f"  gap {np.mean(tr_sc)-np.mean(va_sc):+.4f}   ({time.time()-t0:.1f}s)")

    print(f"\n[3/4] 가중치 grid search + 비교")
    best_w, best_pr = grid_search_weights(y, probas, step=0.05)
    print(f"      best = LR {best_w[0]:.2f}  LGBM {best_w[1]:.2f}  XGB {best_w[2]:.2f}"
          f"  PR-AUC {best_pr:.4f}")

    # step17 vs step18 비교
    s17 = json.load(open(OUT_DIR / "ensemble_hybrid_best_weights.json"))["best_pr_auc"]

    # LGBM gap 비교 (가장 중요 모델)
    lgbm_train = trains["lgbm"]; lgbm_valid = valids["lgbm"]
    step17_lgbm_train = 0.9992  # step17 측정값
    step17_lgbm_valid = 0.6996
    step17_gap = step17_lgbm_train - step17_lgbm_valid
    step18_gap = lgbm_train - lgbm_valid

    print(f"\n=== LGBM Train-Valid gap 비교 ===")
    print(f"  step17 (Optuna)    train {step17_lgbm_train:.4f}  valid {step17_lgbm_valid:.4f}  gap {step17_gap:+.4f}")
    print(f"  step18 (정규화)     train {lgbm_train:.4f}  valid {lgbm_valid:.4f}  gap {step18_gap:+.4f}")
    print(f"  gap 변화          : {step18_gap - step17_gap:+.4f}  ({'감소' if step18_gap < step17_gap else '증가'})")

    print(f"\n=== Final Voting PR-AUC ===")
    print(f"  step17 (Optuna)    : {s17:.4f}")
    print(f"  step18 (정규화)     : {best_pr:.4f}")
    print(f"  변화              : {(best_pr - s17)*100:+.3f}pp")

    print(f"\n[4/4] 산출물 저장")
    out = {
        "lgbm_params": LGBM_REGULARIZED,
        "best_weights": {n: float(w) for n, w in zip(MODEL_NAMES, best_w)},
        "best_pr_auc":  best_pr,
        "lgbm_train":   lgbm_train,
        "lgbm_valid":   lgbm_valid,
        "lgbm_gap":     step18_gap,
        "step17_gap":   step17_gap,
        "gap_reduction": step17_gap - step18_gap,
    }
    with open(OUT_DIR / "step18_regularized.json", "w") as f:
        json.dump(out, f, indent=2)

    plot_overfit_compare(step17_gap, step18_gap, step17_lgbm_valid, lgbm_valid)

    print("\n" + "=" * 70)
    print("step18 — 정규화 강화 완료.")
    print("=" * 70)


if __name__ == "__main__":
    main()
