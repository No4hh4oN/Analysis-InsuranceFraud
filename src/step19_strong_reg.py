"""
step19_strong_reg.py
--------------------
step18 의 gap 0.233 도 여전히 크다는 진단 — 정규화 더 강화 + early stopping
도입으로 train fit 을 valid 수준에 더 가깝게.

핵심 변경:
  1. LGBM 더 보수적
      num_leaves           31  →  20
      min_child_samples    100 → 150
      reg_lambda           20  →  30
      max_depth: 명시 X     →  6 (제한 추가)
  2. XGB 더 보수적
      max_depth            5  →  4
      min_child_weight     1  →  5
      reg_lambda           10 → 15
  3. Early stopping — train 내부 80:20 split, valid_loss plateau에서 멈춤

목표:
  gap 0.233 → 0.10 ~ 0.15 (목표)
  valid PR-AUC 0.6998 → 0.69 (살짝 떨어져도 OK)
"""

from __future__ import annotations

import json
import warnings
import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from lightgbm import LGBMClassifier, early_stopping
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

LGBM_STRONG = {
    "n_estimators":     1000,    # 충분히 + early stopping
    "learning_rate":    0.025,
    "num_leaves":       20,      # ↓
    "max_depth":        6,
    "min_child_samples": 150,    # ↑
    "subsample":        0.7,
    "colsample_bytree": 0.7,
    "reg_alpha":        0.5,
    "reg_lambda":       30.0,   # ↑
}

XGB_STRONG = {
    "n_estimators":     1000,
    "learning_rate":    0.04,
    "max_depth":        4,       # 5 → 4
    "min_child_weight": 5,       # 1 → 5
    "subsample":        0.8,
    "colsample_bytree": 0.8,
    "reg_alpha":        0.5,
    "reg_lambda":       15.0,
}

EARLY_STOP_ROUNDS = 50


def make_lr(spw=None):
    return LogisticRegression(
        class_weight="balanced", max_iter=5000,
        solver="lbfgs", random_state=SEED,
    )


def get_oof_lr(X_lab_base, y_lab, claim):
    """LR — early stopping 없음, 정규화 모델."""
    kf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    oof = np.zeros(len(X_lab_base))
    trs = []; vas = []
    for tr, va in kf.split(X_lab_base, y_lab):
        X_tr_base, X_va_base = X_lab_base.iloc[tr], X_lab_base.iloc[va]
        y_tr, y_va = y_lab.iloc[tr], y_lab.iloc[va]
        X_tr = _apply_feats(X_tr_base, y_tr, claim)
        X_va = _apply_feats(X_va_base, y_tr, claim, train_for_te=(X_tr_base.index, y_tr))
        num, cat = split_columns(X_tr)
        pipe = Pipeline([
            ("pre", build_preprocessor(num, cat, scaler=SCALER)),
            ("clf", make_lr()),
        ])
        pipe.fit(X_tr, y_tr)
        p_tr = pipe.predict_proba(X_tr)[:, 1]
        p_va = pipe.predict_proba(X_va)[:, 1]
        trs.append(average_precision_score(y_tr, p_tr))
        vas.append(average_precision_score(y_va, p_va))
        oof[va] = p_va
    return oof, np.mean(trs), np.mean(vas)


def _apply_feats(X_base, y_tr, claim, train_for_te=None):
    """공통 피처 변환 — interaction + target encoding + time + unused."""
    X = add_interactions(X_base)
    if train_for_te is None:
        train_ids, train_labels = X_base.index, y_tr.copy()
    else:
        train_ids, train_labels = train_for_te
        train_labels = train_labels.copy()
    train_labels.index = train_ids
    chap_rate = compute_chapter_fraud_rate(claim, train_ids, train_labels, alpha=SMOOTHING_ALPHA)
    prior = float(train_labels.mean())
    X = apply_chapter_fraud_score(X, claim, chap_rate, prior)
    X = add_time_features(X, claim)
    X = add_unused_features(X, claim)
    return X


def get_oof_lgbm_es(X_lab_base, y_lab, claim):
    """LGBM — train 80% / es-valid 20% 로 early stopping. 5-fold OOF."""
    kf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    oof = np.zeros(len(X_lab_base))
    trs = []; vas = []; iters = []

    for fold, (tr, va) in enumerate(kf.split(X_lab_base, y_lab), 1):
        X_tr_base, X_va_base = X_lab_base.iloc[tr], X_lab_base.iloc[va]
        y_tr, y_va = y_lab.iloc[tr], y_lab.iloc[va]

        # train 80:20 → early stopping용 inner valid
        tr_idx, es_idx = train_test_split(
            np.arange(len(X_tr_base)), test_size=0.2,
            stratify=y_tr, random_state=SEED,
        )
        X_tr_inner = X_tr_base.iloc[tr_idx]
        X_es_inner = X_tr_base.iloc[es_idx]
        y_tr_inner = y_tr.iloc[tr_idx]
        y_es_inner = y_tr.iloc[es_idx]

        # 피처 변환 — train_inner 로만 target encoding fit
        X_tr_inner_f = _apply_feats(X_tr_inner, y_tr_inner, claim)
        X_es_inner_f = _apply_feats(X_es_inner, y_tr_inner, claim,
                                     train_for_te=(X_tr_inner.index, y_tr_inner))
        X_va_f = _apply_feats(X_va_base, y_tr_inner, claim,
                               train_for_te=(X_tr_inner.index, y_tr_inner))

        num, cat = split_columns(X_tr_inner_f)
        pre = build_preprocessor(num, cat, scaler=SCALER)
        pre.fit(X_tr_inner_f)
        Z_tr = pre.transform(X_tr_inner_f)
        Z_es = pre.transform(X_es_inner_f)
        Z_va = pre.transform(X_va_f)

        spw = (y_tr_inner == 0).sum() / (y_tr_inner == 1).sum()
        clf = LGBMClassifier(**LGBM_STRONG, scale_pos_weight=spw,
                              random_state=SEED, n_jobs=-1, verbosity=-1)
        clf.fit(
            Z_tr, y_tr_inner,
            eval_set=[(Z_es, y_es_inner)],
            eval_metric="average_precision",
            callbacks=[early_stopping(EARLY_STOP_ROUNDS, verbose=False)],
        )
        iters.append(clf.best_iteration_)

        # train 점수는 *전체 train* (inner+es) 에 대해 계산하여 gap 측정
        X_tr_full_f = _apply_feats(X_tr_base, y_tr_inner, claim,
                                    train_for_te=(X_tr_inner.index, y_tr_inner))
        Z_tr_full = pre.transform(X_tr_full_f)
        p_tr_full = clf.predict_proba(Z_tr_full)[:, 1]
        p_va = clf.predict_proba(Z_va)[:, 1]
        trs.append(average_precision_score(y_tr, p_tr_full))
        vas.append(average_precision_score(y_va, p_va))
        oof[va] = p_va

    return oof, np.mean(trs), np.mean(vas), np.mean(iters)


def get_oof_xgb_es(X_lab_base, y_lab, claim):
    """XGB — train 80% / es-valid 20% 로 early stopping. 5-fold OOF."""
    kf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    oof = np.zeros(len(X_lab_base))
    trs = []; vas = []; iters = []

    for fold, (tr, va) in enumerate(kf.split(X_lab_base, y_lab), 1):
        X_tr_base, X_va_base = X_lab_base.iloc[tr], X_lab_base.iloc[va]
        y_tr, y_va = y_lab.iloc[tr], y_lab.iloc[va]

        tr_idx, es_idx = train_test_split(
            np.arange(len(X_tr_base)), test_size=0.2,
            stratify=y_tr, random_state=SEED,
        )
        X_tr_inner = X_tr_base.iloc[tr_idx]
        X_es_inner = X_tr_base.iloc[es_idx]
        y_tr_inner = y_tr.iloc[tr_idx]
        y_es_inner = y_tr.iloc[es_idx]

        X_tr_inner_f = _apply_feats(X_tr_inner, y_tr_inner, claim)
        X_es_inner_f = _apply_feats(X_es_inner, y_tr_inner, claim,
                                     train_for_te=(X_tr_inner.index, y_tr_inner))
        X_va_f = _apply_feats(X_va_base, y_tr_inner, claim,
                               train_for_te=(X_tr_inner.index, y_tr_inner))

        num, cat = split_columns(X_tr_inner_f)
        pre = build_preprocessor(num, cat, scaler=SCALER)
        pre.fit(X_tr_inner_f)
        Z_tr = pre.transform(X_tr_inner_f)
        Z_es = pre.transform(X_es_inner_f)
        Z_va = pre.transform(X_va_f)

        spw = (y_tr_inner == 0).sum() / (y_tr_inner == 1).sum()
        clf = XGBClassifier(
            **XGB_STRONG, scale_pos_weight=spw,
            random_state=SEED, n_jobs=-1, verbosity=0,
            tree_method="hist", eval_metric="aucpr",
            early_stopping_rounds=EARLY_STOP_ROUNDS,
        )
        clf.fit(Z_tr, y_tr_inner, eval_set=[(Z_es, y_es_inner)], verbose=False)
        iters.append(clf.best_iteration)

        X_tr_full_f = _apply_feats(X_tr_base, y_tr_inner, claim,
                                    train_for_te=(X_tr_inner.index, y_tr_inner))
        Z_tr_full = pre.transform(X_tr_full_f)
        p_tr_full = clf.predict_proba(Z_tr_full)[:, 1]
        p_va = clf.predict_proba(Z_va)[:, 1]
        trs.append(average_precision_score(y_tr, p_tr_full))
        vas.append(average_precision_score(y_va, p_va))
        oof[va] = p_va

    return oof, np.mean(trs), np.mean(vas), np.mean(iters)


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


def main():
    print("[1/4] 데이터")
    claim = load_claim(); cust = load_cust()
    X = build_features(claim, cust)
    X_lab, y_lab, _ = split_labeled(X)
    y = y_lab.to_numpy()
    print(f"      labeled {X_lab.shape}")
    print(f"\n      LGBM_STRONG: {LGBM_STRONG}")
    print(f"      XGB_STRONG:  {XGB_STRONG}")
    print(f"      EARLY_STOP_ROUNDS = {EARLY_STOP_ROUNDS}")

    print(f"\n[2/4] 3-모델 OOF (강한 정규화 + early stopping)")
    t0 = time.time()
    oof_lr, lr_tr, lr_va = get_oof_lr(X_lab, y_lab, claim)
    print(f"      lr     train {lr_tr:.4f}  valid {lr_va:.4f}  gap {lr_tr-lr_va:+.4f}   ({time.time()-t0:.1f}s)")

    t0 = time.time()
    oof_lgbm, lgbm_tr, lgbm_va, lgbm_iter = get_oof_lgbm_es(X_lab, y_lab, claim)
    print(f"      lgbm   train {lgbm_tr:.4f}  valid {lgbm_va:.4f}  gap {lgbm_tr-lgbm_va:+.4f}   "
          f"avg_iter {lgbm_iter:.0f}   ({time.time()-t0:.1f}s)")

    t0 = time.time()
    oof_xgb, xgb_tr, xgb_va, xgb_iter = get_oof_xgb_es(X_lab, y_lab, claim)
    print(f"      xgb    train {xgb_tr:.4f}  valid {xgb_va:.4f}  gap {xgb_tr-xgb_va:+.4f}   "
          f"avg_iter {xgb_iter:.0f}   ({time.time()-t0:.1f}s)")

    probas = {"lr": oof_lr, "lgbm": oof_lgbm, "xgb": oof_xgb}

    print(f"\n[3/4] 가중치 grid search")
    best_w, best_pr = grid_search_weights(y, probas, step=0.05)
    print(f"      best = LR {best_w[0]:.2f}  LGBM {best_w[1]:.2f}  XGB {best_w[2]:.2f}"
          f"  PR-AUC {best_pr:.4f}")

    # 비교
    s17 = json.load(open(OUT_DIR / "ensemble_hybrid_best_weights.json"))["best_pr_auc"]
    s18 = json.load(open(OUT_DIR / "step18_regularized.json"))
    print(f"\n=== 단계별 비교 ===")
    print(f"  step17 (Optuna)        valid 0.6996  gap 0.300  final 0.7000")
    print(f"  step18 (정규화 1차)     valid 0.6998  gap 0.233  final {s18['best_pr_auc']:.4f}")
    print(f"  step19 (강한 정규화)    valid {lgbm_va:.4f}  gap {lgbm_tr-lgbm_va:+.4f}  final {best_pr:.4f}")

    out = {
        "lgbm_params": LGBM_STRONG,
        "xgb_params":  XGB_STRONG,
        "early_stopping_rounds": EARLY_STOP_ROUNDS,
        "lr":   {"train": lr_tr, "valid": lr_va, "gap": lr_tr-lr_va},
        "lgbm": {"train": lgbm_tr, "valid": lgbm_va, "gap": lgbm_tr-lgbm_va, "avg_iter": float(lgbm_iter)},
        "xgb":  {"train": xgb_tr, "valid": xgb_va, "gap": xgb_tr-xgb_va, "avg_iter": float(xgb_iter)},
        "best_weights": {n: float(w) for n, w in zip(MODEL_NAMES, best_w)},
        "best_pr_auc":  best_pr,
    }
    with open(OUT_DIR / "step19_strong_reg.json", "w") as f:
        json.dump(out, f, indent=2, default=float)

    # gap 추이 plot
    stages = ["step17\n(Optuna)", "step18\n(정규화 1차)", "step19\n(정규화 2차+ES)"]
    gaps = [0.300, 0.233, lgbm_tr-lgbm_va]
    valids = [0.6996, 0.6998, lgbm_va]
    finals = [0.7000, s18["best_pr_auc"], best_pr]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5.5))
    fig.subplots_adjust(top=0.92, bottom=0.16, left=0.06, right=0.97, wspace=0.27)

    for ax, vals, title, ylab in [
        (axes[0], gaps,   "LGBM Train-Valid gap (낮을수록 좋음)", "Gap"),
        (axes[1], valids, "LGBM Valid PR-AUC",                 "Valid PR-AUC"),
        (axes[2], finals, "Final Voting PR-AUC",                "PR-AUC"),
    ]:
        colors = [PALETTE["normal"], PALETTE["fraud"], "#A85D5D"]
        bars = ax.bar(stages, vals, color=colors, edgecolor="white", width=0.6)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, v + (max(vals)-min(vals))*0.02,
                    f"{v:.4f}" if ylab != "Gap" else f"{v:+.3f}",
                    ha="center", fontsize=11, weight="bold")
        ax.set_ylabel(ylab); ax.set_title(title, fontsize=11.5, color=PALETTE["text"])
        ax.grid(axis="y", linestyle=":", color=PALETTE["grid"])
        if ylab != "Gap":
            ax.set_ylim(min(vals)*0.99, max(vals)*1.01)
    save_fig("overfit_progression")

    print("\n" + "=" * 70)
    print("step19 — 강한 정규화 + early stopping 완료.")
    print("=" * 70)


if __name__ == "__main__":
    main()
