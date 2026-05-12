"""
step11_ensemble_advanced.py
---------------------------
step10 의 피처 개선(interaction + target encoding)을 step09 앙상블에 결합.
"피처 개선 × 모델 앙상블" 두 축이 함께 작동했을 때의 최종 PR-AUC 측정.

흐름:
  1. fold 마다  base features → +interaction → +target encoding(train fold에서만)
  2. 같은 fold에서 LR / LGBM / XGB 5-fold OOF 확률
  3. 단일·균등(1:1:1)·가중치 grid search 비교
  4. step09 (기본 피처 앙상블) 와 함께 PPT 비교 슬라이드 작성

산출물
  outputs/ensemble_advanced_compare.csv         단일·균등·가중 점수
  outputs/ensemble_advanced_best_weights.json   best 가중치 + PR-AUC
  outputs/ensemble_progression.csv              baseline LGBM → step10 → step09 → step11
  figures/ensemble_advanced_compare.png         3-모델 + voting 막대
  figures/ensemble_progression.png              단계별 누적 개선 (PPT 메인 슬라이드)
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
)
from preprocess import build_preprocessor, split_labeled
from model import evaluate

warnings.filterwarnings("ignore", message="X does not have valid feature names")
warnings.filterwarnings("ignore", category=UserWarning, module="xgboost")

setup_korean_font()
apply_plot_style()

N_SPLITS = 5
SEED = 42
SCALER = "robust"   # step10 과 일관성 맞춤 (LR/트리 둘 다 안정)
SMOOTHING_ALPHA = 20.0
MODEL_NAMES = ("lr", "lgbm", "xgb")


def make_model(name: str, spw: float):
    """step09 와 동일 — 비교 가능성 보장."""
    if name == "lr":
        return LogisticRegression(
            class_weight="balanced", max_iter=5000,
            solver="lbfgs", random_state=SEED,
        )
    if name == "lgbm":
        return LGBMClassifier(
            scale_pos_weight=spw,
            n_estimators=500, learning_rate=0.05, num_leaves=31,
            random_state=SEED, n_jobs=-1, verbosity=-1,
        )
    if name == "xgb":
        return XGBClassifier(
            scale_pos_weight=spw,
            n_estimators=500, learning_rate=0.05, max_depth=6,
            random_state=SEED, n_jobs=-1, verbosity=0,
            tree_method="hist", eval_metric="aucpr",
        )
    raise ValueError(name)


def get_oof_with_new_features(
    X_lab_base: pd.DataFrame, y_lab: pd.Series,
    claim: pd.DataFrame, name: str,
) -> np.ndarray:
    """fold 안에서 interaction + target encoding 둘 다 적용한 OOF 확률.

    interaction 은 train/valid 모두 단순 곱이라 leakage 없음.
    target encoding rate 만 train fold 에서 fit, valid 에 transform.
    """
    kf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    oof = np.zeros(len(X_lab_base))

    for tr, va in kf.split(X_lab_base, y_lab):
        X_tr_base = X_lab_base.iloc[tr]
        X_va_base = X_lab_base.iloc[va]
        y_tr = y_lab.iloc[tr]

        # 대안 B
        X_tr = add_interactions(X_tr_base)
        X_va = add_interactions(X_va_base)

        # 대안 A (train fold 에서만 rate 계산)
        train_ids = X_tr_base.index
        train_labels = y_tr.copy()
        train_labels.index = train_ids
        chap_rate = compute_chapter_fraud_rate(
            claim, train_ids, train_labels, alpha=SMOOTHING_ALPHA,
        )
        prior = float(train_labels.mean())
        X_tr = apply_chapter_fraud_score(X_tr, claim, chap_rate, prior)
        X_va = apply_chapter_fraud_score(X_va, claim, chap_rate, prior)

        num, cat = split_columns(X_tr)
        spw = (y_tr == 0).sum() / (y_tr == 1).sum()
        pipe = Pipeline([
            ("pre", build_preprocessor(num, cat, scaler=SCALER)),
            ("clf", make_model(name, spw)),
        ])
        pipe.fit(X_tr, y_tr)
        oof[va] = pipe.predict_proba(X_va)[:, 1]
    return oof


def grid_search_weights(y, probas: dict, step: float = 0.05):
    """step09 와 동일 — 3-모델 가중치 grid search."""
    names = list(probas.keys())
    P = np.stack([probas[n] for n in names])
    grid = np.round(np.arange(0, 1 + step / 2, step), 4)
    best_pr = -1.0
    best_w = None
    rows = []
    for w1 in grid:
        for w2 in grid:
            w3 = 1.0 - w1 - w2
            if w3 < -1e-9 or w3 > 1 + 1e-9:
                continue
            w3 = round(w3, 4)
            w = np.array([w1, w2, w3])
            combined = (w[:, None] * P).sum(axis=0)
            pr = average_precision_score(y, combined)
            rows.append({**{n: float(v) for n, v in zip(names, w)},
                         "pr_auc": float(pr)})
            if pr > best_pr:
                best_pr = pr
                best_w = w.copy()
    return best_w, float(best_pr), pd.DataFrame(rows)


def metric_row(label, y, proba):
    sc = evaluate(y, proba)
    sc["model"] = label
    return sc


def plot_compare(df: pd.DataFrame, fname: str = "ensemble_advanced_compare"):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    fig.subplots_adjust(top=0.92, bottom=0.18, left=0.07, right=0.97, wspace=0.25)
    for ax, metric in [(axes[0], "PR-AUC"), (axes[1], "Recall@Top10%")]:
        labels = df["model"].tolist()
        vals = df[metric].tolist()
        base = max(vals[:3])
        colors = [PALETTE["fraud"] if v > base else PALETTE["normal"] for v in vals]
        bars = ax.bar(labels, vals, color=colors, edgecolor="white", width=0.65)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, v + 0.003,
                    f"{v:.3f}", ha="center", fontsize=11,
                    color=PALETTE["text"], weight="bold")
        ax.axhline(base, color=PALETTE["rule"], linestyle="--",
                   linewidth=0.9, alpha=0.6)
        ax.set_ylabel(metric)
        ax.set_ylim(min(vals) * 0.96, max(vals) * 1.04)
        ax.grid(axis="y", linestyle=":", color=PALETTE["grid"])
        ax.spines["left"].set_visible(False)
        ax.tick_params(axis="x", rotation=10)
    save_fig(fname)


def plot_progression(df: pd.DataFrame, fname: str = "ensemble_progression"):
    """PPT 메인 — 단계별 누적 PR-AUC 막대."""
    fig, ax = plt.subplots(figsize=(10.5, 5.8))
    fig.subplots_adjust(top=0.92, bottom=0.20, left=0.10, right=0.97)
    labels = df["stage"].tolist()
    vals = df["PR-AUC"].tolist()
    colors = [PALETTE["normal"]] + [PALETTE["fraud"]] * (len(vals) - 1)
    bars = ax.bar(labels, vals, color=colors, edgecolor="white", width=0.62)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.002,
                f"{v:.4f}", ha="center", fontsize=11.5,
                color=PALETTE["text"], weight="bold")
    base = vals[0]
    for i, v in enumerate(vals[1:], start=1):
        dlt = (v - base) * 100
        ax.text(bars[i].get_x() + bars[i].get_width() / 2,
                v - (max(vals) - min(vals)) * 0.18,
                f"Δ {dlt:+.2f}pp", ha="center", fontsize=10,
                color="white", weight="bold")
    ax.axhline(base, color=PALETTE["rule"], linestyle="--", linewidth=0.9, alpha=0.6)
    ax.set_ylabel("PR-AUC")
    lo, hi = min(vals), max(vals)
    pad = max((hi - lo) * 0.4, 0.005)
    ax.set_ylim(lo - pad, hi + pad)
    ax.tick_params(axis="x", rotation=10)
    ax.grid(axis="y", linestyle=":", color=PALETTE["grid"])
    save_fig(fname)


def main():
    print("[1/5] 데이터 + 베이스 피처")
    t0 = time.time()
    claim = load_claim()
    cust = load_cust()
    X = build_features(claim, cust)
    X_lab, y_lab, _ = split_labeled(X)
    print(f"      labeled {X_lab.shape}   ({time.time() - t0:.1f}s)")

    print(f"\n[2/5] 3개 모델 OOF (with interaction + target encoding)")
    probas = {}
    for name in MODEL_NAMES:
        t0 = time.time()
        probas[name] = get_oof_with_new_features(X_lab, y_lab, claim, name)
        pr = average_precision_score(y_lab.to_numpy(), probas[name])
        print(f"      {name:>5s}  PR-AUC {pr:.4f}   ({time.time()-t0:.1f}s)")

    y = y_lab.to_numpy()

    print("\n[3/5] 균등 voting (1:1:1)")
    eq = np.mean([probas[n] for n in MODEL_NAMES], axis=0)
    eq_pr = average_precision_score(y, eq)
    print(f"      PR-AUC {eq_pr:.4f}")

    print("\n[4/5] 가중치 grid search (step 0.05)")
    t0 = time.time()
    best_w, best_pr, grid_df = grid_search_weights(y, probas, step=0.05)
    print(f"      best = LR {best_w[0]:.2f}  LGBM {best_w[1]:.2f}  XGB {best_w[2]:.2f}"
          f"  PR-AUC {best_pr:.4f}   ({time.time()-t0:.1f}s)")

    rows = []
    for name in MODEL_NAMES:
        rows.append(metric_row(name, y, probas[name]))
    rows.append(metric_row("equal-voting", y, eq))
    weighted = sum(w * probas[n] for w, n in zip(best_w, MODEL_NAMES))
    rows.append(metric_row("weighted-voting", y, weighted))
    cmp = pd.DataFrame(rows)[["model", "PR-AUC", "ROC-AUC", "F1",
                              "Recall@Top10%", "Recall@Top20%"]]

    print("\n[5/5] 산출물 + 단계별 비교")
    print("\n=== 새 피처 × 3-모델 OOF 비교 ===")
    print(cmp.round(4).to_string(index=False))
    cmp.to_csv(OUT_DIR / "ensemble_advanced_compare.csv", index=False)

    with open(OUT_DIR / "ensemble_advanced_best_weights.json", "w") as f:
        json.dump({
            "best_weights": {n: float(w) for n, w in zip(MODEL_NAMES, best_w)},
            "best_pr_auc":  best_pr,
            "single_best_pr_auc": max(
                average_precision_score(y, probas[n]) for n in MODEL_NAMES
            ),
        }, f, indent=2)

    # 단계별 누적 — PPT 메인 슬라이드용
    # 1) baseline LGBM       step04 (구버전 학습) 와 step10 baseline 동일
    # 2) +interaction+target  step10 best
    # 3) ensemble (basic)     step09 best  (기본 피처)
    # 4) ensemble + new feat  step11 best  (본 스크립트)
    try:
        s09 = json.load(open(OUT_DIR / "ensemble_best_weights.json"))
        s09_best = s09["best_pr_auc"]
    except FileNotFoundError:
        s09_best = float("nan")
    try:
        s10 = pd.read_csv(OUT_DIR / "feature_ablation.csv")
        s10_baseline = float(s10.loc[s10["variant"] == "baseline", "PR-AUC"].iloc[0])
        s10_both = float(s10.loc[s10["variant"] == "+both", "PR-AUC"].iloc[0])
    except (FileNotFoundError, IndexError):
        s10_baseline = float("nan")
        s10_both = float("nan")

    progression = pd.DataFrame([
        {"stage": "baseline\n(LGBM 단일)",       "PR-AUC": s10_baseline},
        {"stage": "+피처개선\n(interaction+TE)", "PR-AUC": s10_both},
        {"stage": "+앙상블\n(기본피처 3-모델)",  "PR-AUC": s09_best},
        {"stage": "+피처+앙상블\n(최종)",        "PR-AUC": best_pr},
    ])
    progression.to_csv(OUT_DIR / "ensemble_progression.csv", index=False)
    print("\n=== 단계별 누적 PR-AUC ===")
    print(progression.round(4).to_string(index=False))

    plot_compare(cmp)
    plot_progression(progression)

    print("\n" + "=" * 70)
    print("step11 새 피처 × 앙상블 결합 완료.")
    print("=" * 70)


if __name__ == "__main__":
    main()
