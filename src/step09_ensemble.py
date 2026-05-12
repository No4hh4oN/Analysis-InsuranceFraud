"""
step09_ensemble.py
------------------
LR + LightGBM + XGBoost 3개 모델 soft voting 앙상블

흐름:
  1. 3개 모델 5-fold OOF 확률 산출
  2. 단일 모델 PR-AUC / R@K% 측정
  3. 균등(1:1:1) voting 측정
  4. 가중치 grid search (PR-AUC 최대)
  5. 최종 비교 — 단일 vs 균등 vs 가중

산출물
  outputs/ensemble_compare.csv         단일·균등·가중 점수
  outputs/ensemble_best_weights.json   best 가중치 + PR-AUC
  figures/ensemble_compare.png         막대 비교
  figures/ensemble_weight_landscape.png  가중치 PR-AUC 등고선
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
from features import build_features, split_columns
from preprocess import build_preprocessor, split_labeled
from model import evaluate

warnings.filterwarnings("ignore", message="X does not have valid feature names")
warnings.filterwarnings("ignore", category=UserWarning, module="xgboost")

setup_korean_font()
apply_plot_style()

N_SPLITS = 5
SEED = 42
SCALER = "standard"
MODEL_NAMES = ("lr", "lgbm", "xgb")


# 모델 빌더 — step09 안에서만 XGB 포함
def make_model(name: str, spw: float):
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


def get_oof_proba(X, y, num, cat, name: str) -> np.ndarray:
    """5-fold out-of-fold 사기 확률"""
    kf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    oof = np.zeros(len(X))
    for tr, va in kf.split(X, y):
        X_tr, X_va = X.iloc[tr], X.iloc[va]
        y_tr = y.iloc[tr]
        spw = (y_tr == 0).sum() / (y_tr == 1).sum()
        pipe = Pipeline([
            ("pre", build_preprocessor(num, cat, scaler=SCALER)),
            ("clf", make_model(name, spw)),
        ])
        pipe.fit(X_tr, y_tr)
        oof[va] = pipe.predict_proba(X_va)[:, 1]
    return oof


def grid_search_weights(y, probas: dict, step: float = 0.05):
    """3-모델 가중치 grid search (w 합 = 1 제약). PR-AUC 최대화."""
    names = list(probas.keys())
    P = np.stack([probas[n] for n in names])    # (3, n_samples)
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


def plot_compare(df: pd.DataFrame, fname: str = "ensemble_compare"):
    """PR-AUC / R@Top10% 막대 비교"""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    fig.subplots_adjust(top=0.92, bottom=0.18, left=0.07, right=0.97, wspace=0.25)
    for ax, metric in [(axes[0], "PR-AUC"), (axes[1], "Recall@Top10%")]:
        labels = df["model"].tolist()
        vals = df[metric].tolist()
        base = max(vals[:3])    # 단일 모델 중 best
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


def plot_weight_landscape(grid_df, names, best_w):
    """LGBM × XGB 가중치 surface (LR = 1 - LGBM - XGB)"""
    # 단순화: lgbm-xgb 격자에서 PR-AUC max 색칠
    pivot = (grid_df.pivot_table(index="lgbm", columns="xgb",
                                  values="pr_auc", aggfunc="max"))
    fig, ax = plt.subplots(figsize=(8.5, 7))
    fig.subplots_adjust(top=0.95, right=0.95, left=0.10, bottom=0.10)
    im = ax.imshow(pivot.values, origin="lower",
                   extent=[pivot.columns.min(), pivot.columns.max(),
                           pivot.index.min(), pivot.index.max()],
                   cmap="RdPu", aspect="auto")
    fig.colorbar(im, ax=ax, label="PR-AUC", shrink=0.85)
    # best 점 마킹
    ax.scatter([best_w[2]], [best_w[1]], s=240, color="#1A2238",
               edgecolor="white", linewidth=2.5, zorder=5)
    ax.annotate(f"best\nLR {best_w[0]:.2f}  LGBM {best_w[1]:.2f}  XGB {best_w[2]:.2f}",
                xy=(best_w[2], best_w[1]), xytext=(best_w[2] + 0.06, best_w[1] + 0.06),
                fontsize=11, weight="bold", color=PALETTE["text"],
                bbox=dict(boxstyle="round,pad=0.5", facecolor="white",
                          edgecolor=PALETTE["fraud"], linewidth=1.2),
                arrowprops=dict(arrowstyle="->", color=PALETTE["fraud"], lw=1.2))
    ax.set_xlabel("XGBoost 가중치")
    ax.set_ylabel("LightGBM 가중치")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    save_fig("ensemble_weight_landscape")


def main():
    print("[1/5] 데이터 + 피처")
    X = build_features(load_claim(), load_cust())
    X_lab, y_lab, _ = split_labeled(X)
    num, cat = split_columns(X)
    y = y_lab.to_numpy()
    print(f"      labeled {X_lab.shape}")

    print(f"\n[2/5] 3개 모델 OOF 확률 ({N_SPLITS} fold)")
    probas = {}
    for name in MODEL_NAMES:
        t0 = time.time()
        probas[name] = get_oof_proba(X_lab, y_lab, num, cat, name)
        pr = average_precision_score(y, probas[name])
        print(f"      {name:>5s}  PR-AUC {pr:.4f}   ({time.time()-t0:.1f}s)")

    # 단일/균등/가중 점수
    print("\n[3/5] 균등 voting (1:1:1)")
    eq = np.mean([probas[n] for n in MODEL_NAMES], axis=0)
    eq_pr = average_precision_score(y, eq)
    print(f"      PR-AUC {eq_pr:.4f}")

    print("\n[4/5] 가중치 grid search (step 0.05)")
    t0 = time.time()
    best_w, best_pr, grid_df = grid_search_weights(y, probas, step=0.05)
    print(f"      best = LR {best_w[0]:.2f}  LGBM {best_w[1]:.2f}  XGB {best_w[2]:.2f}"
          f"  PR-AUC {best_pr:.4f}   ({time.time()-t0:.1f}s)")

    # 최종 비교 표
    rows = []
    for name in MODEL_NAMES:
        rows.append(metric_row(name, y, probas[name]))
    rows.append(metric_row("equal-voting", y, eq))
    weighted = sum(w * probas[n] for w, n in zip(best_w, MODEL_NAMES))
    rows.append(metric_row("weighted-voting", y, weighted))
    cmp = pd.DataFrame(rows)[["model", "PR-AUC", "ROC-AUC", "F1",
                              "Recall@Top10%", "Recall@Top20%"]]

    print("\n[5/5] 산출물 저장\n")
    print("=== 5-fold OOF 비교 ===")
    print(cmp.round(4).to_string(index=False))

    cmp.to_csv(OUT_DIR / "ensemble_compare.csv", index=False)
    with open(OUT_DIR / "ensemble_best_weights.json", "w") as f:
        json.dump({
            "best_weights": {n: float(w) for n, w in zip(MODEL_NAMES, best_w)},
            "best_pr_auc":  best_pr,
            "single_best_pr_auc": max(
                average_precision_score(y, probas[n]) for n in MODEL_NAMES
            ),
        }, f, indent=2)
    grid_df.to_csv(OUT_DIR / "ensemble_weight_grid.csv", index=False)

    plot_compare(cmp)
    plot_weight_landscape(grid_df, MODEL_NAMES, best_w)

    print("\n" + "=" * 70)
    print("step09 앙상블 비교 완료.")
    print("=" * 70)


if __name__ == "__main__":
    main()
