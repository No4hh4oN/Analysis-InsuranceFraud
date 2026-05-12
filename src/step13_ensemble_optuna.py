"""
step13_ensemble_optuna.py
-------------------------
마지막 안 짠 카드. step07 Optuna 가 찾은 LGBM best params 를 step11 앙상블에
통합 — "데이터 천장까지 짜내는 마지막 실험".

흐름:
  step11과 동일하되 LGBM 만 Optuna best params 사용
    LR, XGB 는 step11 그대로
    피처도 동일 (interaction + target encoding, fold OOF)

산출물
  outputs/ensemble_optuna_compare.csv         3-모델 + voting 비교
  outputs/ensemble_optuna_best_weights.json   best 가중치 + 비교
  outputs/ensemble_progression_final.csv      5단계 누적 PR-AUC
  figures/ensemble_progression_final.png      PPT 최종 슬라이드
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
SCALER = "robust"
SMOOTHING_ALPHA = 20.0
MODEL_NAMES = ("lr", "lgbm", "xgb")


def load_optuna_params() -> dict:
    """step07 결과 — LGBM best hyperparameter."""
    with open(OUT_DIR / "optuna_best_params.json") as f:
        d = json.load(f)
    return d["best_params"]


def make_model(name: str, spw: float, lgbm_params: dict):
    if name == "lr":
        return LogisticRegression(
            class_weight="balanced", max_iter=5000,
            solver="lbfgs", random_state=SEED,
        )
    if name == "lgbm":
        # Optuna best params 그대로 적용. scale_pos_weight 만 fold 별 spw 로 override.
        return LGBMClassifier(
            **lgbm_params,
            scale_pos_weight=spw,
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


def get_oof(
    X_lab_base: pd.DataFrame, y_lab: pd.Series,
    claim: pd.DataFrame, name: str, lgbm_params: dict,
) -> np.ndarray:
    """5-fold OOF with interaction + target encoding (fold 안에서 fit)."""
    kf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    oof = np.zeros(len(X_lab_base))

    for tr, va in kf.split(X_lab_base, y_lab):
        X_tr_base = X_lab_base.iloc[tr]
        X_va_base = X_lab_base.iloc[va]
        y_tr = y_lab.iloc[tr]

        X_tr = add_interactions(X_tr_base)
        X_va = add_interactions(X_va_base)

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
            ("clf", make_model(name, spw, lgbm_params)),
        ])
        pipe.fit(X_tr, y_tr)
        oof[va] = pipe.predict_proba(X_va)[:, 1]
    return oof


def grid_search_weights(y, probas: dict, step: float = 0.05):
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


def plot_progression_final(df: pd.DataFrame, fname: str = "ensemble_progression_final"):
    """5단계 누적 PR-AUC — PPT 최종 슬라이드."""
    fig, ax = plt.subplots(figsize=(12.5, 6))
    fig.subplots_adjust(top=0.92, bottom=0.22, left=0.09, right=0.97)
    labels = df["stage"].tolist()
    vals = df["PR-AUC"].tolist()
    colors = [PALETTE["normal"]] + [PALETTE["fraud"]] * (len(vals) - 1)
    bars = ax.bar(labels, vals, color=colors, edgecolor="white", width=0.6)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.0015,
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
    ax.tick_params(axis="x", rotation=8)
    ax.grid(axis="y", linestyle=":", color=PALETTE["grid"])
    save_fig(fname)


def main():
    print("[1/5] 데이터 + 베이스 피처")
    claim = load_claim()
    cust = load_cust()
    X = build_features(claim, cust)
    X_lab, y_lab, _ = split_labeled(X)
    print(f"      labeled {X_lab.shape}")

    lgbm_params = load_optuna_params()
    print(f"\n      Optuna LGBM params:")
    for k, v in lgbm_params.items():
        if isinstance(v, float):
            print(f"        {k:<20s} {v:.4g}")
        else:
            print(f"        {k:<20s} {v}")

    print(f"\n[2/5] 3-모델 OOF (LGBM 만 Optuna params, 나머지는 step11 동일)")
    probas = {}
    for name in MODEL_NAMES:
        t0 = time.time()
        probas[name] = get_oof(X_lab, y_lab, claim, name, lgbm_params)
        pr = average_precision_score(y_lab.to_numpy(), probas[name])
        print(f"      {name:>5s}  PR-AUC {pr:.4f}   ({time.time()-t0:.1f}s)")

    y = y_lab.to_numpy()

    print("\n[3/5] 균등 voting (1:1:1)")
    eq = np.mean([probas[n] for n in MODEL_NAMES], axis=0)
    eq_pr = average_precision_score(y, eq)
    print(f"      PR-AUC {eq_pr:.4f}")

    print("\n[4/5] 가중치 grid search")
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

    print("\n[5/5] 산출물 + 5단계 비교")
    print("\n=== Optuna LGBM × 새 피처 × 3-모델 ===")
    print(cmp.round(4).to_string(index=False))
    cmp.to_csv(OUT_DIR / "ensemble_optuna_compare.csv", index=False)

    with open(OUT_DIR / "ensemble_optuna_best_weights.json", "w") as f:
        json.dump({
            "lgbm_params": lgbm_params,
            "best_weights": {n: float(w) for n, w in zip(MODEL_NAMES, best_w)},
            "best_pr_auc":  best_pr,
            "single_best_pr_auc": max(
                average_precision_score(y, probas[n]) for n in MODEL_NAMES
            ),
        }, f, indent=2)

    # 5단계 progression
    try:
        s09 = json.load(open(OUT_DIR / "ensemble_best_weights.json"))["best_pr_auc"]
    except FileNotFoundError:
        s09 = float("nan")
    try:
        s10 = pd.read_csv(OUT_DIR / "feature_ablation.csv")
        s10_baseline = float(s10.loc[s10["variant"] == "baseline", "PR-AUC"].iloc[0])
        s10_both = float(s10.loc[s10["variant"] == "+both", "PR-AUC"].iloc[0])
    except (FileNotFoundError, IndexError):
        s10_baseline = float("nan"); s10_both = float("nan")
    try:
        s11 = json.load(open(OUT_DIR / "ensemble_advanced_best_weights.json"))["best_pr_auc"]
    except FileNotFoundError:
        s11 = float("nan")

    progression = pd.DataFrame([
        {"stage": "baseline\n(LGBM 단일)",          "PR-AUC": s10_baseline},
        {"stage": "+피처개선\n(interaction+TE)",     "PR-AUC": s10_both},
        {"stage": "+앙상블\n(기본피처)",              "PR-AUC": s09},
        {"stage": "+피처+앙상블\n(step11)",           "PR-AUC": s11},
        {"stage": "+Optuna LGBM\n(최종, step13)",    "PR-AUC": best_pr},
    ])
    progression.to_csv(OUT_DIR / "ensemble_progression_final.csv", index=False)
    print("\n=== 5단계 누적 PR-AUC ===")
    print(progression.round(4).to_string(index=False))
    print(f"\n      Δ step11 → step13: {(best_pr - s11)*100:+.2f}pp")

    plot_progression_final(progression)

    print("\n" + "=" * 70)
    print("step13 — 마지막 카드 완료.")
    print("=" * 70)


if __name__ == "__main__":
    main()
