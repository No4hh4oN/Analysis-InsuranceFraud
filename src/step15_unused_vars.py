"""
step15_unused_vars.py
---------------------
CLAIM의 미사용 변수에서 강한 시그널 6개 추가:
  CHME_LICE_NO 다양성  (의사 수 — 진짜 doctor shopping)
  CAUS_CODE 다양성     (사고 원인 다양)
  HOSP_SPEC_DVSN 다양성 (병원 종별 섞기)
  NON_PAY_RATIO         (비급여 비율 — 역방향)
  CHANG_FP_YN           (설계사 변경)
  HOSP_OTPA             (입원 기간)

흐름: step14 모델에 add_unused_features 추가.

산출물
  outputs/ensemble_unused_compare.csv
  outputs/ensemble_unused_best_weights.json
  outputs/ensemble_progression_v3.csv     7단계 누적
  figures/ensemble_progression_v3.png     PPT 메인
  figures/unused_feature_importance.png   미사용 피처 importance
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
    add_time_features, add_unused_features, UNUSED_FEATURE_COLS,
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
    with open(OUT_DIR / "optuna_best_params.json") as f:
        return json.load(f)["best_params"]


def make_model(name: str, spw: float, lgbm_params: dict):
    if name == "lr":
        return LogisticRegression(
            class_weight="balanced", max_iter=5000,
            solver="lbfgs", random_state=SEED,
        )
    if name == "lgbm":
        return LGBMClassifier(
            **lgbm_params, scale_pos_weight=spw,
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


def get_oof(X_lab_base, y_lab, claim, name, lgbm_params):
    """step14 + add_unused_features."""
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

        X_tr = add_time_features(X_tr, claim)
        X_va = add_time_features(X_va, claim)

        X_tr = add_unused_features(X_tr, claim)
        X_va = add_unused_features(X_va, claim)

        num, cat = split_columns(X_tr)
        spw = (y_tr == 0).sum() / (y_tr == 1).sum()
        pipe = Pipeline([
            ("pre", build_preprocessor(num, cat, scaler=SCALER)),
            ("clf", make_model(name, spw, lgbm_params)),
        ])
        pipe.fit(X_tr, y_tr)
        oof[va] = pipe.predict_proba(X_va)[:, 1]
    return oof


def grid_search_weights(y, probas, step=0.05):
    names = list(probas.keys())
    P = np.stack([probas[n] for n in names])
    grid = np.round(np.arange(0, 1 + step / 2, step), 4)
    best_pr = -1.0
    best_w = None
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


def metric_row(label, y, proba):
    sc = evaluate(y, proba); sc["model"] = label; return sc


def plot_unused_importance(X_lab, y_lab, claim):
    """미사용 변수 피처 importance — 빨강=미사용변수, 코랄=시간피처, 회색=기존."""
    X = add_interactions(X_lab)
    train_ids = X_lab.index
    train_labels = y_lab.copy(); train_labels.index = train_ids
    chap_rate = compute_chapter_fraud_rate(claim, train_ids, train_labels, alpha=SMOOTHING_ALPHA)
    prior = float(train_labels.mean())
    X = apply_chapter_fraud_score(X, claim, chap_rate, prior)
    X = add_time_features(X, claim)
    X = add_unused_features(X, claim)

    from features import TIME_FEATURE_COLS
    num, cat = split_columns(X)
    spw = (y_lab == 0).sum() / (y_lab == 1).sum()
    pipe = Pipeline([
        ("pre", build_preprocessor(num, cat, scaler=SCALER)),
        ("clf", make_model("lgbm", spw, load_optuna_params())),
    ])
    pipe.fit(X, y_lab)
    pre = pipe.named_steps["pre"]; clf = pipe.named_steps["clf"]
    names = pre.get_feature_names_out()
    imp = pd.Series(
        clf.booster_.feature_importance(importance_type="gain"), index=names,
    ).sort_values(ascending=False)

    unused_set = set(UNUSED_FEATURE_COLS)
    time_set = set(TIME_FEATURE_COLS)
    top = imp.head(25)
    for tc in UNUSED_FEATURE_COLS:
        if tc in imp.index and tc not in top.index:
            top.loc[tc] = imp[tc]
    top = top.sort_values()

    def color(n):
        if n in unused_set: return "#D9534F"      # 빨강
        if n in time_set:   return PALETTE["fraud"]  # 코랄
        return PALETTE["normal"]                  # 회색

    fig, ax = plt.subplots(figsize=(11, 9))
    fig.subplots_adjust(left=0.34, right=0.96, top=0.95, bottom=0.07)
    colors = [color(n) for n in top.index]
    ax.barh(top.index, top.values, color=colors, edgecolor="white", height=0.7)
    ax.set_xlabel("Gain 중요도   (빨강=미사용변수, 코랄=시간피처, 회색=기존)")
    ax.grid(axis="x", linestyle=":", color=PALETTE["grid"])
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    save_fig("unused_feature_importance")


def plot_progression_v3(df, fname="ensemble_progression_v3"):
    fig, ax = plt.subplots(figsize=(15, 6.5))
    fig.subplots_adjust(top=0.92, bottom=0.26, left=0.07, right=0.97)
    labels = df["stage"].tolist()
    vals = df["PR-AUC"].tolist()
    colors = [PALETTE["normal"]] + [PALETTE["fraud"]] * (len(vals) - 1)
    bars = ax.bar(labels, vals, color=colors, edgecolor="white", width=0.62)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.0015,
                f"{v:.4f}", ha="center", fontsize=10.5,
                color=PALETTE["text"], weight="bold")
    base = vals[0]
    for i, v in enumerate(vals[1:], start=1):
        dlt = (v - base) * 100
        ax.text(bars[i].get_x() + bars[i].get_width() / 2,
                v - (max(vals) - min(vals)) * 0.18,
                f"Δ {dlt:+.2f}pp", ha="center", fontsize=9.5,
                color="white", weight="bold")
    ax.axhline(base, color=PALETTE["rule"], linestyle="--", linewidth=0.9, alpha=0.6)
    ax.set_ylabel("PR-AUC")
    lo, hi = min(vals), max(vals)
    pad = max((hi - lo) * 0.35, 0.005)
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
    y = y_lab.to_numpy()
    print(f"      labeled {X_lab.shape}")

    lgbm_params = load_optuna_params()

    print(f"\n[2/5] 미사용 변수 피처 importance (단독)")
    plot_unused_importance(X_lab, y_lab, claim)

    print(f"\n[3/5] 3-모델 OOF (with all features)")
    probas = {}
    for name in MODEL_NAMES:
        t0 = time.time()
        probas[name] = get_oof(X_lab, y_lab, claim, name, lgbm_params)
        pr = average_precision_score(y, probas[name])
        print(f"      {name:>5s}  PR-AUC {pr:.4f}   ({time.time()-t0:.1f}s)")

    print(f"\n[4/5] 가중치 grid search")
    best_w, best_pr = grid_search_weights(y, probas, step=0.05)
    print(f"      best = LR {best_w[0]:.2f}  LGBM {best_w[1]:.2f}  XGB {best_w[2]:.2f}"
          f"  PR-AUC {best_pr:.4f}")

    rows = []
    for name in MODEL_NAMES:
        rows.append(metric_row(name, y, probas[name]))
    eq = np.mean([probas[n] for n in MODEL_NAMES], axis=0)
    rows.append(metric_row("equal-voting", y, eq))
    weighted = sum(w * probas[n] for w, n in zip(best_w, MODEL_NAMES))
    rows.append(metric_row("weighted-voting", y, weighted))
    cmp = pd.DataFrame(rows)[["model", "PR-AUC", "ROC-AUC", "F1",
                              "Recall@Top10%", "Recall@Top20%"]]

    print(f"\n[5/5] 산출물 저장")
    print("\n=== 모든 피처 + Optuna LGBM ===")
    print(cmp.round(4).to_string(index=False))
    cmp.to_csv(OUT_DIR / "ensemble_unused_compare.csv", index=False)

    with open(OUT_DIR / "ensemble_unused_best_weights.json", "w") as f:
        json.dump({
            "best_weights": {n: float(w) for n, w in zip(MODEL_NAMES, best_w)},
            "best_pr_auc":  best_pr,
            "lgbm_params": lgbm_params,
        }, f, indent=2)

    # 7단계 progression
    try:
        s09 = json.load(open(OUT_DIR / "ensemble_best_weights.json"))["best_pr_auc"]
    except FileNotFoundError:
        s09 = float("nan")
    try:
        s10 = pd.read_csv(OUT_DIR / "feature_ablation.csv")
        s10_base = float(s10.loc[s10["variant"] == "baseline", "PR-AUC"].iloc[0])
        s10_both = float(s10.loc[s10["variant"] == "+both", "PR-AUC"].iloc[0])
    except (FileNotFoundError, IndexError):
        s10_base = s10_both = float("nan")
    try:
        s11 = json.load(open(OUT_DIR / "ensemble_advanced_best_weights.json"))["best_pr_auc"]
    except FileNotFoundError:
        s11 = float("nan")
    try:
        s13 = json.load(open(OUT_DIR / "ensemble_optuna_best_weights.json"))["best_pr_auc"]
    except FileNotFoundError:
        s13 = float("nan")
    try:
        s14 = json.load(open(OUT_DIR / "ensemble_time_best_weights.json"))["best_pr_auc"]
    except FileNotFoundError:
        s14 = float("nan")

    progression = pd.DataFrame([
        {"stage": "baseline\n(LGBM 단일)",          "PR-AUC": s10_base},
        {"stage": "+피처개선\n(interaction+TE)",     "PR-AUC": s10_both},
        {"stage": "+앙상블\n(기본피처)",              "PR-AUC": s09},
        {"stage": "+피처+앙상블\n(step11)",           "PR-AUC": s11},
        {"stage": "+Optuna LGBM\n(step13)",         "PR-AUC": s13},
        {"stage": "+시간 피처\n(step14)",             "PR-AUC": s14},
        {"stage": "+미사용 변수\n(최종, step15)",       "PR-AUC": best_pr},
    ])
    progression.to_csv(OUT_DIR / "ensemble_progression_v3.csv", index=False)
    print("\n=== 7단계 누적 PR-AUC ===")
    print(progression.round(4).to_string(index=False))
    print(f"\n      Δ step14 → step15: {(best_pr - s14)*100:+.2f}pp")

    plot_progression_v3(progression)

    print("\n" + "=" * 70)
    print("step15 — 미사용 변수 통합 완료.")
    print("=" * 70)


if __name__ == "__main__":
    main()
