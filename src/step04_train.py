"""
end-to-end 학습 스크립트

흐름:
  1. 데이터 로드 → features.build_features
  2. Scaler {standard, robust} × Model {lr, lgbm} = 4조합을
     StratifiedKFold(5) 로 CV → outputs/cv_scores.csv
  3. best combo 로 전체 labeled 데이터를 다시 학습
  4. 라벨 없는 DIVIDED_SET=2 의 1,793명에 대해 사기 확률 산출
      → outputs/predictions_holdout.csv
  5. LightGBM gain · LR 계수 Top 20 시각화
      → figures/feature_importance_lgbm.png, figures/feature_importance_lr.png
"""

from __future__ import annotations

import warnings
import time
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline

from io_utils import (
    load_cust, load_claim, FIG_DIR, OUT_DIR,
    setup_korean_font, apply_plot_style, PALETTE,
)
from features import build_features, split_columns
from preprocess import build_preprocessor, split_labeled
from model import build_model, evaluate

# LightGBM이 ColumnTransformer ndarray 입력에서 띄우는 무해한 경고만 끈다.
warnings.filterwarnings(
    "ignore",
    message="X does not have valid feature names",
)

setup_korean_font()
apply_plot_style()

SCALERS = ["standard", "robust"]
MODELS  = ["lr", "lgbm"]
N_SPLITS = 5
SEED = 42


# CV 루프
def run_cv(X, y, num_cols, cat_cols, scaler: str, model_name: str) -> pd.DataFrame:
    """단일 조합에 대해 StratifiedKFold(5) 점수 dataframe 반환."""
    kf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    rows = []
    for fold, (tr, va) in enumerate(kf.split(X, y), start=1):
        X_tr, X_va = X.iloc[tr], X.iloc[va]
        y_tr, y_va = y.iloc[tr], y.iloc[va]
        spw = (y_tr == 0).sum() / (y_tr == 1).sum()
        pipe = Pipeline([
            ("pre", build_preprocessor(num_cols, cat_cols, scaler=scaler)),
            ("clf", build_model(model_name, scale_pos_weight=spw)),
        ])
        pipe.fit(X_tr, y_tr)
        proba = pipe.predict_proba(X_va)[:, 1]
        sc = evaluate(y_va, proba)
        sc.update({"scaler": scaler, "model": model_name, "fold": fold})
        rows.append(sc)
    return pd.DataFrame(rows)


# 시각화
def plot_lgbm_importance(pipe: Pipeline, top_n: int = 20, fname: str = "feature_importance_lgbm"):
    """LightGBM gain 기반 상위 N개 변수."""
    pre = pipe.named_steps["pre"]
    clf = pipe.named_steps["clf"]
    names = pre.get_feature_names_out()
    imp = pd.Series(clf.booster_.feature_importance(importance_type="gain"),
                    index=names).sort_values(ascending=False).head(top_n)[::-1]

    fig, ax = plt.subplots(figsize=(11, 7))
    fig.subplots_adjust(left=0.32, right=0.96, top=0.95, bottom=0.10)
    ax.barh(imp.index, imp.values, color=PALETTE["fraud"], edgecolor="white", height=0.7)
    ax.set_xlabel("Gain 기반 중요도")
    ax.grid(axis="x", linestyle=":", color=PALETTE["grid"])
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    plt.savefig(FIG_DIR / f"{fname}.png")
    plt.close()
    print(f"  saved → figures/{fname}.png")


def plot_lr_coef(pipe: Pipeline, top_n: int = 20, fname: str = "feature_importance_lr"):
    """LR 계수 절댓값 상위 N개 — 부호도 함께(양수=사기 ↑, 음수=사기 ↓)"""
    pre = pipe.named_steps["pre"]
    clf = pipe.named_steps["clf"]
    names = pre.get_feature_names_out()
    coef = pd.Series(clf.coef_[0], index=names)
    top = coef.reindex(coef.abs().sort_values(ascending=False).head(top_n).index)[::-1]

    fig, ax = plt.subplots(figsize=(11, 7))
    fig.subplots_adjust(left=0.32, right=0.96, top=0.95, bottom=0.10)
    colors = [PALETTE["fraud"] if v > 0 else PALETTE["normal"] for v in top.values]
    ax.barh(top.index, top.values, color=colors, edgecolor="white", height=0.7)
    ax.axvline(0, color=PALETTE["rule"], linewidth=0.8)
    ax.set_xlabel("Logistic Regression 계수  (>0: 사기 ↑,  <0: 사기 ↓)")
    ax.grid(axis="x", linestyle=":", color=PALETTE["grid"])
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    plt.savefig(FIG_DIR / f"{fname}.png")
    plt.close()
    print(f"  saved → figures/{fname}.png")


# main
def main():
    # 1. 데이터 + 피처
    print("[1/5] 데이터 로드 + 피처 빌드")
    t0 = time.time()
    X = build_features(load_claim(), load_cust())
    X_lab, y_lab, X_unlab = split_labeled(X)
    num, cat = split_columns(X)
    print(f"      labeled {X_lab.shape}  unlabeled {X_unlab.shape}  ({time.time()-t0:.1f}s)")

    # 2. CV 4조합
    print(f"\n[2/5] CV {len(SCALERS)*len(MODELS)} 조합 × {N_SPLITS} fold")
    all_scores = []
    for scaler in SCALERS:
        for model_name in MODELS:
            t0 = time.time()
            df = run_cv(X_lab, y_lab, num, cat, scaler, model_name)
            mean_pr = df["PR-AUC"].mean()
            mean_r10 = df["Recall@Top10%"].mean()
            print(f"      {scaler:>8s}-{model_name:<4s}  PR-AUC {mean_pr:.3f}   "
                  f"R@10% {mean_r10:.3f}   ({time.time()-t0:.1f}s)")
            all_scores.append(df)
    scores = pd.concat(all_scores, ignore_index=True)

    # 3. 점수표 저장 + 요약
    summary = (scores.groupby(["scaler", "model"])
                     [["PR-AUC", "ROC-AUC", "F1", "Recall@Top10%", "Recall@Top20%"]]
                     .agg(["mean", "std"])
                     .round(4))
    scores.to_csv(OUT_DIR / "cv_scores.csv", index=False)
    summary.to_csv(OUT_DIR / "cv_summary.csv")
    print(f"\n[3/5] CV 점수표 저장 → outputs/cv_scores.csv + cv_summary.csv")
    print("\n=== CV 평균 (5-fold) ===")
    print(summary.xs("mean", axis=1, level=1).to_string())

    # 4. best combo 로 전체 학습 + holdout 예측
    best = (scores.groupby(["scaler", "model"])["PR-AUC"].mean()
                  .sort_values(ascending=False).index[0])
    best_scaler, best_model = best
    print(f"\n[4/5] best combo = {best_scaler}-{best_model} — 전체 데이터 재학습")
    spw = (y_lab == 0).sum() / (y_lab == 1).sum()
    final = Pipeline([
        ("pre", build_preprocessor(num, cat, scaler=best_scaler)),
        ("clf", build_model(best_model, scale_pos_weight=spw)),
    ])
    final.fit(X_lab, y_lab)

    # 라벨 없는 1,793명 사기 확률 산출
    proba_unlab = final.predict_proba(X_unlab)[:, 1]
    pred_df = pd.DataFrame({
        "CUST_ID":      X_unlab.index,
        "fraud_prob":   proba_unlab,
    }).sort_values("fraud_prob", ascending=False)
    pred_df.to_csv(OUT_DIR / "predictions_holdout.csv", index=False)
    print(f"      예측 저장 → outputs/predictions_holdout.csv  ({len(pred_df)}명)")

    # 5. 변수 중요도 시각화 — 두 모델 다 학습해서 비교용
    print("\n[5/5] 변수 중요도 시각화")
    if best_model == "lgbm":
        plot_lgbm_importance(final)
    # LR 도 별도로 학습 (해석용)
    lr_pipe = Pipeline([
        ("pre", build_preprocessor(num, cat, scaler=best_scaler)),
        ("clf", build_model("lr", scale_pos_weight=spw)),
    ])
    lr_pipe.fit(X_lab, y_lab)
    plot_lr_coef(lr_pipe)
    if best_model != "lgbm":
        lgbm_pipe = Pipeline([
            ("pre", build_preprocessor(num, cat, scaler=best_scaler)),
            ("clf", build_model("lgbm", scale_pos_weight=spw)),
        ])
        lgbm_pipe.fit(X_lab, y_lab)
        plot_lgbm_importance(lgbm_pipe)

    print("\n" + "=" * 70)
    print("step04 학습 완료.")
    print("=" * 70)


if __name__ == "__main__":
    main()
