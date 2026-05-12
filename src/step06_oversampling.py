"""
사기율 8.76% 불균형에 대해 오버샘플링 효과 비교

  baseline          — 기존 (scale_pos_weight ≈ 10)
  SMOTE             — 합성 사기 샘플 생성, 정상과 같은 수로
  BorderlineSMOTE   — 결정 경계 근처 사기 샘플에서 합성, 더 효과적일 수 있음
  SMOTETomek        — SMOTE 후 Tomek-links 정상-사기 페어 제거

모델은 Step 4 best combo (standard-lgbm) 로 고정. sampler 만 교체해 4가지 전략을 5-fold CV 로 비교

산출물:
  outputs/oversample_compare.csv
  figures/oversample_compare.png
"""

from __future__ import annotations

import warnings
import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline as SkPipeline
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE, BorderlineSMOTE
from imblearn.combine import SMOTETomek

from io_utils import (
    load_cust, load_claim, FIG_DIR, OUT_DIR,
    setup_korean_font, apply_plot_style, save_fig, PALETTE,
)
from features import build_features, split_columns
from preprocess import build_preprocessor, split_labeled
from model import build_model, evaluate

warnings.filterwarnings("ignore", message="X does not have valid feature names")

setup_korean_font()
apply_plot_style()

N_SPLITS = 5
SEED = 42
SCALER = "standard"   # Step 4 best


# 4가지 전략 - 각각 (pipeline-class, sampler-or-None, spw-override) 튜플
# baseline 만 sklearn Pipeline, 나머지는 imblearn Pipeline 사용

def make_strategy(name: str, num, cat, spw: float):
    pre = build_preprocessor(num, cat, scaler=SCALER)
    # SMOTE 계열은 합성으로 클래스 균형을 맞추므로 scale_pos_weight 는 1 로 둠
    spw_for_clf = spw if name == "baseline" else 1.0
    clf = build_model("lgbm", scale_pos_weight=spw_for_clf)

    if name == "baseline":
        return SkPipeline([("pre", pre), ("clf", clf)])

    sampler = {
        "SMOTE":          SMOTE(random_state=SEED),
        "BorderlineSMOTE": BorderlineSMOTE(random_state=SEED),
        "SMOTETomek":     SMOTETomek(random_state=SEED),
    }[name]
    return ImbPipeline([("pre", pre), ("sampler", sampler), ("clf", clf)])


# CV 루프
def run_cv(X, y, num, cat, strategy: str) -> pd.DataFrame:
    kf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    rows = []
    for fold, (tr, va) in enumerate(kf.split(X, y), start=1):
        X_tr, X_va = X.iloc[tr], X.iloc[va]
        y_tr, y_va = y.iloc[tr], y.iloc[va]
        spw = (y_tr == 0).sum() / (y_tr == 1).sum()
        pipe = make_strategy(strategy, num, cat, spw)
        pipe.fit(X_tr, y_tr)
        proba = pipe.predict_proba(X_va)[:, 1]
        sc = evaluate(y_va, proba)
        sc.update({"strategy": strategy, "fold": fold})
        rows.append(sc)
    return pd.DataFrame(rows)


# 시각화 - 전략별 PR-AUC / Recall@Top10% 막대 비교
def plot_compare(summary: pd.DataFrame):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    fig.subplots_adjust(top=0.92, bottom=0.18, left=0.07, right=0.97, wspace=0.25)

    for ax, metric, ylabel in [
        (axes[0], "PR-AUC", "PR-AUC"),
        (axes[1], "Recall@Top10%", "Recall@Top10%"),
    ]:
        means = summary[metric]["mean"]
        stds  = summary[metric]["std"]
        baseline_val = means["baseline"]
        # baseline 대비 향상이면 코랄, 아니면 회색
        colors = [PALETTE["fraud"] if v >= baseline_val else PALETTE["normal"]
                  for v in means.values]
        bars = ax.bar(means.index, means.values, yerr=stds.values,
                      color=colors, edgecolor="white", width=0.65,
                      capsize=4, error_kw=dict(elinewidth=1, ecolor=PALETTE["subtext"]))
        for bar, v in zip(bars, means.values):
            ax.text(bar.get_x() + bar.get_width() / 2, v + stds.max() * 0.4,
                    f"{v:.3f}", ha="center", fontsize=11,
                    color=PALETTE["text"], weight="bold")
        ax.axhline(baseline_val, color=PALETTE["rule"],
                   linestyle="--", linewidth=0.9, alpha=0.6)
        ax.set_ylabel(ylabel)
        ax.set_ylim(means.min() * 0.95, means.max() * 1.05)
        ax.grid(axis="y", linestyle=":", color=PALETTE["grid"])
        ax.spines["left"].set_visible(False)
        ax.tick_params(axis="x", rotation=15)

    save_fig("oversample_compare")


# main
def main():
    print("[1/3] 데이터 + 피처")
    X = build_features(load_claim(), load_cust())
    X_lab, y_lab, _ = split_labeled(X)
    num, cat = split_columns(X)
    print(f"      labeled {X_lab.shape}  fraud rate {y_lab.mean():.4f}")

    strategies = ["baseline", "SMOTE", "BorderlineSMOTE", "SMOTETomek"]
    print(f"\n[2/3] 4 전략 × {N_SPLITS} fold CV")
    all_scores = []
    for s in strategies:
        t0 = time.time()
        df = run_cv(X_lab, y_lab, num, cat, s)
        print(f"      {s:<15s}  PR-AUC {df['PR-AUC'].mean():.3f} ± {df['PR-AUC'].std():.3f}   "
              f"R@10% {df['Recall@Top10%'].mean():.3f}   ({time.time()-t0:.1f}s)")
        all_scores.append(df)
    scores = pd.concat(all_scores, ignore_index=True)

    metrics = ["PR-AUC", "ROC-AUC", "F1", "Recall@Top10%", "Recall@Top20%"]
    summary = (scores.groupby("strategy")[metrics]
                     .agg(["mean", "std"])
                     .round(4)
                     .reindex(strategies))
    scores.to_csv(OUT_DIR / "oversample_compare.csv", index=False)
    summary.to_csv(OUT_DIR / "oversample_summary.csv")
    print("\n[3/3] 점수표 저장 → outputs/oversample_compare.csv + oversample_summary.csv\n")
    print("=== 평균 (5-fold) ===")
    print(summary.xs("mean", axis=1, level=1).to_string())

    plot_compare(summary)
    print("\n" + "=" * 70)
    print("step06 오버샘플링 비교 완료.")
    print("=" * 70)


if __name__ == "__main__":
    main()
