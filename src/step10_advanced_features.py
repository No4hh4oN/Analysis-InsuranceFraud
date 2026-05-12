"""
step10_advanced_features.py
---------------------------
EDA에서 발견한 "챕터별 사기율 차이"(M 23% vs C 7.8%) 통찰을 모델 신호로
직접 주입하여 PR-AUC를 추가로 끌어올린다.

대안 A — 챕터 사기율 target encoding (Bayesian smoothed, OOF로 leakage 방지)
대안 B — 챕터 × 강한 변수 interaction (M×입원일수, 비-암×지급액 등)

흐름:
  1. 4가지 피처 세트를 5-fold OOF 비교
     baseline           기존 features.build_features 그대로
     +interaction       대안 B 만
     +target_enc        대안 A 만
     +both              두 가지 다 적용
  2. 각 세트 × LGBM 으로 PR-AUC / R@Top10% / R@Top20% 측정
  3. 결과 표 + 막대 그래프 저장

산출물
  outputs/feature_ablation.csv          4-way 비교 표
  figures/feature_ablation.png          PR-AUC / R@TopK 막대 비교
  figures/feature_ablation_delta.png    baseline 대비 개선폭
"""

from __future__ import annotations

import warnings
import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline

from io_utils import (
    load_cust, load_claim, FIG_DIR, OUT_DIR,
    setup_korean_font, apply_plot_style, save_fig, PALETTE,
)
from features import (
    build_features, split_columns,
    add_interactions, compute_chapter_fraud_rate, apply_chapter_fraud_score,
    INTERACTION_COLS, TARGET_ENC_COL,
)
from preprocess import build_preprocessor, split_labeled
from model import build_model, evaluate

warnings.filterwarnings("ignore", message="X does not have valid feature names")

setup_korean_font()
apply_plot_style()

N_SPLITS = 5
SEED = 42
SCALER = "robust"
SMOOTHING_ALPHA = 20.0     # target encoding Bayesian smoothing

# 4-way 비교 세트
VARIANTS = ["baseline", "+interaction", "+target_enc", "+both"]


def make_features(X_base: pd.DataFrame, variant: str) -> pd.DataFrame:
    """train fold에선 *target encoding을 호출자가 따로 적용* 하므로 여기선 interaction만."""
    if variant in ("+interaction", "+both"):
        return add_interactions(X_base)
    return X_base.copy()


def run_oof(X_lab_base: pd.DataFrame, y_lab: pd.Series, claim: pd.DataFrame,
            variant: str) -> dict:
    """5-fold OOF 점수. target encoding은 fold마다 train에서 fit, valid+test에 transform."""
    kf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    oof = np.zeros(len(X_lab_base))

    for fold, (tr, va) in enumerate(kf.split(X_lab_base, y_lab), start=1):
        # ── train fold 만들기 ────────────────────────────────────────────
        X_tr_base = X_lab_base.iloc[tr]
        X_va_base = X_lab_base.iloc[va]
        y_tr = y_lab.iloc[tr]

        # interaction (대안 B) — 양쪽 fold 모두 적용 (leakage 없음, 단순 곱)
        X_tr = make_features(X_tr_base, variant)
        X_va = make_features(X_va_base, variant)

        # target encoding (대안 A) — 반드시 train fold에서만 rate 계산
        if variant in ("+target_enc", "+both"):
            train_ids = X_tr_base.index
            train_labels = y_tr.copy()
            train_labels.index = train_ids
            chap_rate = compute_chapter_fraud_rate(
                claim, train_ids, train_labels, alpha=SMOOTHING_ALPHA,
            )
            prior = float(train_labels.mean())
            X_tr = apply_chapter_fraud_score(X_tr, claim, chap_rate, prior)
            X_va = apply_chapter_fraud_score(X_va, claim, chap_rate, prior)

        # 컬럼 분할 — 새 컬럼은 자동으로 num에 들어감 (CATEGORICAL_COLS 명시 외)
        num, cat = split_columns(X_tr)
        spw = (y_tr == 0).sum() / (y_tr == 1).sum()
        pipe = Pipeline([
            ("pre", build_preprocessor(num, cat, scaler=SCALER)),
            ("clf", build_model("lgbm", scale_pos_weight=spw)),
        ])
        pipe.fit(X_tr, y_tr)
        oof[va] = pipe.predict_proba(X_va)[:, 1]

    sc = evaluate(y_lab.to_numpy(), oof)
    sc["variant"] = variant
    return sc, oof


def plot_ablation(df: pd.DataFrame, fname: str = "feature_ablation"):
    """3-지표 막대 비교 — PR-AUC, R@Top10%, R@Top20%."""
    metrics = ["PR-AUC", "Recall@Top10%", "Recall@Top20%"]
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 5.5))
    fig.subplots_adjust(top=0.92, bottom=0.20, left=0.05, right=0.98, wspace=0.28)

    for ax, m in zip(axes, metrics):
        vals = df[m].tolist()
        labels = df["variant"].tolist()
        baseline_val = vals[0]
        colors = [PALETTE["normal"] if v <= baseline_val + 1e-9 else PALETTE["fraud"]
                  for v in vals]
        bars = ax.bar(labels, vals, color=colors, edgecolor="white", width=0.65)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, v + 0.002,
                    f"{v:.3f}", ha="center", fontsize=11,
                    color=PALETTE["text"], weight="bold")
        ax.axhline(baseline_val, color=PALETTE["rule"], linestyle="--",
                   linewidth=0.9, alpha=0.6, label="baseline")
        ax.set_ylabel(m)
        lo, hi = min(vals), max(vals)
        pad = max((hi - lo) * 0.4, 0.005)
        ax.set_ylim(lo - pad, hi + pad)
        ax.tick_params(axis="x", rotation=15)
        ax.grid(axis="y", linestyle=":", color=PALETTE["grid"])
    save_fig(fname)


def plot_delta(df: pd.DataFrame, fname: str = "feature_ablation_delta"):
    """baseline 대비 PR-AUC 개선폭(pp) 막대 — PPT 헤드라인용."""
    base = df.loc[df["variant"] == "baseline", "PR-AUC"].iloc[0]
    delta = (df["PR-AUC"] - base) * 100  # pp 환산
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    fig.subplots_adjust(top=0.90, bottom=0.18, left=0.10, right=0.97)

    colors = [PALETTE["muted"] if v == 0 else PALETTE["fraud"] for v in delta]
    bars = ax.bar(df["variant"], delta, color=colors, edgecolor="white", width=0.6)
    for bar, v in zip(bars, delta):
        offset = 0.02 if v >= 0 else -0.05
        ax.text(bar.get_x() + bar.get_width() / 2, v + offset,
                f"{v:+.2f}pp", ha="center", fontsize=11,
                color=PALETTE["text"], weight="bold")
    ax.axhline(0, color=PALETTE["rule"], linewidth=0.9)
    ax.set_ylabel("PR-AUC Δ (baseline 대비, pp)")
    ax.tick_params(axis="x", rotation=10)
    ax.grid(axis="y", linestyle=":", color=PALETTE["grid"])
    save_fig(fname)


def main():
    print("[1/4] 데이터 + 베이스 피처")
    t0 = time.time()
    claim = load_claim()
    cust = load_cust()
    X = build_features(claim, cust)
    X_lab, y_lab, _ = split_labeled(X)
    print(f"      labeled {X_lab.shape}   ({time.time() - t0:.1f}s)")

    print(f"\n[2/4] 4-way OOF 비교 (LGBM × {N_SPLITS}-fold, scaler={SCALER})")
    rows = []
    for v in VARIANTS:
        t0 = time.time()
        sc, _ = run_oof(X_lab, y_lab, claim, v)
        rows.append(sc)
        print(f"      {v:<14s}  PR-AUC {sc['PR-AUC']:.4f}   "
              f"R@10% {sc['Recall@Top10%']:.4f}   "
              f"R@20% {sc['Recall@Top20%']:.4f}   "
              f"({time.time() - t0:.1f}s)")

    df = pd.DataFrame(rows)[["variant", "PR-AUC", "ROC-AUC", "F1",
                             "Recall@Top10%", "Recall@Top20%"]]

    print("\n[3/4] 결과 표 저장")
    print(df.round(4).to_string(index=False))
    df.to_csv(OUT_DIR / "feature_ablation.csv", index=False)

    base_pr = df.loc[df["variant"] == "baseline", "PR-AUC"].iloc[0]
    best_row = df.loc[df["PR-AUC"].idxmax()]
    delta_pp = (best_row["PR-AUC"] - base_pr) * 100
    print(f"\n      baseline PR-AUC  : {base_pr:.4f}")
    print(f"      best      variant: {best_row['variant']}  PR-AUC {best_row['PR-AUC']:.4f}  "
          f"(Δ {delta_pp:+.2f}pp)")

    print("\n[4/4] 시각화")
    plot_ablation(df)
    plot_delta(df)

    print("\n" + "=" * 70)
    print("step10 피처 개선 ablation 완료.")
    print("=" * 70)


if __name__ == "__main__":
    main()
