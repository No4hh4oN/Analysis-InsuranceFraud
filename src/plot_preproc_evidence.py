"""
plot_preproc_evidence.py
------------------------
PPT 캡처용 — "왜 이렇게 결측·이상치 처리했나"의 근거 차트 + 보조 CSV.

산출물
  figures/missing_rate_policy.png    결측률 가로 막대 + 처리 정책 색띠
  figures/skew_scaler_policy.png     skew 가로 막대 + scaler 정책 색띠
  figures/outlier_keep_evidence.png  tail decile 의 사기율 — "이상치 = 사기" 근거
  outputs/distribution_stats.csv     수치 변수별 mean/median/p99/skew/선택 scaler
  outputs/outlier_decile_fraud.csv   변수별 decile 사기율 표
"""

from __future__ import annotations

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from io_utils import (
    OUT_DIR, FIG_DIR, load_cust, load_claim,
    setup_korean_font, apply_plot_style, save_fig, PALETTE,
)
from features import build_features, split_columns
from preprocess import split_labeled

setup_korean_font()
apply_plot_style()


# ──────────────────────────────────────────────────────────────────
# 정책 밴드 정의 — 한 곳에서만 결정
# ──────────────────────────────────────────────────────────────────
MISS_BANDS = [
    (0, 30,  PALETTE["C"],    "0~30% — 정상 imputer (median / most_frequent)"),
    (30, 70, PALETTE["S"],    "30~70% — imputer + 의미 추출 (예: MATE_* = 미혼 시그널)"),
    (70, 101,PALETTE["muted"],"70%+ — 피처 제외 (사실상 정보 없음)"),
]
SKEW_BANDS = [
    (0, 1, PALETTE["C"],    "|skew| < 1 — StandardScaler (정규 분포 가정 OK)"),
    (1, 2, PALETTE["S"],    "1 ≤ |skew| < 2 — 중간 (median imputer 로 안전)"),
    (2, 9e9,PALETTE["fraud"],"|skew| ≥ 2 — RobustScaler (long-tail = 사기 시그널 보존)"),
]


def band_color(v: float, bands, use_abs: bool = False) -> str:
    target = abs(v) if use_abs else v
    for lo, hi, color, _ in bands:
        if lo <= target < hi:
            return color
    return PALETTE["muted"]


def band_legend(bands):
    return [mpatches.Patch(color=c, label=lbl) for _, _, c, lbl in bands]


# ──────────────────────────────────────────────────────────────────
# 1. 결측 정책 차트
# ──────────────────────────────────────────────────────────────────
def plot_missing_policy():
    dd = pd.read_csv(OUT_DIR / "data_dictionary.csv")
    # 0.01% 같은 사실상 0% 행은 정책 차트에서 제외 (≥ 0.5% 만)
    miss = (dd[dd["결측률(%)"] >= 0.5]
            .sort_values("결측률(%)", ascending=True)
            .reset_index(drop=True))
    rates = miss["결측률(%)"].tolist()
    labels = [f"{r['테이블']}·{r['컬럼']}" for _, r in miss.iterrows()]
    colors = [band_color(r, MISS_BANDS) for r in rates]

    fig, ax = plt.subplots(figsize=(13.5, max(6, 0.32 * len(rates) + 1.5)))
    fig.subplots_adjust(left=0.28, right=0.97, top=0.93, bottom=0.08)

    bars = ax.barh(range(len(rates)), rates, color=colors, edgecolor="white")
    for i, v in enumerate(rates):
        ax.text(v + 1.0, i, f"{v:.1f}%",
                va="center", fontsize=9.5,
                color=PALETTE["text"], weight="bold")

    ax.set_yticks(range(len(rates)))
    ax.set_yticklabels(labels, fontsize=9.5)
    ax.set_xlabel("결측률 (%)")
    ax.set_xlim(0, 100)

    for x, txt in [(30, "30%"), (70, "70%")]:
        ax.axvline(x, color=PALETTE["rule"], linestyle="--",
                   linewidth=0.8, alpha=0.65, zorder=1)
        ax.text(x, len(rates) - 0.5, txt,
                color=PALETTE["subtext"], fontsize=9,
                ha="center", va="bottom")

    ax.legend(handles=band_legend(MISS_BANDS),
              loc="lower right", frameon=False, fontsize=10)
    ax.grid(axis="x", linestyle=":", color=PALETTE["grid"])
    save_fig("missing_rate_policy")
    plt.close(fig)


# ──────────────────────────────────────────────────────────────────
# 2. Skew·Scaler 정책 차트 + 보조 CSV
# ──────────────────────────────────────────────────────────────────
def compute_numeric_stats() -> pd.DataFrame:
    claim = load_claim()
    cust = load_cust()
    X = build_features(claim, cust)
    num_cols, _ = split_columns(X)

    rows = []
    for c in num_cols:
        s = pd.to_numeric(X[c], errors="coerce")
        if s.notna().sum() < 2 or s.nunique() < 2:
            continue
        sk = float(s.skew())
        a = abs(sk)
        if a < 1:
            scaler = "StandardScaler"
        elif a >= 2:
            scaler = "RobustScaler"
        else:
            scaler = "—"
        rows.append({
            "feature": c,
            "mean":    float(s.mean()),
            "median":  float(s.median()),
            "p99":     float(s.quantile(0.99)),
            "skew":    sk,
            "p99/median(+1)": float(s.quantile(0.99) / (s.median() + 1)),
            "scaler 선택":  scaler,
        })
    return pd.DataFrame(rows).sort_values("skew", ascending=False).reset_index(drop=True)


def plot_skew_policy(stats: pd.DataFrame):
    # |skew| 기준으로 색·정렬 — 극단 음수 skew(예: YYYYMM)도 long-tail 로 인식.
    sub = stats.assign(abs_skew=stats["skew"].abs()) \
               .sort_values("abs_skew", ascending=True) \
               .reset_index(drop=True)
    vals = sub["skew"].tolist()
    labels = sub["feature"].tolist()
    colors = [band_color(v, SKEW_BANDS, use_abs=True) for v in vals]

    fig, ax = plt.subplots(figsize=(13.5, max(6, 0.32 * len(vals) + 1.5)))
    fig.subplots_adjust(left=0.26, right=0.97, top=0.93, bottom=0.08)

    ax.barh(range(len(vals)), vals, color=colors, edgecolor="white")
    xmax = max(vals) * 1.06
    for i, v in enumerate(vals):
        ax.text(v + xmax * 0.005, i, f"{v:.2f}",
                va="center", fontsize=9.5,
                color=PALETTE["text"], weight="bold")

    ax.set_yticks(range(len(vals)))
    ax.set_yticklabels(labels, fontsize=9.5)
    ax.set_xlabel("Skewness (왜도)")
    ax.set_xlim(min(0, min(vals) * 1.1), xmax)

    for x in (-2, -1, 1, 2):
        ax.axvline(x, color=PALETTE["rule"], linestyle="--",
                   linewidth=0.8, alpha=0.45, zorder=1)
    for x, txt in [(-2, "|skew|=2"), (-1, "|skew|=1"),
                   (1, "|skew|=1"), (2, "|skew|=2")]:
        ax.text(x, len(vals) - 0.5, txt,
                color=PALETTE["subtext"], fontsize=8.5,
                ha="center", va="bottom")

    ax.legend(handles=band_legend(SKEW_BANDS),
              loc="lower right", frameon=False, fontsize=10)
    ax.grid(axis="x", linestyle=":", color=PALETTE["grid"])
    fig.text(0.26, 0.015,
             "※ CUST_RGST 는 YYYYMM 정수 인코딩으로 음수 skew. "
             "RobustScaler 가 이런 비정형 분포까지 안전하게 처리.",
             fontsize=9, color=PALETTE["subtext"])
    save_fig("skew_scaler_policy")
    plt.close(fig)


# ──────────────────────────────────────────────────────────────────
# 3. 이상치 = 사기 시그널 근거 차트 (decile 사기율)
# ──────────────────────────────────────────────────────────────────
# H1·H3·H4·H5 가설을 대표하는 long-tail 변수들 — README 의 가설 프레임과 일치.
KEY_TAIL_VARS = [
    ("paym_sum",          "지급액 합계 (H3)"),
    ("paym_max",          "단일 청구 최대 지급액 (H3)"),
    ("n_claim",           "청구 건수 (H1)"),
    ("n_hospital",        "방문 병원 수 (H1)"),
    ("vlid_otda_sum",     "유효 입원/통원 일수 합 (H4)"),
    ("risky_hosp_visits", "요주의 병원 방문 (H5)"),
]


def decile_fraud_rate(s: pd.Series, y: pd.Series, n: int = 10) -> pd.DataFrame:
    """rank 기반 정확히 n등분 후 decile 별 사기율을 돌려준다."""
    ranks = s.rank(method="first")
    deciles = pd.qcut(ranks, n, labels=False) + 1
    out = pd.DataFrame({"decile": deciles, "y": y.values})
    return (out.groupby("decile")["y"]
              .agg(fraud_rate="mean", n="count", n_fraud="sum")
              .reset_index())


def plot_outlier_keep_evidence(X_lab: pd.DataFrame, y: pd.Series):
    base = float(y.mean())
    fig, axes = plt.subplots(2, 3, figsize=(15, 8.6))
    fig.subplots_adjust(top=0.92, bottom=0.10, left=0.06, right=0.97,
                        hspace=0.50, wspace=0.28)

    decile_rows = []
    for ax, (col, title) in zip(axes.flat, KEY_TAIL_VARS):
        s = pd.to_numeric(X_lab[col], errors="coerce")
        d = decile_fraud_rate(s, y)
        for _, r in d.iterrows():
            decile_rows.append({"variable": col, **r.to_dict()})

        # top decile 만 강조색
        colors = [PALETTE["normal"]] * 9 + [PALETTE["fraud"]]
        ax.bar(d["decile"], d["fraud_rate"] * 100,
               color=colors, edgecolor="white", width=0.72)

        # 전체 사기율 baseline
        ax.axhline(base * 100, color=PALETTE["rule"], linestyle="--",
                   linewidth=0.9, alpha=0.7, zorder=1)
        ax.text(0.6, base * 100 + 0.4, f"전체 평균 {base*100:.2f}%",
                fontsize=8.5, color=PALETTE["subtext"], va="bottom")

        # decile 10 의 사기율·배수 라벨
        top_rate = float(d["fraud_rate"].iloc[-1]) * 100
        ratio = top_rate / (base * 100)
        ax.text(10, top_rate + max(top_rate * 0.04, 0.6),
                f"{top_rate:.1f}%\n× {ratio:.1f}",
                ha="center", va="bottom", fontsize=10.5,
                color=PALETTE["fraud"], weight="bold")

        ax.set_title(title, fontsize=11.5,
                     color=PALETTE["text"], weight="bold", pad=8)
        ax.set_xlabel("Decile  (1=하위 10%  …  10=tail)", fontsize=9.5)
        ax.set_ylabel("사기율 (%)", fontsize=9.5)
        ax.set_xticks(range(1, 11))
        ax.set_ylim(0, max(top_rate * 1.25, base * 100 * 2.2))
        ax.grid(axis="y", linestyle=":", color=PALETTE["grid"])

    fig.text(0.5, 0.018,
             "Top decile (=tail) 사기율이 전체 평균 8.76% 의 1.7~5.3배. "
             "이상치를 자르면 사기 양성 시그널이 함께 잘려나감 → 보존이 정답.",
             ha="center", fontsize=10.5,
             color=PALETTE["text"], weight="bold")
    save_fig("outlier_keep_evidence")
    plt.close(fig)
    return pd.DataFrame(decile_rows)


def main():
    print("[1/4] 결측 정책 차트")
    plot_missing_policy()
    print(f"      saved → {FIG_DIR / 'missing_rate_policy.png'}")

    print("[2/4] 수치 변수 분포 통계 계산")
    stats = compute_numeric_stats()
    stats.to_csv(OUT_DIR / "distribution_stats.csv", index=False)
    print(f"      saved → {OUT_DIR / 'distribution_stats.csv'}  ({len(stats)} features)")

    print("[3/4] Skew·Scaler 정책 차트")
    plot_skew_policy(stats)
    print(f"      saved → {FIG_DIR / 'skew_scaler_policy.png'}")

    print("[4/4] 이상치 = 사기 근거 차트")
    claim = load_claim()
    cust = load_cust()
    X = build_features(claim, cust)
    X_lab, y, _ = split_labeled(X)
    print(f"      labeled set: {X_lab.shape}   사기율 {y.mean()*100:.2f}%")
    decile_df = plot_outlier_keep_evidence(X_lab, y)
    decile_df.to_csv(OUT_DIR / "outlier_decile_fraud.csv", index=False)
    print(f"      saved → {FIG_DIR / 'outlier_keep_evidence.png'}")
    print(f"      saved → {OUT_DIR / 'outlier_decile_fraud.csv'}")


if __name__ == "__main__":
    main()
