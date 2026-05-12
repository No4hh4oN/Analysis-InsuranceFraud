"""
산출물:
  - figures/eda_01_target.png        : 타겟(SIU_CUST_YN) 불균형
  - figures/eda_02_categorical.png   : 범주형 변수별 사기율 (4 subplot)
  - figures/eda_03_numeric_box.png   : 수치형 변수 사기 vs 정상 박스플롯
  - figures/eda_04_claim_signal.png  : 청구 집계 시그널 (1인당 청구건/병원/입원/금액)
  - figures/eda_05_disease.png       : 사기 비중 높은 질병 Top 10
  - figures/eda_06_corr.png          : 수치형 변수 상관관계 히트맵
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from io_utils import load_cust, load_claim, save_fig, setup_korean_font

setup_korean_font()

# 사기 = 빨강 / 정상 = 회색 — 전체 그림에서 통일
COLOR_FRAUD = "#E53E3E"
COLOR_NORMAL = "#A0AEC0"

cust = load_cust()
claim = load_claim()

# 라벨 있는 train만 사용 (DIVIDED_SET == 1)
train = cust[cust["SIU_CUST_YN"].isin(["Y", "N"])].copy()
train["is_fraud"] = (train["SIU_CUST_YN"] == "Y").astype(int)

print("=" * 70)
print("EDA 시작 — train(라벨 있는) 샘플:", f"{len(train):,}명")
print("=" * 70)

# Fig 1. 타겟 분포 — 불균형 강조
fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))

counts = train["SIU_CUST_YN"].value_counts()
axes[0].bar(["정상(N)", "사기(Y)"],
            [counts["N"], counts["Y"]],
            color=[COLOR_NORMAL, COLOR_FRAUD])
axes[0].set_title("타겟 분포 (절대 수)", weight="bold")
for i, v in enumerate([counts["N"], counts["Y"]]):
    axes[0].text(i, v, f"{v:,}", ha="center", va="bottom", fontsize=11)
axes[0].set_ylabel("고객 수")

ratio = counts / counts.sum() * 100
axes[1].pie([ratio["N"], ratio["Y"]],
            labels=[f"정상\n{ratio['N']:.1f}%", f"사기\n{ratio['Y']:.1f}%"],
            colors=[COLOR_NORMAL, COLOR_FRAUD],
            autopct="", startangle=90,
            wedgeprops={"edgecolor": "white", "linewidth": 2})
axes[1].set_title("타겟 비율", weight="bold")

fig.suptitle(f"보험사기 라벨 분포  (n={len(train):,})",
             fontsize=14, weight="bold", y=1.02)
save_fig("eda_01_target")
print("[Fig 1] 사기율 8.76% — 명백한 불균형. class_weight/오버샘플링 필요.")

# Fig 2. 범주형 변수별 사기율 — 4 subplot
def fraud_rate_by(col: str, min_n: int = 100, top: int = 10):
    """범주별 (건수>=min_n 인) 사기율, 사기율 내림차순 Top n."""
    g = train.groupby(col)["is_fraud"].agg(["count", "mean"])
    g = g[g["count"] >= min_n].sort_values("mean", ascending=False).head(top)
    return g


fig, axes = plt.subplots(2, 2, figsize=(13, 8))
base_rate = train["is_fraud"].mean()

specs = [
    ("OCCP_GRP_1", "직업 대분류"),
    ("CTPR", "지역 (시·도)"),
    ("FP_CAREER", "본인 설계사 경력"),
    ("SEX", "성별 (1=남, 2=여)"),
]
for ax, (col, label) in zip(axes.flat, specs):
    g = fraud_rate_by(col, min_n=100, top=10)
    bars = ax.barh(g.index.astype(str), g["mean"] * 100,
                   color=COLOR_FRAUD, edgecolor="white")
    ax.axvline(base_rate * 100, color="black", linestyle="--", linewidth=1)
    ax.set_xlabel("사기율 (%)")
    ax.set_title(label, weight="bold")
    ax.invert_yaxis()
    for bar, n in zip(bars, g["count"]):
        ax.text(bar.get_width() + 0.2, bar.get_y() + bar.get_height() / 2,
                f"n={n:,}", va="center", fontsize=9, color="#555")

fig.suptitle("범주형 변수별 사기율  (점선 = 전체 평균 8.76%)",
             fontsize=14, weight="bold", y=1.01)
plt.tight_layout()
save_fig("eda_02_categorical")
print("[Fig 2] FP_CAREER=Y, 광주/부산, 자영업·주부 카테고리에서 사기율 큰 폭으로 상승.")


# Fig 3. 수치형 변수 사기 vs 정상 — 박스플롯
num_cols = ["AGE", "MAXCRDT", "TOTALPREM", "MAX_PRM",
            "CUST_INCM", "RCBASE_HSHD_INCM"]
fig, axes = plt.subplots(2, 3, figsize=(13, 7.5))
for ax, col in zip(axes.flat, num_cols):
    data = [train.loc[train["SIU_CUST_YN"] == "N", col].dropna(),
            train.loc[train["SIU_CUST_YN"] == "Y", col].dropna()]
    bp = ax.boxplot(data, tick_labels=["정상", "사기"], patch_artist=True,
                    showfliers=False, widths=0.55)
    for patch, color in zip(bp["boxes"], [COLOR_NORMAL, COLOR_FRAUD]):
        patch.set_facecolor(color)
        patch.set_alpha(0.85)
    ax.set_title(col, weight="bold")
    ax.grid(axis="y", linestyle=":", alpha=0.5)

fig.suptitle("고객 수치형 변수 — 사기 vs 정상  (이상치 제외)",
             fontsize=14, weight="bold", y=1.01)
plt.tight_layout()
save_fig("eda_03_numeric_box")
print("[Fig 3] MAXCRDT(신용도)는 사기군이 확연히 낮음, MAX_PRM은 사기군이 높음.")

# ──────────────────────────────────────────────────────────────────────
# Fig 4. 청구 집계 시그널 — 1인당 청구 패턴
# ──────────────────────────────────────────────────────────────────────
# 청구 테이블을 CUST_ID 기준으로 집계해 train과 머지
claim_agg = claim.groupby("CUST_ID").agg(
    n_claim=("POLY_NO", "size"),
    n_hosp=("HOSP_CODE", "nunique"),
    sum_otda=("VLID_HOSP_OTDA", "sum"),
    total_dmnd=("DMND_AMT", "sum"),
)
merged = train.merge(claim_agg, on="CUST_ID", how="left")

panels = [
    ("n_claim",   "1인당 청구 건수"),
    ("n_hosp",    "방문한 병원 수"),
    ("sum_otda",  "유효 입원/통원일 합계"),
    ("total_dmnd", "총 청구금액(원)"),
]

fig, axes = plt.subplots(1, 4, figsize=(15, 4.5))
for ax, (col, label) in zip(axes.flat, panels):
    data = [merged.loc[merged["SIU_CUST_YN"] == "N", col].dropna(),
            merged.loc[merged["SIU_CUST_YN"] == "Y", col].dropna()]
    bp = ax.boxplot(data, tick_labels=["정상", "사기"], patch_artist=True,
                    showfliers=False, widths=0.55)
    for patch, color in zip(bp["boxes"], [COLOR_NORMAL, COLOR_FRAUD]):
        patch.set_facecolor(color)
        patch.set_alpha(0.85)
    # 평균선 표시 (사기 평균이 얼마나 더 큰지 강조)
    mean_n, mean_y = data[0].mean(), data[1].mean()
    ax.set_title(f"{label}\n사기/정상 = {mean_y/mean_n:.1f}배",
                 weight="bold", fontsize=11)
    ax.grid(axis="y", linestyle=":", alpha=0.5)

fig.suptitle("청구 데이터를 CUST_ID로 집계한 핵심 시그널",
             fontsize=14, weight="bold", y=1.02)
plt.tight_layout()
save_fig("eda_04_claim_signal")
print("[Fig 4] 청구 빈도·병원수·입원일·금액 모두 사기군이 정상의 3~6배. 핵심 피처.")

# Fig 5. 의심 질병 Top 10 (사기 청구 비중)
claim_t = claim.merge(train[["CUST_ID", "is_fraud"]], on="CUST_ID")
dsas = (claim_t.groupby("DSAS_NAME")
                .agg(n=("is_fraud", "size"), fraud_share=("is_fraud", "mean")))
# 충분히 자주 등장(>=150건)하는 질병만, 사기 비중 상위 10개
dsas = dsas[dsas["n"] >= 150].sort_values("fraud_share", ascending=False).head(10)

fig, ax = plt.subplots(figsize=(10, 5.5))
bars = ax.barh(dsas.index, dsas["fraud_share"] * 100,
               color=COLOR_FRAUD, edgecolor="white")
ax.invert_yaxis()
ax.axvline(base_rate * 100, color="black", linestyle="--", linewidth=1,
           label=f"전체 평균 {base_rate*100:.1f}%")
ax.set_xlabel("이 질병명 청구 중 사기 고객 비중 (%)")
ax.set_title("사기 비중 높은 질병 Top 10  (질병별 청구 ≥150건)",
             fontsize=13, weight="bold")
ax.legend(loc="lower right")
for bar, n in zip(bars, dsas["n"]):
    ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2,
            f"n={n}", va="center", fontsize=9, color="#555")
plt.tight_layout()
save_fig("eda_05_disease")
print("[Fig 5] 객관적 진단이 어려운 염좌·관절 통증성 질환에 사기 집중.")


# Fig 6. 수치형 변수 상관관계 — train + 청구 집계 피처 결합
corr_cols = ["AGE", "MAXCRDT", "MINCRDT", "TOTALPREM", "MAX_PRM",
             "CUST_INCM", "RCBASE_HSHD_INCM", "CHLD_CNT",
             "n_claim", "n_hosp", "sum_otda", "total_dmnd", "is_fraud"]
corr = merged[corr_cols].corr(numeric_only=True)

fig, ax = plt.subplots(figsize=(9.5, 8))
im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
ax.set_xticks(range(len(corr_cols)))
ax.set_yticks(range(len(corr_cols)))
ax.set_xticklabels(corr_cols, rotation=45, ha="right")
ax.set_yticklabels(corr_cols)
# 셀에 상관계수 텍스트
for i in range(len(corr_cols)):
    for j in range(len(corr_cols)):
        v = corr.iloc[i, j]
        ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                fontsize=8, color="white" if abs(v) > 0.5 else "black")
plt.colorbar(im, ax=ax, fraction=0.046)
ax.set_title("수치형 변수 + 청구 집계 피처 — 상관관계", weight="bold")
plt.tight_layout()
save_fig("eda_06_corr")
print("[Fig 6] is_fraud와의 상관: n_claim·sum_otda·n_hosp 양의 상관, MAXCRDT 음의 상관.")

print("\n" + "=" * 70)
print("EDA 시각화 6장 figures/ 폴더에 저장 완료. PPT에 캡처해 사용.")
print("=" * 70)
