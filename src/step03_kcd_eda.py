"""
step03_kcd_eda.py
-----------------
[프로젝트 주제 — "고액 청구 vs SIU 위험의 차이 분석"]
PPT 시안 수준으로 정돈된 EDA 시각화 5장 생성.

산출물 (figures/):
  - eda_kcd_01_scatter.png       : 챕터별 지급액 vs SIU율 산점도 (메시지: 통념 반박)
  - eda_kcd_02_dual_bar.png      : 챕터별 평균 지급액 ↔ SIU율 좌우 막대 (정량 비교)
  - eda_kcd_03_rule_fail.png     : 단순 룰베이스(청구액 상위 N%)의 한계
  - eda_kcd_04_diversity.png     : 한 고객이 청구한 챕터 수 — 사기 vs 정상
  - eda_kcd_05_M_breakdown.png   : M(근골격계) 안 세부 진단명별 SIU율

공통 스타일은 io_utils.apply_plot_style() 한 곳에서 관리.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from io_utils import (
    load_cust, load_claim, save_fig,
    setup_korean_font, apply_plot_style, add_footnote,
    PALETTE, CHAPTER_PALETTE, HIGHLIGHT_CHAPTERS,
)

# 한글 폰트 + 공통 스타일 적용
setup_korean_font()
apply_plot_style()

# KCD-10 챕터 한국어 라벨
KCD_LABEL = {
    "A": "감염성·기생충", "B": "감염성·기생충", "C": "신생물(암)",
    "D": "혈액/면역",     "E": "내분비·대사",   "F": "정신/행동",
    "G": "신경계",        "H": "눈/귀",          "I": "순환계",
    "J": "호흡계",        "K": "소화계",         "L": "피부",
    "M": "근골격계",      "N": "비뇨생식기",     "O": "임신/출산",
    "P": "주산기",        "Q": "선천기형",       "R": "증상/징후",
    "S": "손상/외상",     "T": "손상/중독",      "V": "외인",
    "W": "외인",          "X": "외인",            "Y": "외인",
    "Z": "보건접촉",
}

# 강조 챕터(C/M/S)만 진한 파스텔, 나머지는 옅은 블루그레이로 통일.
# 핵심 데이터에 시선이 가도록 색을 두 그룹으로만 나눔.
def chap_color(chap):
    if chap in HIGHLIGHT_CHAPTERS:
        return CHAPTER_PALETTE[chap]
    return PALETTE["normal"]


# ====================================================================
# 데이터 준비
# ====================================================================
cust = load_cust()
claim = load_claim()

train = cust[cust["SIU_CUST_YN"].isin(["Y", "N"])].copy()
train["is_fraud"] = (train["SIU_CUST_YN"] == "Y").astype(int)
base_rate = train["is_fraud"].mean()

# 청구 + 라벨 + KCD 챕터
claim_t = claim.merge(train[["CUST_ID", "is_fraud"]], on="CUST_ID")
claim_t["KCD_CHAPTER"] = claim_t["RESL_CD1"].astype(str).str[0]
claim_t = claim_t[claim_t["KCD_CHAPTER"].str.fullmatch(r"[A-Z]")].copy()

# 챕터별 집계
chap_stats = (
    claim_t.groupby("KCD_CHAPTER")
           .apply(lambda g: pd.Series({
               "n_claim":   len(g),
               "mean_paym": g["PAYM_AMT"].mean(),
           }), include_groups=False)
)

# 고객 단위 SIU율
rates = []
for chap in chap_stats.index:
    ids = claim_t.loc[claim_t["KCD_CHAPTER"] == chap, "CUST_ID"].unique()
    rates.append(train[train["CUST_ID"].isin(ids)]["is_fraud"].mean())
chap_stats["siu_rate"] = rates
chap_stats["label"] = [KCD_LABEL.get(c, c) for c in chap_stats.index]

# 표본이 충분한 챕터만 (청구 ≥ 1,000)
chap_main = chap_stats[chap_stats["n_claim"] >= 1000].copy()


# ====================================================================
# Fig 1. 챕터별 지급액 × SIU율 — 산점도 (PPT 메인 한 장)
# ====================================================================
fig, ax = plt.subplots(figsize=(13, 7.2))
fig.subplots_adjust(top=0.96, right=0.95, left=0.08, bottom=0.13)

sizes = np.sqrt(chap_main["n_claim"]) * 5
colors = [chap_color(c) for c in chap_main.index]

# 축 범위 먼저 고정 (라벨 잘림 방지)
xmax = chap_main["mean_paym"].max() / 10000 * 1.12
ax.set_xlim(0, xmax)
ax.set_ylim(0, 28)

# 통념 영역 음영 (우상단 = 고액·고위험: 통념이 옳다면 점이 몰려야 할 영역)
ax.axhspan(base_rate * 100, 28,
           xmin=(200 / xmax), xmax=1.0,
           alpha=0.06, color=PALETTE["fraud"])
ax.text(xmax * 0.98, 26,
        "통념 영역: '고액 = 고위험'\n(데이터는 텅 비어 있다)",
        ha="right", va="top", fontsize=11,
        color=PALETTE["fraud"], style="italic", weight="bold", alpha=0.9)

ax.scatter(chap_main["mean_paym"] / 10000, chap_main["siu_rate"] * 100,
           s=sizes, c=colors, alpha=0.88,
           edgecolor="white", linewidth=2.2, zorder=3)

# 강조 챕터(C/M/S)는 라벨 박스, 나머지는 점 위 글자
for chap, row in chap_main.iterrows():
    x, y = row["mean_paym"] / 10000, row["siu_rate"] * 100
    if chap == "C":   # 우측 끝 — 점 왼쪽으로 라벨
        ax.annotate(f"{chap} · {row['label']}",
                    xy=(x, y), xytext=(-18, 0),
                    textcoords="offset points",
                    ha="right", va="center",
                    fontsize=12, weight="bold", color=chap_color(chap))
    elif chap in {"M", "S"}:
        ax.annotate(f"{chap} · {row['label']}",
                    xy=(x, y), xytext=(0, 22),
                    textcoords="offset points",
                    ha="center", fontsize=12, weight="bold",
                    color=chap_color(chap))
    else:
        ax.annotate(chap, xy=(x, y), ha="center", va="center",
                    fontsize=9, color="white", weight="bold")

# 평균선 — 왼쪽 가장자리에 작게 라벨
ax.axhline(base_rate * 100, color=PALETTE["rule"], linestyle="--",
           linewidth=1, alpha=0.4)
ax.text(xmax * 0.02, base_rate * 100 + 0.5,
        f"전체 평균 SIU율 {base_rate*100:.1f}%",
        ha="left", fontsize=11, color=PALETTE["text"], weight="bold")

ax.set_xlabel("청구 1건당 평균 지급금액 (만원)")
ax.set_ylabel("이 챕터 청구 경험 고객의 SIU율 (%)")
ax.grid(axis="y", linestyle=":", color=PALETTE["grid"])
ax.spines["left"].set_visible(False)
add_footnote(fig, "데이터: 보험사 SIU 데이터셋 — 라벨 보유 고객 20,607명, 청구 119,020건.")
save_fig("eda_kcd_01_scatter")


# ====================================================================
# Fig 2. 챕터별 평균 지급액 ↔ SIU율  좌우 막대 (정량 비교)
# ====================================================================
# 디자인: 가운데에 챕터 라벨, 좌측으로 지급액(파/회), 우측으로 SIU율(빨/회)
# 자라는 'Tornado/Dumbbell' 스타일 — 같은 챕터의 두 지표가 한눈에 비교됨.
top = chap_main.sort_values("n_claim", ascending=False).head(8).iloc[::-1]

fig = plt.figure(figsize=(14, 7))
fig.subplots_adjust(top=0.94, bottom=0.12, left=0.06, right=0.97, wspace=0.0)
gs = fig.add_gridspec(1, 3, width_ratios=[6, 2, 6])
left  = fig.add_subplot(gs[0])
mid   = fig.add_subplot(gs[1])
right = fig.add_subplot(gs[2])

y_pos = np.arange(len(top))

# 왼쪽: 평균 지급액 — 안쪽(오른쪽)으로 자라게 보이도록 x축 반전
bars_l = left.barh(y_pos, top["mean_paym"] / 10000,
                   color=[chap_color(c) for c in top.index],
                   edgecolor="white", height=0.65)
left.invert_xaxis()
left.set_xlabel("평균 지급액 (만원, 청구 1건당)", color=PALETTE["subtext"])
left.grid(axis="x", linestyle=":", color=PALETTE["grid"])
left.spines["bottom"].set_color(PALETTE["grid"])
left.spines["left"].set_visible(False)
left.set_yticks([])
for bar, v, chap in zip(bars_l, top["mean_paym"], top.index):
    # 강조 막대(C·M·S)는 진한 색이라 흰 글씨, 나머지(회색)는 본문 색
    is_highlight = chap in {"C", "M", "S"}
    left.text(bar.get_width(), bar.get_y() + bar.get_height() / 2,
              f"  {v/10000:,.0f}만  ", va="center", ha="left",
              fontsize=11,
              color="white" if is_highlight else PALETTE["text"],
              weight="bold")

# 가운데: 챕터 라벨
mid.set_xlim(0, 1); mid.set_ylim(-0.5, len(top) - 0.5)
mid.axis("off")
for i, chap in enumerate(top.index):
    mid.text(0.5, i, f"{chap}\n{KCD_LABEL.get(chap, chap)}",
             ha="center", va="center", fontsize=11.5,
             color=PALETTE["text"], weight="bold")

# 오른쪽: SIU율
bars_r = right.barh(y_pos, top["siu_rate"] * 100,
                    color=[chap_color(c) for c in top.index],
                    edgecolor="white", height=0.65)
right.axvline(base_rate * 100, color=PALETTE["rule"], linestyle="--",
              linewidth=1, alpha=0.5)
right.text(base_rate * 100, len(top) - 0.3,
           f"  평균 {base_rate*100:.1f}%",
           fontsize=11, color=PALETTE["text"], va="bottom", weight="bold")
right.set_xlabel("고객 단위 SIU율 (%)", color=PALETTE["subtext"])
right.set_yticks([])
right.grid(axis="x", linestyle=":", color=PALETTE["grid"])
right.spines["bottom"].set_color(PALETTE["grid"])
right.spines["left"].set_visible(False)
for bar, v, chap in zip(bars_r, top["siu_rate"], top.index):
    # 우측은 막대가 보통 길지 않아 글씨가 막대 바깥에 위치 — 본문 색 그대로
    right.text(bar.get_width(), bar.get_y() + bar.get_height() / 2,
               f"  {v*100:.1f}%", va="center",
               fontsize=11, color=PALETTE["text"], weight="bold")

# 차트 안에는 제목 두지 않음 (PPT 슬라이드에서 제목 부착).
add_footnote(fig, "지급액 = PAYM_AMT (실제 지급된 보험금).  SIU율 = 그 챕터를 청구한 적 있는 고객 중 사기 비율.")
save_fig("eda_kcd_02_dual_bar")


# ====================================================================
# Fig 3. 단순 룰베이스의 한계 — 청구액 상위 N% 의심 시 사기 잡힘률
# ====================================================================
cust_paym = claim.groupby("CUST_ID")["PAYM_AMT"].sum().rename("paym_total")
df = train.merge(cust_paym, on="CUST_ID", how="left").fillna({"paym_total": 0})
df = df.sort_values("paym_total", ascending=False).reset_index(drop=True)

N = len(df)
total_fraud = df["is_fraud"].sum()
percents = np.arange(1, 101)
recalls = [df["is_fraud"].iloc[:int(N * p / 100)].sum() / total_fraud
           for p in percents]
precisions = [df["is_fraud"].iloc[:int(N * p / 100)].mean()
              for p in percents]

fig, ax = plt.subplots(figsize=(13, 6.8))
fig.subplots_adjust(top=0.96, right=0.86, left=0.08, bottom=0.14)

# 선 그리기
recall_line, = ax.plot(percents, np.array(recalls) * 100,
                       color=PALETTE["fraud"], linewidth=2.8)
precision_line, = ax.plot(percents, np.array(precisions) * 100,
                          color=PALETTE["text"], linewidth=2.4)
random_line, = ax.plot(percents, percents, linestyle=":",
                       color=PALETTE["subtext"], linewidth=1.4)

# 범례 대신 — 각 선 끝에 직접 라벨 (그래프와 안 겹치는 빈 공간 활용)
# Recall·Precision 은 우측 그림 밖에, 무작위 기준선은 그래프 안 대각선 위에 회전 텍스트로.
ax.text(101, recalls[-1] * 100, "  Recall",
        va="center", ha="left", fontsize=11,
        color=PALETTE["fraud"], weight="bold")
ax.text(101, precisions[-1] * 100, "  Precision",
        va="center", ha="left", fontsize=11,
        color=PALETTE["text"], weight="bold")
# 무작위 기준선 — 대각선 위에 살짝 띄워 회전 텍스트로
ax.text(72, 76, "무작위 기준선", fontsize=10,
        color=PALETTE["subtext"], rotation=33,
        rotation_mode="anchor", va="bottom", ha="center")

# Top 10% 강조 — 곡선 사이 빈 공간(좌하단)에 박스 배치
i = 10
ax.scatter([i], [recalls[i - 1] * 100], color=PALETTE["fraud"],
           zorder=5, s=140, edgecolor="white", linewidth=2.5)
ax.annotate(
    f"Top 10% 의심 시\nRecall {recalls[i-1]*100:.1f}%   Precision {precisions[i-1]*100:.1f}%",
    xy=(i, recalls[i - 1] * 100),
    xytext=(20, 55),
    textcoords="data",
    fontsize=11, color=PALETTE["text"], weight="bold",
    bbox=dict(boxstyle="round,pad=0.7", facecolor="white",
              edgecolor=PALETTE["fraud"], linewidth=1.4),
    arrowprops=dict(arrowstyle="->", color=PALETTE["fraud"], lw=1.2),
)

# 보조 설명 — Recall · Precision 의미 한 줄 캡션 (그래프 안 깔끔하게)
ax.text(0.02, -0.13,
        "Recall = 전체 사기 중 잡힌 비율   |   Precision = 의심한 사람 중 진짜 사기 비율",
        transform=ax.transAxes, fontsize=10, color=PALETTE["subtext"])

ax.set_xlabel("청구 지급액 합계 기준 — 상위 K% 의심")
ax.set_ylabel("백분율 (%)")
ax.set_xlim(0, 100); ax.set_ylim(0, 100)
ax.grid(linestyle=":", color=PALETTE["grid"])
add_footnote(fig, "고객별 PAYM_AMT 합계로 내림차순 정렬 후 상위 K% 의심 시뮬레이션.")
save_fig("eda_kcd_03_rule_fail")

print(f"\n[Fig 3 캡션 숫자] Top10% → Recall {recalls[9]*100:.1f}%, "
      f"Precision {precisions[9]*100:.1f}%")
print(f"[Fig 3 캡션 숫자] Top20% → Recall {recalls[19]*100:.1f}%, "
      f"Precision {precisions[19]*100:.1f}%")


# ====================================================================
# Fig 4. 챕터 다양성 — 사기 vs 정상 (한 고객이 몇 챕터를 청구?)
# ====================================================================
n_chap = (claim_t.groupby("CUST_ID")["KCD_CHAPTER"]
                 .nunique().rename("n_chapter"))
df2 = train.merge(n_chap, on="CUST_ID", how="left").fillna({"n_chapter": 0})

max_bin = 10
df2["n_chapter_capped"] = df2["n_chapter"].clip(upper=max_bin)

fig, ax = plt.subplots(figsize=(13, 6.5))
fig.subplots_adjust(top=0.96, right=0.95, left=0.08, bottom=0.14)

xs = np.arange(0, max_bin + 1)
for label, color, offset in [("N", PALETTE["normal"], -0.2),
                             ("Y", PALETTE["fraud"], 0.2)]:
    sub = df2.loc[df2["SIU_CUST_YN"] == label, "n_chapter_capped"]
    pct = (sub.value_counts(normalize=True)
              .reindex(xs, fill_value=0) * 100)
    ax.bar(xs + offset, pct.values, width=0.4,
           label=f"{'사기' if label=='Y' else '정상'} (n={len(sub):,})",
           color=color, edgecolor="white", linewidth=1.2)

mean_n = df2.loc[df2["SIU_CUST_YN"] == "N", "n_chapter"].mean()
mean_y = df2.loc[df2["SIU_CUST_YN"] == "Y", "n_chapter"].mean()

# 평균 화살표 주석
ax.text(0.99, 0.95, f"평균 챕터 수\n정상  {mean_n:.2f}    사기  {mean_y:.2f}  "
        f"({mean_y/mean_n:.1f}배)",
        transform=ax.transAxes, ha="right", va="top",
        fontsize=12, color=PALETTE["text"], weight="bold",
        bbox=dict(boxstyle="round,pad=0.7", facecolor="#F8FAFC",
                  edgecolor=PALETTE["grid"]))

ax.set_xticks(xs)
xtick_labels = [str(x) for x in xs]
xtick_labels[-1] = f"{max_bin}+"
ax.set_xticklabels(xtick_labels)
ax.set_xlabel("한 고객이 청구한 KCD 챕터 종류 수")
ax.set_ylabel("그룹 내 비율 (%)")
ax.legend(loc="upper right", bbox_to_anchor=(1.0, 0.82))
ax.grid(axis="y", linestyle=":", color=PALETTE["grid"])
add_footnote(fig, "KCD-10 챕터 = 주상병코드(RESL_CD1)의 첫 글자. 10 이상은 10+ 로 합산.")
save_fig("eda_kcd_04_diversity")


# ====================================================================
# Fig 5. M(근골격계) 안 — 세부 진단명별 SIU율 Top 12
# ====================================================================
m_claims = claim_t[claim_t["KCD_CHAPTER"] == "M"]
dsas_in_m = (m_claims.groupby("DSAS_NAME")
                     .agg(n_claim=("CUST_ID", "size"),
                          n_cust=("CUST_ID", "nunique")))
dsas_in_m = dsas_in_m[dsas_in_m["n_claim"] >= 50].copy()
sui_rates = []
for d in dsas_in_m.index:
    ids = m_claims.loc[m_claims["DSAS_NAME"] == d, "CUST_ID"].unique()
    sui_rates.append(train[train["CUST_ID"].isin(ids)]["is_fraud"].mean())
dsas_in_m["siu_rate"] = sui_rates
dsas_in_m = dsas_in_m.sort_values("siu_rate", ascending=False).head(12)[::-1]

fig, ax = plt.subplots(figsize=(13, 7))
fig.subplots_adjust(top=0.96, right=0.97, left=0.28, bottom=0.12)

# '요추/관절' 들어가는 진단명은 더 강하게 색칠 (강조)
def bar_color(name):
    return PALETTE["M"] if any(k in name for k in ["요추", "관절", "척추"]) \
        else PALETTE["muted"]

bars = ax.barh(dsas_in_m.index, dsas_in_m["siu_rate"] * 100,
               color=[bar_color(d) for d in dsas_in_m.index],
               edgecolor="white", height=0.7)

ax.axvline(base_rate * 100, color=PALETTE["rule"], linestyle="--",
           linewidth=1, alpha=0.6)
ax.text(base_rate * 100 + 0.5, -0.5,
        f"전체 평균 {base_rate*100:.1f}%",
        fontsize=11, color=PALETTE["text"], weight="bold")

# 값 + 표본 크기 라벨 — 퍼센트는 볼드로 강조, n 표본 크기는 보조 텍스트
for bar, n_cust, rate in zip(bars, dsas_in_m["n_cust"], dsas_in_m["siu_rate"]):
    ax.text(bar.get_width() + 0.6, bar.get_y() + bar.get_height() / 2,
            f"{rate*100:.1f}%",
            va="center", fontsize=11.5, color=PALETTE["text"], weight="bold")
    # n=표본수는 옅은 회색으로 작게
    ax.text(bar.get_width() + 5.5, bar.get_y() + bar.get_height() / 2,
            f"(n={n_cust})",
            va="center", fontsize=10, color=PALETTE["subtext"])

ax.set_xlabel("이 진단명을 한 번이라도 청구한 고객의 SIU율 (%)")
ax.set_xlim(0, max(dsas_in_m["siu_rate"]) * 100 * 1.25)
ax.grid(axis="x", linestyle=":", color=PALETTE["grid"])
ax.spines["left"].set_visible(False)
ax.tick_params(axis="y", length=0)
add_footnote(fig, "M 챕터 청구 ≥50건인 진단명만. n = 고객 수.")
save_fig("eda_kcd_05_M_breakdown")

print("\n" + "=" * 70)
print("EDA 시각화 5장 (figures/eda_kcd_01~05) 재생성 완료 — PPT 스타일 적용.")
print("=" * 70)
