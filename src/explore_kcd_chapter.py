"""
[프로젝트 핵심 가설 검증]

가설:
  · C 계열(암)         : 평균 지급액 ↑ , SIU 비율 ↓
  · M 계열(근골격계)   : 평균 지급액 중간 , SIU 비율 ↑
  · S 계열(손상·염좌)  : 청구 건수 ↑ , 반복 패턴

방법:
  CLAIM 테이블의 RESL_CD1(결과 코드, 주상병)은 KCD-10 코드.
  첫 글자가 챕터를 나타냄.
    예) S00 → S 챕터(손상), M51 → M 챕터(근골격계), C50 → C 챕터(악성신생물/암)

산출물:
  - outputs/kcd_chapter_summary.csv : 챕터별 청구건수·평균지급액·고객수·SIU율·Lift
  - figures/kcd_chapter_scatter.png : 평균 지급액 × SIU 비율 산점도
                                       (점 크기 = 청구 건수)
                                       → 'C는 우상단/M은 좌상단' 통념 반박 시각화
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from io_utils import load_cust, load_claim, save_fig, setup_korean_font, OUT_DIR

setup_korean_font()

cust = load_cust()
claim = load_claim()

# 라벨 있는 고객만 (DIVIDED_SET == 1)
train = cust[cust["SIU_CUST_YN"].isin(["Y", "N"])].copy()
train["is_fraud"] = (train["SIU_CUST_YN"] == "Y").astype(int)

# 청구 + 고객 라벨
claim_t = claim.merge(train[["CUST_ID", "is_fraud"]], on="CUST_ID")

# RESL_CD1 첫 글자 → KCD-10 챕터
claim_t["KCD_CHAPTER"] = claim_t["RESL_CD1"].astype(str).str[0]
# 코드가 비어있거나 알파벳이 아닌 행 제외
mask_alpha = claim_t["KCD_CHAPTER"].str.fullmatch(r"[A-Z]")
claim_t = claim_t[mask_alpha].copy()

# KCD-10 챕터 한국어 라벨 매핑
# (대표 챕터만 — 데이터에 실제 등장하는 것을 기준으로 표시)
KCD_LABEL = {
    "A": "감염성·기생충",
    "B": "감염성·기생충",
    "C": "신생물(암)",
    "D": "혈액/면역",
    "E": "내분비·대사",
    "F": "정신/행동",
    "G": "신경계",
    "H": "눈/귀",
    "I": "순환계",
    "J": "호흡계",
    "K": "소화계",
    "L": "피부",
    "M": "근골격계",
    "N": "비뇨생식기",
    "O": "임신/출산",
    "P": "주산기",
    "Q": "선천기형",
    "R": "증상/징후 NEC",
    "S": "손상/외상",
    "T": "손상/중독",
    "V": "외인",
    "W": "외인",
    "X": "외인",
    "Y": "외인",
    "Z": "보건서비스 접촉",
}

# ---------- 챕터별 집계 ----------
# 청구 단위 평균 지급금액·청구건수 + 고객 단위 사기율
# (고객 단위 사기율 = 그 챕터를 한 번이라도 청구한 적 있는 고객 중 사기 비중)

base_rate = train["is_fraud"].mean()

records = []
for chap, sub in claim_t.groupby("KCD_CHAPTER"):
    n_claim = len(sub)
    mean_paym = sub["PAYM_AMT"].mean()
    median_paym = sub["PAYM_AMT"].median()
    cust_ids = sub["CUST_ID"].unique()
    sub_cust = train[train["CUST_ID"].isin(cust_ids)]
    n_cust = len(sub_cust)
    fraud_rate = sub_cust["is_fraud"].mean() if n_cust else np.nan
    lift = fraud_rate / base_rate if base_rate else np.nan
    records.append({
        "챕터": chap,
        "라벨": KCD_LABEL.get(chap, chap),
        "청구건수": n_claim,
        "평균지급액": round(mean_paym, 0),
        "중앙값지급액": round(median_paym, 0),
        "이챕터청구_고객수": n_cust,
        "고객단위_SIU율": round(fraud_rate, 4),
        "Lift": round(lift, 2),
    })

summary = pd.DataFrame(records).sort_values("청구건수", ascending=False)
pd.set_option("display.max_columns", 20)
pd.set_option("display.width", 200)
pd.set_option("display.float_format", lambda x: f"{x:,.0f}" if x > 100 else f"{x:.4g}")

print("=" * 100)
print("[KCD-10 챕터별 — 청구건수·평균지급액·SIU율]  (전체 평균 SIU율 = 8.76%)")
print("=" * 100)
print(summary.to_string(index=False))

summary.to_csv(OUT_DIR / "kcd_chapter_summary.csv", index=False, encoding="utf-8-sig")
print(f"\n  saved → outputs/kcd_chapter_summary.csv")

# ---------- 시각화: 평균지급액 × SIU율 산점도 ----------
# 통념(우상단: 고액=고위험)이 깨지는 것을 한눈에 보여주는 차트
# 충분한 표본만 (청구 ≥1000)
plot_df = summary[summary["청구건수"] >= 1000].copy()

fig, ax = plt.subplots(figsize=(10, 7))
# 점 크기 = 청구 건수 (sqrt로 스케일 — 너무 커지지 않게)
sizes = np.sqrt(plot_df["청구건수"]) * 3
# 핵심 챕터(C/M/S)만 색상 강조, 나머지는 회색
def color_for(chap):
    return {"C": "#3182CE",   # 파랑 — 암
            "M": "#E53E3E",   # 빨강 — 근골격계
            "S": "#DD6B20"}.get(chap, "#A0AEC0")  # 주황 — 손상

colors = [color_for(c) for c in plot_df["챕터"]]
ax.scatter(plot_df["평균지급액"] / 10000, plot_df["고객단위_SIU율"] * 100,
           s=sizes, c=colors, alpha=0.75, edgecolor="white", linewidth=1.5)

# 챕터 라벨 표시
for _, row in plot_df.iterrows():
    ax.annotate(f"{row['챕터']}\n{row['라벨']}",
                (row["평균지급액"] / 10000, row["고객단위_SIU율"] * 100),
                fontsize=10, ha="center", va="center",
                color="white" if row["챕터"] in {"C", "M", "S"} else "#333",
                weight="bold")

# 전체 평균 SIU율 가로선
ax.axhline(base_rate * 100, color="black", linestyle="--", linewidth=1,
           label=f"전체 평균 SIU율 {base_rate*100:.1f}%")
ax.set_xlabel("청구 1건당 평균 지급금액 (만원)")
ax.set_ylabel("이 챕터 청구 경험 고객의 SIU율 (%)")
ax.set_title("질병 챕터별 — 지급금액 vs SIU 위험\n"
             "(점 크기 = 청구 건수.  통념: '고액=고위험' 이 깨지는지 확인)",
             fontsize=12, weight="bold")
ax.legend(loc="upper right")
ax.grid(linestyle=":", alpha=0.5)
plt.tight_layout()
save_fig("kcd_chapter_scatter")
print("[Fig] 챕터별 산점도 저장. C/M/S 위치 비교가 핵심 메시지.")
