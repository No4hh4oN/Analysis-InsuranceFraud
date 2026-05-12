"""
"요추·유방 등 특정 질병의 사기 시그널이 진짜 유의미한가?" 검증

EDA에서 청구 건 단위로 본 사기 비중은 두 가지 함정이 있다:
  (1) 한 사기 고객이 같은 질병으로 여러 번 청구하면 카운트가 부풀려짐
  (2) 청구 단위에서는 베이스라인 사기율이 8.76%가 아니라 더 높음
      (사기군이 청구를 3배 더 많이 하기 때문)

→ 본 스크립트는
  1.  청구 단위 사기 비중(이전과 동일)
  2.  고객 단위 사기 비중 — '이 질병명으로 청구한 적 있는 고객' 기준
  3.  카이제곱 검정으로 통계적 유의성(p-value, lift)
을 함께 산출.

산출물:
  - outputs/disease_significance.csv : 질병별 검정 결과 표
"""

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency

from io_utils import load_cust, load_claim, OUT_DIR

cust = load_cust()
claim = load_claim()

# 라벨 있는 고객만
train = cust[cust["SIU_CUST_YN"].isin(["Y", "N"])].copy()
train["is_fraud"] = (train["SIU_CUST_YN"] == "Y").astype(int)

# 청구 + 고객 라벨 결합
claim_t = claim.merge(train[["CUST_ID", "is_fraud"]], on="CUST_ID")

# 청구 단위 기준
base_claim = claim_t["is_fraud"].mean()
# 고객 단위 기준
base_cust = train["is_fraud"].mean()
print(f"[기준선] 청구 단위 사기 비중: {base_claim:.4f}  (참고)")
print(f"[기준선] 고객 단위 사기율  : {base_cust:.4f}  ← 모델링 베이스라인\n")

# 질병별 분석
# 자주 등장하는 질병만 (>=150 청구)
focus = (claim_t.groupby("DSAS_NAME")["is_fraud"]
                .agg(n_claim="size")
                .query("n_claim >= 150").index)

records = []
total_cust = train.shape[0]
total_fraud = train["is_fraud"].sum()
total_normal = total_cust - total_fraud

for dsas in focus:
    # 1. 청구 단위
    sub_claim = claim_t[claim_t["DSAS_NAME"] == dsas]
    n_claim = len(sub_claim)
    fraud_share_claim = sub_claim["is_fraud"].mean()

    # 2. 고객 단위: 그 질병명으로 한 번이라도 청구한 고객
    cust_with = claim_t.loc[claim_t["DSAS_NAME"] == dsas, "CUST_ID"].unique()
    sub_cust = train[train["CUST_ID"].isin(cust_with)]
    n_cust_with = len(sub_cust)
    fraud_rate_cust = sub_cust["is_fraud"].mean() if n_cust_with else np.nan

    # 3. 카이제곱: '이 질병 청구 경험 있음 × is_fraud' 2x2 교차표
    #   [ [이질병청구O & 사기O, 이질병청구O & 사기X],
    #     [이질병청구X & 사기O, 이질병청구X & 사기X] ]
    a = int((sub_cust["is_fraud"] == 1).sum())                # 질병O · 사기O
    b = n_cust_with - a                                       # 질병O · 사기X
    c = total_fraud - a                                       # 질병X · 사기O
    d = total_normal - b                                      # 질병X · 사기X
    table = np.array([[a, b], [c, d]])
    chi2, p, _, _ = chi2_contingency(table)

    # Lift = (해당 질병 고객 사기율) / (전체 사기율)
    lift = fraud_rate_cust / base_cust if base_cust else np.nan

    records.append({
        "DSAS_NAME": dsas,
        "청구건수": n_claim,
        "청구단위_사기비중": round(fraud_share_claim, 4),
        "이질병청구_고객수": n_cust_with,
        "고객단위_사기율": round(fraud_rate_cust, 4),
        "Lift_vs_base": round(lift, 2),
        "chi2": round(chi2, 2),
        "p_value": p,
        "유의(p<0.001)": "★" if p < 0.001 else ("○" if p < 0.05 else ""),
    })

result = pd.DataFrame(records).sort_values("고객단위_사기율", ascending=False)

# 두 가지 관점으로 Top 출력
pd.set_option("display.max_columns", 20)
pd.set_option("display.width", 200)
pd.set_option("display.float_format", lambda x: f"{x:.4g}")

print("=" * 90)
print("[A] 고객 단위 사기율 Top 15  (이 질병으로 청구한 적 있는 고객 중 사기 비율)")
print("=" * 90)
print(result.head(15).to_string(index=False))

print("\n" + "=" * 90)
print("[B] 사용자가 궁금해한 — 요추/유방/관절 관련 질병만 추출")
print("=" * 90)
mask = result["DSAS_NAME"].str.contains("요추|유방|관절|척추|경추")
print(result[mask].to_string(index=False))

# 저장
out = OUT_DIR / "disease_significance.csv"
result.to_csv(out, index=False, encoding="utf-8-sig")
print(f"\n  saved → outputs/disease_significance.csv  ({len(result)} 질병)")
