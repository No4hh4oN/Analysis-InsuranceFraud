"""
[PPT 슬라이드 1: 데이터 소개(데이터 변형 X)]

산출물:
  - outputs/data_intro_summary.txt  : 행/열, 파일 크기, dtype 요약 텍스트
  - figures/sample_cust.png         : 고객 테이블 샘플 5행
  - figures/sample_claim.png        : 청구 테이블 샘플 5행
  - outputs/data_dictionary.csv     : 컬럼별 의미/dtype/결측률 정리

"""

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

from io_utils import (
    load_cust, load_claim, save_fig, setup_korean_font,
    DATA_DIR, OUT_DIR, PROJECT_ROOT,
)

setup_korean_font()

# 1) 데이터 로드
cust = load_cust()
claim = load_claim()

# 2) 기본 메타정보
cust_path = DATA_DIR / "CUST_DATA.csv"
claim_path = DATA_DIR / "CLAIM_DATA.csv"
cust_mb = cust_path.stat().st_size / 1024**2
claim_mb = claim_path.stat().st_size / 1024**2

# 메모리 점유량 (실제 로드 후 in-memory 사이즈)
cust_mem_mb = cust.memory_usage(deep=True).sum() / 1024**2
claim_mem_mb = claim.memory_usage(deep=True).sum() / 1024**2

lines = []
lines.append("=" * 70)
lines.append("데이터 소개 요약 (PPT 슬라이드 1번 캡션용)")
lines.append("=" * 70)
lines.append("")
lines.append("[1] 파일/형태")
lines.append(f"  · CUST_DATA.csv  : {cust.shape[0]:>7,} 행 × {cust.shape[1]:>2} 열"
             f"   (디스크 {cust_mb:.1f} MB / 메모리 {cust_mem_mb:.1f} MB)")
lines.append(f"  · CLAIM_DATA.csv : {claim.shape[0]:>7,} 행 × {claim.shape[1]:>2} 열"
             f"   (디스크 {claim_mb:.1f} MB / 메모리 {claim_mem_mb:.1f} MB)")
lines.append(f"  · 인코딩         : UTF-16 LE (한글 포함)")
lines.append(f"  · 조인 키        : CUST_ID  (CUST 1 : CLAIM N — 1인당 평균 "
             f"{len(claim)/len(cust):.1f}건)")
lines.append("")
lines.append("[2] 라벨")
vc = cust["SIU_CUST_YN"].value_counts(dropna=False)
lines.append(f"  · 사기(Y) : {vc.get('Y', 0):>6,}명")
lines.append(f"  · 정상(N) : {vc.get('N', 0):>6,}명")
lines.append(f"  · NaN     : {vc.isna().sum() + vc.get(float('nan'), 0):>6}명 (test 분할, DIVIDED_SET=2)")
labeled = vc.get("Y", 0) + vc.get("N", 0)
lines.append(f"  · 사기율  : {vc.get('Y',0)/labeled:.2%}  ← 불균형 분류")
lines.append("")
lines.append("[3] 변수 구성")
for name, df in [("CUST", cust), ("CLAIM", claim)]:
    n_num = df.select_dtypes(include="number").shape[1]
    n_obj = df.select_dtypes(include="object").shape[1]
    lines.append(f"  · {name:5s}: 수치형 {n_num}개 / 문자형 {n_obj}개")

summary_text = "\n".join(lines)
print(summary_text)

with open(OUT_DIR / "data_intro_summary.txt", "w") as f:
    f.write(summary_text)
print(f"\n  saved → outputs/data_intro_summary.txt")

# 샘플 행 이미지
def render_sample_table(df: pd.DataFrame, n_rows: int, n_cols: int,
                        title: str, fname: str):
    """DataFrame 앞 n_rows × n_cols 부분을 matplotlib 테이블 이미지로 저장."""
    sample = df.iloc[:n_rows, :n_cols].copy()
    # 보기 좋게 긴 값은 줄임
    for c in sample.columns:
        sample[c] = sample[c].astype(str).str.slice(0, 14)

    fig, ax = plt.subplots(figsize=(min(0.95 * n_cols + 1.5, 16), 0.55 * n_rows + 1.2))
    ax.axis("off")
    ax.set_title(title, fontsize=13, weight="bold", loc="left", pad=10)
    tbl = ax.table(
        cellText=sample.values,
        colLabels=sample.columns,
        cellLoc="center",
        loc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1.0, 1.4)
    # 헤더 셀 강조
    for (r, c), cell in tbl.get_celld().items():
        if r == 0:
            cell.set_facecolor("#2b6cb0")
            cell.set_text_props(color="white", weight="bold")
        else:
            cell.set_facecolor("#f7fafc" if r % 2 else "white")
    save_fig(fname)


# CUST는 컬럼이 25개라 너무 많아서, 핵심 12개만
cust_show_cols = ["CUST_ID", "SIU_CUST_YN", "DIVIDED_SET", "SEX", "AGE",
                  "OCCP_GRP_1", "CTPR", "WEDD_YN", "CHLD_CNT",
                  "TOTALPREM", "CUST_INCM", "MAXCRDT"]
render_sample_table(
    cust[cust_show_cols], n_rows=5, n_cols=len(cust_show_cols),
    title="CUST_DATA — 샘플 5행 (핵심 12개 컬럼)",
    fname="sample_cust",
)

# CLAIM도 핵심 12개만
claim_show_cols = ["CUST_ID", "POLY_NO", "RECP_DATE", "ACCI_DVSN",
                   "DSAS_NAME", "HOSP_CODE", "DMND_AMT", "PAYM_AMT",
                   "VLID_HOSP_OTDA", "NON_PAY_RATIO", "HEED_HOSP_YN",
                   "CHANG_FP_YN"]
render_sample_table(
    claim[claim_show_cols], n_rows=5, n_cols=len(claim_show_cols),
    title="CLAIM_DATA — 샘플 5행 (핵심 12개 컬럼)",
    fname="sample_claim",
)

# 4) 데이터 사전 (컬럼 의미 추정)
# 도메인 추정 의미 — 보험사 SIU 데이터셋 컨벤션 기반
DICT = {
    # CUST
    "CUST_ID": "고객 고유번호",
    "DIVIDED_SET": "데이터 분할 (1=train, 2=test)",
    "SIU_CUST_YN": "★ 타겟: 보험사기 의심 고객 여부 (Y/N)",
    "SEX": "성별 (1=남, 2=여)",
    "AGE": "나이",
    "RESI_COST": "거주 비용",
    "RESI_TYPE_CODE": "주거 형태 코드",
    "FP_CAREER": "본인 보험설계사 경력 여부 (Y/N)",
    "CUST_RGST": "고객 등록 연월 (YYYYMM)",
    "CTPR": "거주 시·도",
    "OCCP_GRP_1": "직업 대분류",
    "OCCP_GRP_2": "직업 소분류",
    "TOTALPREM": "총 납입 보험료",
    "MINCRDT": "최저 신용도",
    "MAXCRDT": "최고 신용도",
    "WEDD_YN": "결혼 여부",
    "MATE_OCCP_GRP_1": "배우자 직업 대분류",
    "MATE_OCCP_GRP_2": "배우자 직업 소분류",
    "CHLD_CNT": "자녀 수",
    "LTBN_CHLD_AGE": "막내 자녀 나이",
    "MAX_PAYM_YM": "최대 납입 연월",
    "MAX_PRM": "최대 보험료",
    "CUST_INCM": "고객 소득",
    "RCBASE_HSHD_INCM": "건강보험 기준 가구 소득",
    "JPBASE_HSHD_INCM": "직장 기준 가구 소득",
    # CLAIM
    "POLY_NO": "보험증권 번호",
    "ACCI_OCCP_GRP1": "사고 시 직업 대분류",
    "ACCI_OCCP_GRP2": "사고 시 직업 소분류",
    "CHANG_FP_YN": "설계사 변경 여부",
    "CNTT_RECP_SQNO": "접수 일련번호",
    "RECP_DATE": "접수일자 (YYYYMMDD)",
    "ORIG_RESN_DATE": "최초 사유일자",
    "RESN_DATE": "사유일자",
    "CRNT_PROG_DVSN": "현재 진행 구분",
    "ACCI_DVSN": "사고 구분 (1=상해/2=질병 추정)",
    "CAUS_CODE": "원인 코드",
    "CAUS_CODE_DTAL": "원인 코드 상세",
    "DSAS_NAME": "질병/상병명",
    "DMND_RESN_CODE": "청구 사유 코드",
    "DMND_RSCD_SQNO": "청구 사유 일련번호",
    "HOSP_OTPA_STDT": "통원/입원 시작일",
    "HOSP_OTPA_ENDT": "통원/입원 종료일",
    "RESL_CD1": "결과 코드(주상병)",
    "RESL_NM1": "결과명(주상병명)",
    "VLID_HOSP_OTDA": "유효 입원/통원 일수",
    "HOUSE_HOSP_DIST": "자택-병원 거리",
    "HOSP_CODE": "병원 코드",
    "ACCI_HOSP_ADDR": "사고/병원 주소",
    "HOSP_SPEC_DVSN": "병원 전문 구분",
    "CHME_LICE_NO": "한방/일반 면허 구분",
    "PAYM_DATE": "지급일",
    "DMND_AMT": "청구 금액",
    "PAYM_AMT": "지급 금액",
    "PMMI_DLNG_YN": "의료비 처리 구분 Y/N",
    "SELF_CHAM": "본인부담금",
    "NON_PAY": "비급여",
    "TAMT_SFCA": "총 본인부담",
    "PATT_CHRG_TOTA": "환자부담총액",
    "DSCT_AMT": "할인 금액",
    "COUNT_TRMT_ITEM": "치료 항목 수",
    "DCAF_CMPS_XCPA": "공제 후 자기부담 제외 금액",
    "NON_PAY_RATIO": "비급여 비율",
    "HEED_HOSP_YN": "요주의 병원 여부",
}

rows = []
for tbl_name, df in [("CUST", cust), ("CLAIM", claim)]:
    for col in df.columns:
        rows.append({
            "테이블": tbl_name,
            "컬럼": col,
            "의미(추정)": DICT.get(col, ""),
            "dtype": str(df[col].dtype),
            "결측률(%)": round(df[col].isna().mean() * 100, 2),
            "고유값수": df[col].nunique(dropna=True),
        })
data_dict = pd.DataFrame(rows)
data_dict.to_csv(OUT_DIR / "data_dictionary.csv", index=False, encoding="utf-8-sig")
print(f"  saved → outputs/data_dictionary.csv  ({len(data_dict)} rows)")
