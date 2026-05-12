"""
청구 데이터(CLAIM_DATA)를 고객 단위로 집계해 모델 입력 행렬을 생성

가설(plan.md)별로 피처를 묶어서 만들며, 누가 어떤 피처를 추가/삭제하더라도
_aggregate_* 단위 함수만 손대면 되도록 분리

진입점:
  build_features(claim, cust) → pd.DataFrame   (CUST_ID 인덱스, 모든 피처 포함)
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# KCD 챕터 — 개별 비중으로 둘 8개. 그 외는 'ETC'로 묶어 한 컬럼으로 합산.
# (plan.md §2.2: 노이즈·다중공선성 줄이고 LR 계수/LGBM importance 가독성 확보)
TARGET_CHAPTERS: tuple[str, ...] = ("M", "S", "C", "K", "I", "N", "J", "D")

# 요추/관절/척추 계열 진단명을 잡는 키워드 (H2 — soft injury)
SOFT_INJURY_KEYWORDS: tuple[str, ...] = ("요추", "관절", "척추", "허리")


# 그룹별 집계 - 가설별로 함수 하나

def _aggregate_frequency(claim: pd.DataFrame) -> pd.DataFrame:
    """H1 — 청구 빈도·다양성."""
    g = claim.groupby("CUST_ID")
    out = pd.DataFrame({
        "n_claim":    g.size(),
        "n_hospital": g["HOSP_CODE"].nunique(),
        "n_dsas":     g["DSAS_NAME"].nunique(),
        "n_chapter":  g["KCD_CHAPTER"].nunique(),
    })
    # 챕터 분포 엔트로피 (고객이 한 챕터에만 청구하면 0, 고르게 분산되면 큰 값)
    def _entropy(s: pd.Series) -> float:
        p = s.value_counts(normalize=True).to_numpy()
        return float(-(p * np.log(p)).sum())
    out["chapter_entropy"] = g["KCD_CHAPTER"].agg(_entropy)
    return out


def _aggregate_payment(claim: pd.DataFrame) -> pd.DataFrame:
    """H3 — 지급액 통계."""
    g = claim.groupby("CUST_ID")["PAYM_AMT"]
    return pd.DataFrame({
        "paym_sum":  g.sum(),
        "paym_mean": g.mean(),
        "paym_max":  g.max(),
    })


def _aggregate_hospital_days(claim: pd.DataFrame) -> pd.DataFrame:
    """H4 — 입통원 일수. ACCI_DVSN(사고구분 1/2/3)별로 분리해 모델이 가중을 잡게 둠."""
    g = claim.groupby("CUST_ID")["VLID_HOSP_OTDA"]
    out = pd.DataFrame({
        "vlid_otda_sum":  g.sum(),
        "vlid_otda_mean": g.mean(),
        "vlid_otda_max":  g.max(),
    })
    # ACCI_DVSN(1/2/3) × 일수 합계 — 사고구분 코드의 의미는 데이터 정의서가 없으니
    # 분리해서 그대로 두고 모델이 가중을 학습하도록 한다.
    by_acci = (claim.groupby(["CUST_ID", "ACCI_DVSN"])["VLID_HOSP_OTDA"]
                    .sum().unstack(fill_value=0))
    by_acci.columns = [f"acci{c}_otda_sum" for c in by_acci.columns]
    return out.join(by_acci, how="left").fillna(0)


def _aggregate_risky_hosp(claim: pd.DataFrame) -> pd.DataFrame:
    """H5 — 요주의병원(HEED_HOSP_YN=='Y') 신호."""
    flag = (claim["HEED_HOSP_YN"] == "Y").astype(int)
    g = flag.groupby(claim["CUST_ID"])
    visits = g.sum()
    n_claim = claim.groupby("CUST_ID").size()
    return pd.DataFrame({
        "risky_hosp_visits": visits,
        "risky_hosp_ratio":  visits / n_claim,
        "risky_hosp_flag":   (visits > 0).astype(int),
    })


def _aggregate_chapter_ratio(claim: pd.DataFrame) -> pd.DataFrame:
    """H2·H3 — KCD 챕터 비중. 주요 8개는 개별 컬럼, 나머지는 'ETC' 한 컬럼."""
    bucket = claim["KCD_CHAPTER"].where(
        claim["KCD_CHAPTER"].isin(TARGET_CHAPTERS), other="ETC"
    )
    counts = (pd.crosstab(claim["CUST_ID"], bucket)
                .reindex(columns=list(TARGET_CHAPTERS) + ["ETC"], fill_value=0))
    ratio = counts.div(counts.sum(axis=1), axis=0)
    ratio.columns = [f"{c}_ratio" for c in ratio.columns]
    return ratio


def _aggregate_soft_injury(claim: pd.DataFrame) -> pd.DataFrame:
    """H2 — 요추/관절/척추/허리 키워드 진단명 비율."""
    pattern = "|".join(SOFT_INJURY_KEYWORDS)
    hit = claim["DSAS_NAME"].str.contains(pattern, na=False).astype(int)
    g = hit.groupby(claim["CUST_ID"])
    n_claim = claim.groupby("CUST_ID").size()
    return pd.DataFrame({"soft_injury_ratio": g.sum() / n_claim})


# Interaction features (대안 B — 챕터 × 강한 변수)
# ----------------------------------------------------------------------
# "EDA에서 챕터별 사기 패턴이 다르다"는 사용자 통찰을 직접 신호로 주입.
# 단일 변수만으론 선형 모델(LR)이 못 잡고 얕은 트리도 놓치는 상호작용을
# 명시적으로 만들어 PR-AUC 를 더 끌어올린다.
INTERACTION_COLS: tuple[str, ...] = (
    "M_x_otda",          # 요추(M) × 입원일수 — 가장 강한 가설 (H2 × H4)
    "nonC_x_paym",       # 비-암(1-C) × 고액 청구 — "암 아닌데 고액"이면 의심 (H3)
    "risky_x_nclaim",    # 요주의병원 방문 × 청구건수 — H5 × H1
    "softinj_x_nhosp",   # 연조직 진단 비율 × 방문병원수 — H2 × H1
    "highrisk_chap_sum", # 사기율 높은 챕터(M,S,J,K) 비율 합 — 챕터 prior
)


def add_interactions(X: pd.DataFrame) -> pd.DataFrame:
    """build_features() 결과에 interaction 컬럼을 추가하여 반환.

    원본은 건드리지 않고 copy. 누락된 컬럼이 있으면 그냥 0으로 둠
    (예: 청구 0건 고객은 M_ratio=0, vlid_otda_sum=0이라 곱도 0)
    """
    out = X.copy()
    # 안전하게 .get(col, 0) 처리 — 컬럼이 빠져도 죽지 않게
    M_ratio   = out.get("M_ratio", 0)
    C_ratio   = out.get("C_ratio", 0)
    S_ratio   = out.get("S_ratio", 0)
    J_ratio   = out.get("J_ratio", 0)
    K_ratio   = out.get("K_ratio", 0)
    otda      = out.get("vlid_otda_sum", 0)
    paym      = out.get("paym_sum", 0)
    risky     = out.get("risky_hosp_visits", 0)
    nclaim    = out.get("n_claim", 0)
    softinj   = out.get("soft_injury_ratio", 0)
    nhosp     = out.get("n_hospital", 0)

    out["M_x_otda"]          = M_ratio * otda
    out["nonC_x_paym"]       = (1 - C_ratio) * paym
    out["risky_x_nclaim"]    = risky * nclaim
    out["softinj_x_nhosp"]   = softinj * nhosp
    out["highrisk_chap_sum"] = M_ratio + S_ratio + J_ratio + K_ratio
    return out


# Target encoding (대안 A — 챕터별 사기율 prior)
# ----------------------------------------------------------------------
# 각 청구의 KCD 챕터에 *그 챕터에서 관측된 사기율* 을 mapping 한 뒤,
# 고객별로 평균/가중합을 내서 한 컬럼으로 압축. raw 챕터 비율(M_ratio 등)
# 만으론 "M챕터가 위험하다"는 prior가 모델 안에서 암묵적으로 학습되지만,
# Bayesian smoothed 사기율을 직접 주면 적은 청구 고객도 강한 신호를 받음.
#
# Leakage 방지: 챕터별 사기율은 *반드시 train fold에서만* 계산해야 한다.
# step10 의 OOF 루프 안에서 호출.
TARGET_ENC_COL = "chapter_fraud_score"


def compute_chapter_fraud_rate(
    claim: pd.DataFrame,
    train_cust_ids,
    train_labels,
    alpha: float = 20.0,
) -> dict[str, float]:
    """train fold에서 KCD 챕터별 사기율(Bayesian smoothed) 산출.

    Parameters
    ----------
    claim : pd.DataFrame
        원본 청구 테이블 (KCD_CHAPTER 컬럼은 함수 내부에서 생성).
    train_cust_ids : array-like
        현재 fold의 train 고객 CUST_ID — 누수 방지를 위해 이 고객들의 청구만 사용.
    train_labels : pd.Series
        index=CUST_ID, value=0/1 (사기 라벨). train 고객만 들어 있어야 함.
    alpha : float
        smoothing 강도. 작은 표본 챕터(O,P,Q 등)는 전체 prior 쪽으로 끌어당김.

    Returns
    -------
    dict[chapter_letter -> smoothed_fraud_rate]
    """
    cl = claim.copy()
    cl["KCD_CHAPTER"] = cl["RESL_CD1"].astype(str).str[0]
    cl = cl[cl["CUST_ID"].isin(train_cust_ids)]

    # 청구마다 그 고객의 사기 라벨 부여 (1:N → 같은 라벨 반복)
    cl["label"] = cl["CUST_ID"].map(train_labels)

    # 챕터별 고객-수준 사기율: 그 챕터에 청구한 고객들의 (사기/전체)
    # 청구 건수가 아니라 *고객 unique* 기준 — 한 고객이 여러 번 청구해도 1로 셈
    chap_uniq = (cl.drop_duplicates(["CUST_ID", "KCD_CHAPTER"])
                   .groupby("KCD_CHAPTER")["label"]
                   .agg(["sum", "count"]))
    prior = train_labels.mean()
    rate = ((chap_uniq["sum"] + alpha * prior) /
            (chap_uniq["count"] + alpha))
    return rate.to_dict()


def apply_chapter_fraud_score(
    X: pd.DataFrame,
    claim: pd.DataFrame,
    chapter_rate: dict[str, float],
    prior: float,
) -> pd.DataFrame:
    """모든 고객(train+valid)에 챕터 사기율 가중합 컬럼 추가.

    score(cust) = Σ_chapter  chapter_ratio[cust, c] × chapter_rate[c]

    즉 *그 고객의 챕터 분포를 가중치로* 챕터별 사기율 prior을 평균낸 값.
    청구 0건 고객은 prior (전체 사기율) 로 채움.
    """
    out = X.copy()
    cl = claim.copy()
    cl["KCD_CHAPTER"] = cl["RESL_CD1"].astype(str).str[0]
    cl["chap_rate"] = cl["KCD_CHAPTER"].map(chapter_rate).fillna(prior)
    # 고객별 청구 단위 평균 — 청구가 많을수록 그 고객 챕터 분포가 충실히 반영
    score = cl.groupby("CUST_ID")["chap_rate"].mean()
    out[TARGET_ENC_COL] = out.index.to_series().map(score).fillna(prior)
    return out


# 시간 기반 피처 (step14 — 미사용 변수 활용)
# ----------------------------------------------------------------------
# CLAIM 의 RECP_DATE / ORIG_RESN_DATE 등 미사용 날짜 변수를 활용해
# 청구 패턴의 *시간 다이내믹스* 를 잡는다. 누적 합/평균만으로는 잡히지 않는
# sequential 신호 — "사고→청구 늦음", "청구 간격 불규칙", "최근 폭증" 등.
#
# 모든 날짜 컬럼은 YYYYMMDD int 포맷 — pd.to_datetime 으로 변환.
TIME_FEATURE_COLS: tuple[str, ...] = (
    "days_acci_to_claim_mean",   # 사고일 → 청구일 평균 일수 (늦을수록 의심)
    "days_acci_to_claim_max",    # 가장 늦게 접수한 청구 (이상치)
    "claim_interval_mean",       # 청구 간 평균 일수 (짧을수록 빈번)
    "claim_interval_std",        # 청구 간 간격 분산 (불규칙성)
    "claim_span_days",           # 첫 청구 ~ 마지막 청구 기간 (총 활동 기간)
    "claim_velocity",            # 최근 1/3 기간 청구 수 / 평균 (가속도)
    "max_same_day_claims",       # 같은 날 동시 청구 최대 수 (벌크 청구)
)


def _parse_yyyymmdd(s: pd.Series) -> pd.Series:
    """YYYYMMDD int → datetime. 결측·잘못된 값은 NaT."""
    return pd.to_datetime(s.astype("Int64").astype(str),
                          format="%Y%m%d", errors="coerce")


def add_time_features(X: pd.DataFrame, claim: pd.DataFrame) -> pd.DataFrame:
    """build_features() 결과에 시간 기반 피처를 추가하여 반환.

    누락 데이터 처리:
      - 청구 0건 또는 단일 청구 고객: interval/std/velocity 는 0
      - 모든 컬럼은 fillna(0) — "정보 없음 = 사기 신호 없음" 으로 둠
    """
    cl = claim[["CUST_ID", "RECP_DATE", "ORIG_RESN_DATE"]].copy()
    cl["recp"] = _parse_yyyymmdd(cl["RECP_DATE"])
    cl["acci"] = _parse_yyyymmdd(cl["ORIG_RESN_DATE"])
    cl["lag_days"] = (cl["recp"] - cl["acci"]).dt.days

    # 1) 사고 → 청구 lag (고객별 통계)
    lag_g = cl.groupby("CUST_ID")["lag_days"]
    lag_mean = lag_g.mean()
    lag_max  = lag_g.max()

    # 2) 청구 간 간격 — 고객별로 청구일 정렬 후 diff
    cl_sorted = cl.sort_values(["CUST_ID", "recp"])
    cl_sorted["interval"] = (cl_sorted.groupby("CUST_ID")["recp"]
                                       .diff().dt.days)
    int_g = cl_sorted.groupby("CUST_ID")["interval"]
    interval_mean = int_g.mean()
    interval_std  = int_g.std()

    # 3) 청구 span (첫청구 ~ 마지막청구 일수)
    span = (cl.groupby("CUST_ID")["recp"].max() -
            cl.groupby("CUST_ID")["recp"].min()).dt.days

    # 4) Velocity — 청구 기간을 3등분, 마지막 1/3 의 청구 수 / 평균 청구 수
    def _velocity(g: pd.DataFrame) -> float:
        if len(g) < 3:
            return 1.0   # 표본 부족 → 가속 없다고 가정
        t0, t1 = g["recp"].min(), g["recp"].max()
        if t0 == t1:
            return 1.0
        third = t0 + (t1 - t0) * (2 / 3)
        last_third = (g["recp"] >= third).sum()
        # 마지막 1/3 기간 청구 수 / (전체 청구 수 / 3)
        return float(last_third / (len(g) / 3))
    velocity = cl.groupby("CUST_ID").apply(_velocity, include_groups=False)

    # 5) 같은 날 동시 청구 최대 수
    same_day = (cl.groupby(["CUST_ID", "recp"]).size()
                  .groupby("CUST_ID").max())

    out = X.copy()
    # 인덱스 mapping — X 의 index 는 CUST_ID
    out["days_acci_to_claim_mean"] = out.index.to_series().map(lag_mean).fillna(0)
    out["days_acci_to_claim_max"]  = out.index.to_series().map(lag_max).fillna(0)
    out["claim_interval_mean"]     = out.index.to_series().map(interval_mean).fillna(0)
    out["claim_interval_std"]      = out.index.to_series().map(interval_std).fillna(0)
    out["claim_span_days"]         = out.index.to_series().map(span).fillna(0)
    out["claim_velocity"]          = out.index.to_series().map(velocity).fillna(1.0)
    out["max_same_day_claims"]     = out.index.to_series().map(same_day).fillna(0)
    return out


# 미사용 변수 활용 피처 (step15)
# ----------------------------------------------------------------------
# CLAIM 39개 컬럼 중 step14 까지 사용 8개 + 시간 3개 = 11개. 나머지 28개에서
# 사기/정상 비율 검증 후 강한 시그널 6개 변수 활용:
#
#   CHME_LICE_NO 다양성  사기 2.89배  (의사 6.5명 vs 2.3명 — 진짜 doctor shopping)
#   CAUS_CODE 다양성     사기 2.16배  (사고 원인 4.4개 vs 2.0개)
#   HOSP_SPEC_DVSN 다양성 사기 1.78배 (병원 종별 — 종합/의원/한방 섞음)
#   NON_PAY_RATIO         사기 0.32배 (역방향 — 보험금 노려 급여 위주 청구)
#   CHANG_FP_YN           사기율 10.2% vs 7.2% (설계사 변경)
#   HOSP_OTPA 입원기간     사기 max 1.19배 (약하지만 보조)
UNUSED_FEATURE_COLS: tuple[str, ...] = (
    "n_doctors",            # 방문 의사 면허번호 unique 수
    "n_caus_codes",         # 사고 원인 코드 unique 수
    "n_hosp_spec",          # 병원 종별 unique 수
    "non_pay_ratio_mean",   # 비급여 비율 평균 (역방향 신호)
    "non_pay_ratio_max",    # 비급여 비율 최대
    "fp_change_ratio",      # 설계사 변경 청구 비율
    "hosp_otpa_max",        # 최대 입원기간 (일)
    "hosp_otpa_mean",       # 평균 입원기간
)


def add_unused_features(X: pd.DataFrame, claim: pd.DataFrame) -> pd.DataFrame:
    """build_features() 결과에 미사용 변수 기반 피처 추가."""
    out = X.copy()
    g_cust = claim.groupby("CUST_ID")

    # 1) 방문 의사 다양성 — 같은 병원 안에서도 의사 옮기는 패턴 잡음
    n_doc = g_cust["CHME_LICE_NO"].nunique()

    # 2) 사고 원인 다양성
    n_caus = g_cust["CAUS_CODE"].nunique()

    # 3) 병원 종별 다양성 (종합/의원/한방 등 — 12종)
    n_spec = g_cust["HOSP_SPEC_DVSN"].nunique()

    # 4) 비급여 비율 (역방향 — 사기는 비급여 적게)
    npr_mean = g_cust["NON_PAY_RATIO"].mean()
    npr_max  = g_cust["NON_PAY_RATIO"].max()

    # 5) 설계사 변경 청구 비율
    fp_flag = (claim["CHANG_FP_YN"] == "Y").astype(int)
    fp_ratio = fp_flag.groupby(claim["CUST_ID"]).mean()

    # 6) 입원기간 (HOSP_OTPA_ENDT - HOSP_OTPA_STDT)
    stdt = pd.to_datetime(claim["HOSP_OTPA_STDT"], format="%Y%m%d", errors="coerce")
    endt = pd.to_datetime(claim["HOSP_OTPA_ENDT"], format="%Y%m%d", errors="coerce")
    otpa = (endt - stdt).dt.days
    otpa_g = otpa.groupby(claim["CUST_ID"])
    otpa_mean = otpa_g.mean()
    otpa_max  = otpa_g.max()

    out["n_doctors"]          = out.index.to_series().map(n_doc).fillna(0)
    out["n_caus_codes"]       = out.index.to_series().map(n_caus).fillna(0)
    out["n_hosp_spec"]        = out.index.to_series().map(n_spec).fillna(0)
    out["non_pay_ratio_mean"] = out.index.to_series().map(npr_mean).fillna(0)
    out["non_pay_ratio_max"]  = out.index.to_series().map(npr_max).fillna(0)
    out["fp_change_ratio"]    = out.index.to_series().map(fp_ratio).fillna(0)
    out["hosp_otpa_mean"]     = out.index.to_series().map(otpa_mean).fillna(0)
    out["hosp_otpa_max"]      = out.index.to_series().map(otpa_max).fillna(0)
    return out


# 진입점
def build_features(claim: pd.DataFrame, cust: pd.DataFrame) -> pd.DataFrame:
    """청구를 고객 단위로 집계하고 CUST 원본 변수와 merge한 모델 입력 행렬 생성

    Parameters
    ----------
    claim : pd.DataFrame
        CLAIM_DATA — 청구 1행 = 1건
    cust : pd.DataFrame
        CUST_DATA — 청구 0건인 고객도 포함될 수 있음 (이때 청구 집계 피처는 0)

    Returns
    -------
    pd.DataFrame
        index = CUST_ID, columns = (CUST 원본 + 청구 집계 피처)
        결측은 청구가 없는 고객의 집계 컬럼에만 발생 → 0으로 채움
    """
    # KCD 챕터(주상병코드 첫 글자) — 다른 헬퍼들이 공유하니 한 번에 만들어 둠
    claim = claim.copy()
    claim["KCD_CHAPTER"] = claim["RESL_CD1"].astype(str).str[0]

    parts = [
        _aggregate_frequency(claim),
        _aggregate_payment(claim),
        _aggregate_hospital_days(claim),
        _aggregate_risky_hosp(claim),
        _aggregate_chapter_ratio(claim),
        _aggregate_soft_injury(claim),
    ]
    agg = pd.concat(parts, axis=1)

    # CUST에 left-join — 청구 0건인 고객은 집계 컬럼이 NaN → 0
    out = cust.set_index("CUST_ID").join(agg, how="left")
    out[agg.columns] = out[agg.columns].fillna(0)
    return out


# 컬럼 분류 헬퍼 (preprocess.py 에서 사용)
# 모델에 들어가지 않는 식별/메타 컬럼
META_COLS: tuple[str, ...] = ("DIVIDED_SET", "SIU_CUST_YN")

# CUST 측 범주형 (One-Hot 대상)
CATEGORICAL_COLS: tuple[str, ...] = (
    "SEX", "RESI_TYPE_CODE", "FP_CAREER",
    "CTPR", "OCCP_GRP_1", "OCCP_GRP_2",
    "WEDD_YN", "MATE_OCCP_GRP_1", "MATE_OCCP_GRP_2",
)


def split_columns(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    """피처 행렬의 컬럼을 (수치, 범주) 두 그룹으로 나눠 돌려준다.

    META_COLS 는 어느 쪽에도 들어가지 않음 — 호출부에서 미리 제외할 것.
    """
    cat = [c for c in CATEGORICAL_COLS if c in df.columns]
    num = [c for c in df.columns if c not in cat and c not in META_COLS]
    return num, cat
