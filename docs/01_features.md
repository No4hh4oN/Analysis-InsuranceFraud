# 01 · 피처 엔지니어링 (features.py)

> 청구 데이터(CLAIM_DATA, 1:N)를 고객 단위로 집계해 모델 입력 행렬을 만든다.
> sklearn 파이프라인 *외부*에서 한 번 수행 → 그 결과에 표준 전처리(`preprocess.py`)를 붙인다.

## 수행 사항

- `src/features.py` 작성. 진입점은 `build_features(claim, cust) → DataFrame`.
- 가설(plan.md H1~H5)별로 집계 함수를 분리해 변경에 강한 구조로 둠.

### 만든 피처 (26개)

| 그룹 (가설) | 피처 | 비고 |
|---|---|---|
| **H1 빈도·다양성** | `n_claim`, `n_hospital`, `n_dsas`, `n_chapter`, `chapter_entropy` | 챕터 엔트로피는 한 챕터 집중도(0) vs 다챕터 분산(↑) |
| **H3 지급액** | `paym_sum`, `paym_mean`, `paym_max` | `paym_per_claim`은 `paym_mean`과 중복이라 제외 |
| **H4 입통원 일수** | `vlid_otda_sum`, `vlid_otda_mean`, `vlid_otda_max`, `acci1_otda_sum`, `acci2_otda_sum`, `acci3_otda_sum` | 사고구분 1/2/3별 일수도 분리 (트러블슈팅 참고) |
| **H5 요주의병원** | `risky_hosp_visits`, `risky_hosp_ratio`, `risky_hosp_flag` | `HEED_HOSP_YN=='Y'` 기반 |
| **H2·H3 챕터 비중** | `M/S/C/K/I/N/J/D_ratio` + `etc_ratio` | 주요 8 + 나머지 1 (plan.md §2.2) |
| **H2 진단 텍스트** | `soft_injury_ratio` | 키워드: 요추·관절·척추·허리 |

CUST 원본 25개 컬럼은 그대로 join하므로 최종 행렬은 **22,400 × 51** (피처 49 + 메타 2).

## 이슈 / 트러블슈팅

### 1. ACCI_DVSN 코드 의미 미상
`ACCI_DVSN`은 1/2/3 세 값(분포: 3=75,606, 1=33,628, 2=9,786). 데이터 정의서가 없어 "입원/통원" 매핑이 명확치 않음.
**대응**: 임의로 매핑하지 않고 **세 코드별 `VLID_HOSP_OTDA` 합계를 그대로 분리한 피처(`acci{1,2,3}_otda_sum`)로 둠**. 모델이 가중을 학습. 추후 데이터 정의서가 확보되면 `hospz_ratio` 같은 직접적 파생 변수로 교체 가능.

### 2. 청구 0건 고객 처리
CUST에는 있지만 CLAIM에는 없는 고객을 위해 `cust.set_index("CUST_ID").join(agg, how="left")` 후 청구 집계 컬럼만 `fillna(0)`. CUST 원본 컬럼의 결측은 건드리지 않고 다음 단계(`SimpleImputer`)에 넘긴다.

### 3. 결측 위치 — 다음 단계로 인계
청구 집계 피처는 결측 0건. CUST 원본 10개 컬럼에 결측 존재 (`TOTALPREM` 5,791, `MINCRDT/MAXCRDT` 각 9,476, `MAX_PRM` 6,486, `CUST_INCM` 5,263 등). 비율로는 ~42%까지 가는 컬럼이 있어 단순 drop은 불가 — `preprocess.py`의 `SimpleImputer(median)` 으로 처리 예정.

## 성능

| 항목 | 값 |
|---|---|
| 입력 | CLAIM (119,020 × 39), CUST (22,400 × 25) |
| 출력 | (22,400 × 51) |
| 빌드 시간 | **2.2초** (M1 mac, pandas 단일 스레드) |
| 청구 집계 컬럼 결측 | **0건** |
| CUST 원본 결측 | 10개 컬럼 (다음 단계 imputer 처리) |

### 가설 검증 — 정상 vs 사기 평균 비교 (라벨 보유 20,607명)

| 피처 | 정상 | 사기 | 배율 | 가설 | EDA 보고치 |
|---|---:|---:|---:|---|---|
| `n_claim` | 4.53 | 13.60 | **3.0×** | H1 | 3× ✓ |
| `n_hospital` | 2.07 | 5.47 | **2.6×** | H1 | 2.6× ✓ |
| `n_dsas` | 3.38 | 10.80 | **3.2×** | H1 | 3.2× ✓ |
| `n_chapter` | 1.68 | 2.72 | **1.6×** | H1 | 1.6× ✓ |
| `chapter_entropy` | 0.34 | 0.66 | 2.0× | H1 | — |
| `paym_sum` | 3.6M | 11.8M | 3.3× | H3 | — (EDA는 분포 비교) |
| `vlid_otda_sum` | 31.4 | 203.3 | **6.5×** | H4 | 6.5× ✓ |
| `M_ratio` | 0.10 | 0.27 | 2.7× | H2 | — |
| `soft_injury_ratio` | 0.19 | 0.46 | 2.5× | H2 | — |
| `risky_hosp_visits` | 0.11 | 0.49 | **4.4×** | H5 | 4.5× ✓ |
| `risky_hosp_ratio` | 0.021 | 0.036 | 1.7× | H5 | — |

EDA에서 보고된 시그널 배율과 거의 일치 — 집계 로직 정합성 확인 완료.

> **관찰 한 가지**: 요주의병원은 *방문 건수*는 4.4배인데 *전체 청구 대비 비율*은 1.7배. 사기 고객은 요주의병원만 가는 게 아니라 일반 병원도 많이 방문하며 그중 요주의병원 비중이 같이 늘어나는 형태 — H1(doctor shopping)과 H5(요주의병원)가 서로 독립이 아니라는 신호. 모델에서 두 피처 모두 살려두고 상호작용은 트리계가 잡도록.

## 사용 예시

```python
from io_utils import load_cust, load_claim
from features import build_features, split_columns

claim = load_claim()
cust  = load_cust()

X = build_features(claim, cust)          # (22_400, 51)
num_cols, cat_cols = split_columns(X)    # 수치 40, 범주 9
```

## 다음 단계 (02_preprocess.md)

- `num_cols / cat_cols` 를 받아 `ColumnTransformer` 조립
- 수치: `SimpleImputer(median)` → Scaler (StandardScaler vs RobustScaler 비교)
- 범주: `SimpleImputer(most_frequent)` → `OneHotEncoder(handle_unknown="ignore")`
- 라벨/분할: `SIU_CUST_YN`(Y=1, N=0) + `DIVIDED_SET`(1=train, 2=holdout)
