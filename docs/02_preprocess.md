# 02 · 전처리 파이프라인 (preprocess.py)

> `features.build_features()` 결과 행렬을 모델 입력 행렬로 바꾸는 sklearn `ColumnTransformer` 빌더.
> 라벨/분할 헬퍼도 같이 둠.

## 수행 사항

- `src/preprocess.py` 작성. 진입점은 두 개:
  - `build_preprocessor(num_cols, cat_cols, scaler="robust")` → `ColumnTransformer`
  - `split_labeled(X)` → `(X_labeled, y_labeled, X_unlabeled)`

### 전처리 정책

| 컬럼 그룹 | imputer | 다음 단계 |
|---|---|---|
| 수치 (40개) | `SimpleImputer(strategy="median")` | **Scaler — `"standard"` ↔ `"robust"` 비교 축** |
| 범주 (9개) | `SimpleImputer(strategy="most_frequent")` | `OneHotEncoder(handle_unknown="ignore")` |

기본 Scaler는 `RobustScaler` (우리 집계 피처의 long-tail/이상치 특성 때문 — plan.md §2.3).

### 라벨 분리 — `split_labeled(X)`

`SIU_CUST_YN`이 라벨이고 `DIVIDED_SET`은 보조 정보. test set(`DIVIDED_SET=2`)은 라벨이 NaN이므로 **모델 평가에는 쓰지 못하고 최종 예측 산출(제출)용**. 분리 기준은 `DIVIDED_SET`이 아니라 라벨 유무로 통일.

## 이슈 / 트러블슈팅

### 1. `DIVIDED_SET=2`의 정체 — 평가용이 아닌 제출용
plan.md 초안에는 "최종 holdout, 마지막 한 번만 점수 측정"으로 적었지만, 실제로 `SIU_CUST_YN`이 NaN이라 점수를 계산할 수 없음. **모델 평가는 `DIVIDED_SET=1` 안의 StratifiedKFold(5)** 로 통일하고, `DIVIDED_SET=2`는 학습 끝난 모델로 예측을 만들어 제출하는 용도로만 사용. plan.md §2.4 표현 추후 정정 필요.

### 2. RobustScaler 출력의 표준편차가 큰 게 정상인지
sanity 결과 `RobustScaler` 적용 후 수치 컬럼 평균 ≈ -4.89, 표준편차 ≈ 447. 처음엔 버그를 의심했지만 의도된 동작:
- `RobustScaler`는 **중앙값을 0, IQR을 1로** 맞춤. 평균·표준편차는 직접 정규화하지 않음.
- 우리 데이터는 일부 고객의 `vlid_otda_sum`이 수백~수천 일에 달해 IQR 단위로 환산해도 큰 값이 그대로 남음 — 이상치 정보를 보존하는 게 RobustScaler의 설계 목적.
- 반면 `StandardScaler`는 평균=0/표준편차=1을 강제하느라 이상치를 분산 안에 흡수 → 일반 고객의 신호가 압축됨.

이 차이가 모델 성능에 영향을 줄지는 03 단계에서 본다.

### 3. OneHot 후 컬럼 폭발
범주 9개 → OHE 98컬럼 (직업 코드·지역 코드의 카테고리가 많음). 트리계 모델에는 부담 미미하지만, LR은 정규화(`penalty="l2"`)에 의존해 처리. 추후 카디널리티가 높은 컬럼(`OCCP_GRP_2`, `MATE_OCCP_GRP_2`)은 target encoding 등으로 교체 검토 가능 — 본 단계 범위 밖.

## 성능

| 항목 | StandardScaler | RobustScaler |
|---|---|---|
| 입력 shape | (20,607, 49) | (20,607, 49) |
| 출력 shape | (20,607, 138) | (20,607, 138) |
| `fit_transform` 시간 | 0.06초 | 0.06초 |
| 출력 결측 | 0건 | 0건 |
| 출력 수치 중앙값 | -0.222 | **0.000** |
| 출력 수치 평균 | 0.000 | -4.892 |
| 출력 수치 표준편차 | 1.000 | 447.227 |

라벨/분할:

| 그룹 | 행 수 | 비고 |
|---|---|---|
| 학습 (DIVIDED_SET=1, 라벨 보유) | 20,607 | fraud rate 8.76% |
| 제출 (DIVIDED_SET=2, 라벨 NaN) | 1,793 | 최종 예측 산출용 |

## 사용 예시

```python
from features import build_features, split_columns
from preprocess import build_preprocessor, split_labeled
from io_utils import load_cust, load_claim

X = build_features(load_claim(), load_cust())
X_lab, y_lab, X_unlab = split_labeled(X)
num, cat = split_columns(X)

pre = build_preprocessor(num, cat, scaler="robust")  # 또는 "standard"
Xt = pre.fit_transform(X_lab)   # (20_607, 138)
```

## 다음 단계 (03_model.md)

- `Pipeline([("pre", pre), ("clf", model)])` 으로 모델 앞단에 부착
- 모델 빌더 두 개: `LogisticRegression(class_weight="balanced")`, `LightGBM(scale_pos_weight≈10)`
- 평가지표 헬퍼 `evaluate(y_true, y_proba)` — PR-AUC / ROC-AUC / Recall@Top K / F1
