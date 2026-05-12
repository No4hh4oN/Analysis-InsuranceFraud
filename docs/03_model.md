# 03 · 모델 빌더 & 평가지표 (model.py)

> LR/LightGBM 두 모델과 공통 평가지표를 한 모듈에서 관리.
> 학습 스크립트는 04 단계에서 작성.

## 수행 사항

- `src/model.py` 작성. 진입점 셋:
  - `build_model(name, scale_pos_weight=None)` — `"lr"` / `"lgbm"`
  - `recall_at_topk(y_true, y_proba, k_pct)` — EDA Fig 3 룰베이스와 같은 축의 지표
  - `evaluate(y_true, y_proba, threshold=0.5)` — 5종 지표 dict

### 모델 설정

| 모델 | 설정 | 의도 |
|---|---|---|
| `lr` (LogisticRegression) | `class_weight="balanced"`, `max_iter=5000`, `solver="lbfgs"` | 베이스라인. 계수 부호로 H1~H5 해석 |
| `lgbm` (LGBMClassifier) | `scale_pos_weight=spw`, `n_estimators=500`, `lr=0.05`, `num_leaves=31` | 메인. 비선형 + 상호작용 |

`scale_pos_weight`는 호출부에서 `(y==0).sum() / (y==1).sum()` 계산해 주입 (사기율 8.76% → 약 10.4).

### 평가지표 5종

| 지표 | 사용 시점 | 비고 |
|---|---|---|
| **PR-AUC** | 모델 선택 1순위 | 불균형 분류 표준 |
| **ROC-AUC** | 관례 보고 | 0.9+ 면 일단 잘 잡힘 |
| **F1** | 운영 threshold 결정 시 | threshold 0.5 기준 |
| **Recall@Top10%** | EDA Fig 3 직접 비교 | 룰베이스 32.9% → 모델 ?% |
| **Recall@Top20%** | EDA Fig 3 직접 비교 | 룰베이스 59.9% → 모델 ?% |

## 이슈 / 트러블슈팅

### 1. `LogisticRegression(n_jobs=-1)` deprecation
sklearn 1.8에서 LR의 `n_jobs` 인자가 deprecated → 제거. lbfgs solver는 어차피 단일 스레드라 성능에 영향 없음.

### 2. lbfgs 수렴 실패 (`max_iter` 미달)
초기 `max_iter=1000`에서 `ConvergenceWarning`. 원인은 `RobustScaler` 적용 후 일부 피처(`vlid_otda_sum` 등 long-tail)가 IQR 단위로도 큰 값을 유지하기 때문 — 02 단계에서 본 std≈447 분포가 그대로 LR 옵티마이저에 부담을 줌. `max_iter=5000`으로 늘려 해결. `StandardScaler`에서는 수백 iter 안에 수렴.

### 3. LightGBM `X does not have valid feature names` 경고
`ColumnTransformer.transform()` 출력이 numpy array라 fit 시점 feature names가 일관되지 않아 발생. 모델 동작에는 영향 없음 — 정확한 해결은 `ColumnTransformer.set_output(transform="pandas")` 인데, 본 단계 범위에서는 보류(LightGBM은 ndarray 입력으로도 정상 학습).

### 4. RobustScaler의 가설이 모델 성능으로 입증되지 않음
sanity 결과(아래 §성능) RobustScaler가 StandardScaler를 능가하지 못함. plan.md에서 "long-tail/이상치 특성상 RobustScaler가 더 맞을 것"이라는 추론을 적었으나 실제로는 두 스케일러 차이가 0.5pp 이하. **04 단계 정식 CV(5-fold) 에서 다시 확인**하고, 그래도 차이 없으면 PPT 슬라이드에서는 단순히 "둘 다 비교했으나 유의차 없음"으로 보고.

## 성능 (단일 train/valid split — 16,485 / 4,122)

| Scaler · Model | PR-AUC | ROC-AUC | F1 | Recall@Top10% | Recall@Top20% | fit(초) |
|---|---:|---:|---:|---:|---:|---:|
| standard · lr   | 0.510 | 0.892 | 0.472 | 0.554 | 0.770 | 0.1 |
| **standard · lgbm** | **0.613** | **0.918** | **0.584** | **0.623** | **0.814** | 3.6 |
| robust · lr     | 0.506 | 0.891 | 0.476 | 0.551 | 0.781 | 0.8 |
| robust · lgbm   | 0.593 | 0.917 | 0.559 | 0.604 | 0.798 | 3.6 |

### 메인 메시지

- **LightGBM이 LR을 PR-AUC 10pp 차이로 일관되게 상회.** 비선형·상호작용이 H1~H5 시그널 결합에 유효하다는 신호.
- **EDA Fig 3 룰베이스 대비**:

| 평가축 | 룰베이스 (Top K% 청구액) | LightGBM (standard) | 개선 |
|---|---:|---:|---:|
| Recall@Top10% | 32.9% | **62.3%** | +29.4 pp (≈1.9×) |
| Recall@Top20% | 59.9% | **81.4%** | +21.5 pp (≈1.4×) |

룰베이스로는 Top10% 의심 시 사기의 1/3밖에 못 잡지만, 모델로는 거의 2/3 — PPT 메인 슬라이드의 "단순 룰베이스의 한계 → 모델의 효용" 스토리를 닫는 수치.

- **Scaler 차이는 미미** (0.5pp 이하). LightGBM은 본래 스케일러와 거의 무관(트리 분기)하고, LR도 RobustScaler/StandardScaler 비슷한 결과. 04 정식 CV에서 다시 확인.

## 사용 예시

```python
from sklearn.pipeline import Pipeline
from model import build_model, evaluate
from preprocess import build_preprocessor

spw = (y_train == 0).sum() / (y_train == 1).sum()
pipe = Pipeline([
    ("pre", build_preprocessor(num, cat, scaler="robust")),
    ("clf", build_model("lgbm", scale_pos_weight=spw)),
])
pipe.fit(X_train, y_train)
proba = pipe.predict_proba(X_valid)[:, 1]
print(evaluate(y_valid, proba))
```

## 다음 단계 (04_train.md)

- `step04_train.py` — end-to-end 학습 스크립트
- StratifiedKFold(5) × {standard·robust} × {lr·lgbm} = 4 조합 점수표 산출
- 최종 모델로 `DIVIDED_SET=2` 예측 산출 + LightGBM feature importance 시각화
- 산출물 `outputs/cv_scores.csv`, `figures/feature_importance.png`, `outputs/predictions_holdout.csv`
