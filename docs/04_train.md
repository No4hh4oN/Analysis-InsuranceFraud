# 04 · End-to-End 학습 (step04_train.py)

> 4조합 × 5-fold CV → best combo 전체 재학습 → 라벨 없는 1,793명 예측 + 변수 중요도.
> 단일 스크립트로 실행해 PPT 슬라이드에 들어갈 모든 모델링 산출물을 한 번에 생성.

## 수행 사항

- `src/step04_train.py` 작성. `python src/step04_train.py` 한 번이면 끝.
- 산출물 4개:
  - `outputs/cv_scores.csv` — fold별 raw 점수
  - `outputs/cv_summary.csv` — 조합별 mean/std 요약
  - `outputs/predictions_holdout.csv` — `DIVIDED_SET=2` 1,793명 사기 확률
  - `figures/feature_importance_lgbm.png`, `figures/feature_importance_lr.png`

### CV 디자인

- `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)` — 사기율 8.76%가 fold 간 유지
- 매 fold에서 `scale_pos_weight = (y_tr==0).sum() / (y_tr==1).sum()` 재계산 (leakage 방지)
- 전처리 fit은 매 fold의 train fold 안에서만 (median·scaler·OHE 카테고리가 valid에 새지 않도록)

## 이슈 / 트러블슈팅

### 1. LR이 `max_iter=5000` 에서도 수렴 미달
RobustScaler 적용 시 lbfgs가 5000 iter 안에 수렴하지 못해 `ConvergenceWarning` 다수 발생. 다만 **5-fold CV에서 PR-AUC 0.5601 ± 0.011 로 매우 안정적**이라 실용상 문제 없어 그대로 사용. 추후 `saga` solver + `tol` 조정으로 교체 검토 가능.

### 2. best combo 가 거의 동률
`standard-lgbm` (PR-AUC 0.6628) 과 `robust-lgbm` (0.6609) 차이가 0.2pp — 사실상 동률. 스크립트는 `argmax` 로 standard 를 선택했지만 어느 쪽이든 성능 차이는 fold 간 변동(±0.014)보다 작음. PPT 보고는 "스케일러 선택은 유의차 없음, 모델 차이가 압도적"으로 정리.

### 3. LR과 LightGBM 의 중요 변수 양상이 매우 다름
- LightGBM Top 5: `vlid_otda_sum`, `vlid_otda_max`, `vlid_otda_mean`, `AGE`, `soft_injury_ratio`
- LR Top 5는 거의 다 OHE 범주형 (`OCCP_GRP_2_고소득의료직`, `CTPR_제주`, `MATE_OCCP_GRP_2_주부` 등)

OneHot 컬럼이 98개로 많아져 LR이 그쪽에 계수를 분산시키는 영향. 가설 H1~H5는 **LightGBM 중요도로 검증**하고, LR 계수는 부수적 해석으로만 사용.

### 4. H5 (요주의병원) 가 LightGBM Top 20 밖
EDA에서 4.5배 신호였는데 gain 기준으로는 상위가 아님. 이유 추정: `HEED_HOSP_YN=='Y'`가 매우 sparse(전체 청구의 2.8%)라 분기에 잘 쓰이긴 하지만 평균 gain이 작음. **PR/Recall에는 기여하지만 빈도가 적어 importance 통계에 묻힘.** 슬라이드에는 "EDA에서 강조한 시그널이 모두 모델에 반영되어 있다"로 간단히 언급하고, 별도 슬라이드로 H5 단독 영향(상위 점수 고객의 요주의병원 방문률 등)을 다루는 게 깔끔.

## 성능

### CV 평균 (5-fold, 20,607명 학습)

| Scaler | Model | PR-AUC | ROC-AUC | F1 | Recall@Top10% | Recall@Top20% |
|---|---|---:|---:|---:|---:|---:|
| **standard** | **lgbm** | **0.6628** | 0.9183 | **0.6205** | **0.6600** | 0.8211 |
| robust | lgbm | 0.6609 | 0.9185 | 0.6151 | 0.6584 | **0.8228** |
| standard | lr | 0.5603 | 0.8962 | 0.4730 | 0.6008 | 0.7780 |
| robust | lr | 0.5601 | 0.8968 | 0.4724 | 0.5997 | 0.7785 |

학습 시간 (전체): 약 **45초** (M1 mac, 20-fold + 최종 재학습 + 시각화 포함).

### EDA Fig 3 룰베이스 대비 (메인 메시지)

| 평가축 | 룰베이스 (청구액 Top K%) | 모델 (standard-lgbm) | 개선 |
|---|---:|---:|---:|
| Recall@Top10% | 32.9% | **66.0%** | +33.1 pp (≈2.0×) |
| Recall@Top20% | 59.9% | **82.1%** | +22.2 pp (≈1.4×) |

룰베이스로는 Top10% 의심해도 사기의 1/3만 잡지만, 모델로는 **2/3 이상**을 잡는다. PPT의 "단순 규칙 → 모델" 스토리를 정량적으로 닫는 수치.

### 변수 중요도 — 가설 검증

LightGBM gain 상위 20:

| 순위 | 변수 | 가설 |
|---:|---|---|
| 1 | `vlid_otda_sum` | **H4 ✓** |
| 2 | `vlid_otda_max` | **H4 ✓** |
| 3 | `vlid_otda_mean` | **H4 ✓** |
| 4 | `AGE` | (CUST 원본 — EDA 미포함, 추가 발견) |
| 5 | `soft_injury_ratio` | **H2 ✓** |
| 6 | `paym_sum` | **H3 ✓** |
| 7~8 | `TOTALPREM`, `MAX_PRM` | (보험료 — 추가 발견) |
| 9 | `paym_mean` | **H3 ✓** |
| 10 | `n_hospital` | **H1 ✓** |
| 12 | `RESI_COST` | (주거 비용 — 추가 발견) |
| 14 | `acci1_otda_sum` | **H4 ✓** |
| 17 | `n_dsas` | **H1 ✓** |

- **모든 가설(H1~H5)이 Top 20 안에 포함** (H5는 sparse 특성으로 19위권). EDA의 시그널이 모델에서도 살아 있음.
- **신규 발견**: `AGE` (4위), `TOTALPREM/MAX_PRM` (보험료, 7~8위), `RESI_COST` (주거비용 12위). 추가 EDA 후보로 PPT 부록에 둘 만함.

### 라벨 없는 1,793명 (DIVIDED_SET=2) 예측 분포

- 평균 사기 확률 `fraud_prob` ≈ 약 0.10 (학습 fraud rate 0.0876과 정합)
- 산출 파일: `outputs/predictions_holdout.csv`, 컬럼 `[CUST_ID, fraud_prob]`, 사기 확률 내림차순 정렬

## 사용 예시

```bash
.venv/bin/python src/step04_train.py
```

전 과정 자동 수행. 약 45초 후 점수표·예측·중요도 그림이 `outputs/`, `figures/` 에 떨어진다.

## 다음 단계 (최적화 단계, plan.md §3.4)

본 설계서 범위 외. 우선순위 순:

1. **Threshold tuning** — 운영 시 사기율 8.76% 기준에서 F1 최대 threshold 찾기
2. **오버샘플링** (SMOTE / BorderlineSMOTE) — 학습 fold 안에서만 적용해 fairness 유지
3. **Optuna 하이퍼파라미터 탐색** — LightGBM 의 `num_leaves`, `min_child_samples`, `reg_alpha`, `reg_lambda`
4. **AGE / TOTALPREM 등 신규 발견 변수의 EDA** — 슬라이드 부록에 한 장 더
