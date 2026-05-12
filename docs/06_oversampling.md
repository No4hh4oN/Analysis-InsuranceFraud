# 06 · 오버샘플링 비교 (step06_oversampling.py)

> 사기율 8.76% 불균형에 SMOTE 계열을 적용해 baseline(`scale_pos_weight`) 대비 효과를 본다.
> Step 4 best combo(`standard-lgbm`)에서 sampler만 교체.

## 수행 사항

4가지 전략 × 5-fold StratifiedKFold CV.

| 전략 | 의도 |
|---|---|
| **baseline** | `scale_pos_weight ≈ 10` (Step 4와 동일) |
| **SMOTE** | 사기 샘플을 k-NN 합성해 정상과 같은 수로 |
| **BorderlineSMOTE** | 결정 경계 근처 사기 샘플에서만 합성 |
| **SMOTETomek** | SMOTE 후 Tomek-links 정상-사기 페어 제거 |

SMOTE 계열은 클래스 균형을 합성으로 맞추므로 `scale_pos_weight=1` 로 설정해 이중 처리를 피함.

산출물:
- `outputs/oversample_compare.csv` (fold별 raw)
- `outputs/oversample_summary.csv` (전략별 mean/std)
- `figures/oversample_compare.png` (막대 비교)

## 이슈 / 트러블슈팅

### 1. SMOTE + `scale_pos_weight` 이중 처리 회피
오버샘플로 클래스를 균형 맞춘 뒤에도 `scale_pos_weight` 가 켜져 있으면 사기 클래스에 가중치가 두 번 적용됨. 해결: SMOTE 계열 사용 시 `scale_pos_weight = 1` 로 명시.

### 2. `imblearn.pipeline.Pipeline` 사용
sklearn 의 `Pipeline` 은 `fit_resample` 단계를 지원하지 않아 SMOTE 가 transform 단계로 인식돼 valid fold 에도 합성이 적용되는 leakage 위험. `imblearn.pipeline.Pipeline` 으로 교체해 학습 fold 안에서만 합성되도록 보장.

### 3. F1 점수 하락 — 합성 데이터의 확률 분포 변화
SMOTE 계열에서 F1이 0.02~0.03 떨어짐 (`baseline 0.621` → `SMOTE 0.596`). PR-AUC 는 거의 동일한데 F1만 떨어진 이유는 합성 데이터로 학습한 모델의 사기 확률 분포가 `0.5` 기준에서 더 많이 양성을 찍기 때문. **F1 비교는 무의미** — threshold tuning(다음 단계)에서 다시 측정해야 함.

## 성능 / 인사이트

### 5-fold CV 평균

| 전략 | PR-AUC | ROC-AUC | F1 | Recall@Top10% | Recall@Top20% |
|---|---:|---:|---:|---:|---:|
| **baseline** | **0.6628** ± 0.020 | 0.9183 | **0.6205** | 0.6600 | 0.8211 |
| SMOTE | 0.6607 ± 0.026 | 0.9209 | 0.5955 | 0.6634 | 0.8317 |
| BorderlineSMOTE | 0.6599 ± 0.029 | 0.9205 | 0.6002 | **0.6639** | **0.8328** |
| SMOTETomek | 0.6597 ± 0.024 | 0.9191 | 0.5896 | 0.6600 | 0.8223 |

### 결론 — **오버샘플링은 채택하지 않는다**

| 평가축 | baseline 대비 |
|---|---|
| PR-AUC | -0.002 ~ -0.003 (열위) — fold 표준편차(±0.020+) 안 |
| Recall@Top10% | +0.000 ~ +0.004 (미세 우세) — 표준편차 안 |
| Recall@Top20% | +0.001 ~ +0.012 (BorderlineSMOTE 약간 우세) — 표준편차 안 |
| F1 | -0.020 ~ -0.031 (열위, threshold 영향) |

**유의차 없음**. LightGBM이 `scale_pos_weight` 가중치 방식으로 이미 충분히 불균형을 다루고 있어 합성 샘플 추가로 얻는 정보가 없음. 학습 시간만 30~50% 늘어남 (18s → 24~28s).

> **PPT 분석 단계 근거**: "사기율 8.76% 불균형에 SMOTE/BorderlineSMOTE/SMOTETomek 세 가지를 5-fold CV 로 비교했으나 PR-AUC 변동이 fold 간 표준편차 안에 머물러 채택하지 않음. LightGBM 의 `scale_pos_weight` 만으로 충분."

## 다음 단계 (07_hyperparameter.md)

- Optuna 로 LightGBM 핵심 하이퍼파라미터 탐색
  - `num_leaves`, `min_child_samples`, `learning_rate`, `n_estimators`, `reg_alpha`, `reg_lambda`, `subsample`, `colsample_bytree`
- 목적함수: 5-fold CV 평균 PR-AUC 최대화
- 시도 횟수 30~50회, 시간 5~10분 예상
- 산출물: `outputs/optuna_best_params.json`, `figures/optuna_history.png`
