# 07 · Optuna 하이퍼파라미터 탐색 (step07_optuna.py)

> Step 4 best combo (`standard-lgbm`)에 Optuna TPE 샘플러로 30 trials.
> 목적함수 = 5-fold StratifiedKFold 평균 PR-AUC.

## 수행 사항

- `src/step07_optuna.py` 작성. `python src/step07_optuna.py` 한 번이면 끝.
- 탐색한 파라미터 8개:

| 파라미터 | 탐색 범위 | 기본값(Step 4) |
|---|---|---|
| `n_estimators` | 200~800 | 500 |
| `learning_rate` | 0.01~0.15 (log) | 0.05 |
| `num_leaves` | 15~127 | 31 |
| `min_child_samples` | 5~100 | 20 |
| `subsample` | 0.5~1.0 | 1.0 |
| `colsample_bytree` | 0.5~1.0 | 1.0 |
| `reg_alpha` | 1e-8~10 (log) | 0 |
| `reg_lambda` | 1e-8~10 (log) | 0 |

산출물:
- `outputs/optuna_best_params.json` — best 조합 + 점수
- `outputs/optuna_trials.csv` — 전 30 trial raw
- `figures/optuna_history.png` — 탐색 곡선
- `figures/optuna_param_importance.png` — fanova 기반 파라미터 중요도

## 이슈 / 트러블슈팅

### 1. 첫 실행 stdout 누락
`tail -100`로 백그라운드 실행 시 Python `print()` 가 line-buffered 라 stdout 이 파이프 끝까지 안 흘러갔음. 산출물 파일은 정상이지만 trial 진행 로그가 안 보임. 해결: `python -u` 옵션 (unbuffered) 으로 재실행 — 단, 산출물에는 영향 없음.

### 2. 사기율 8.76% 불균형 + `scale_pos_weight`
모든 trial 에서 `scale_pos_weight = (y_tr==0).sum() / (y_tr==1).sum()` 을 매 fold 에서 재계산. Step 6 결론에 따라 SMOTE 는 적용 안 함.

### 3. PR-AUC 개선폭이 fold 표준편차 안
+0.0073 의 개선은 baseline fold 표준편차(±0.020) 보다 작음. 통계적으로 유의차라고 단정 못 함. 다만 30 trials 중 상위 5개가 동일한 영역(낮은 lr + 큰 트리)에 수렴해 *경향성*은 있음 — PPT 에서는 "marginal 개선, 본질 신호는 베이스라인이 이미 잡고 있음" 톤으로 보고.

## 성능

### Best 파라미터 (Trial #23)

```json
{
  "n_estimators": 796,
  "learning_rate": 0.0184,
  "num_leaves": 90,
  "min_child_samples": 63,
  "subsample": 0.772,
  "colsample_bytree": 0.879,
  "reg_alpha": 6.2e-08,
  "reg_lambda": 8.04
}
```

> **패턴**: **느린 학습률(0.018) + 큰 트리(90 leaves) + 많은 트리(796)** + 강한 L2 정규화(`reg_lambda=8`). 정밀하게 학습시키되 과적합은 L2 로 누른 형태.

### 점수 비교 (5-fold CV 평균)

| 항목 | baseline (Step 4) | tuned | Δ |
|---|---:|---:|---:|
| **PR-AUC** | 0.6628 | **0.6701** | **+0.0073** |
| (탐색 범위 전체) | — | 0.6354 ~ 0.6701 | (worst-case -0.027) |

- 30 trials 중 상위 5개가 모두 PR-AUC 0.667 이상 — 탐색이 **올바른 영역(low lr, large trees, L2 강함)** 에 수렴함을 시사.
- worst trial(0.635) 은 작은 정규화 + 큰 lr 조합 — 과적합 영역.

### 파라미터 중요도 (fanova)

| 파라미터 | 중요도 |
|---|---:|
| `reg_alpha` | 0.21 |
| `n_estimators` | 0.19 |
| `learning_rate` | 0.16 |
| `reg_lambda` | 0.15 |
| `num_leaves` | 0.14 |
| `colsample_bytree` | 0.06 |
| `subsample` | 0.05 |
| `min_child_samples` | 0.03 |

- **정규화·트리수·학습률** 5개가 거의 균등하게 영향. 단일 dominant 파라미터 없음.
- `min_child_samples` 가 가장 영향 작음 — 우리 데이터(20k 행, 균형 안 맞춤)에서는 잎 크기 제약이 결정에 큰 차이 안 만듦.

## 채택 여부 결정

- **PPT 메인 보고는 baseline (Step 4) 그대로**. tuning 으로 +0.7pp 개선되지만 fold 표준편차(±2pp) 안이라 "유의차 있다"고 단정하기 어려움.
- 최종 제출용 예측(`outputs/predictions_holdout.csv`) 도 baseline 그대로 두고, **부록 슬라이드**로 "Optuna 탐색 → +0.7pp marginal 개선 / 5개 파라미터가 균등 영향 / 30 trials 로는 충분한 신호" 를 보고.
- 더 큰 개선을 노린다면 (a) trials 100+ 까지 늘리거나 (b) 추가 피처 엔지니어링이 우선 — hyperparameter ceiling 에 가까이 와 있음.

## 사용 예시

```python
# best params 로 모델 재학습 (필요 시)
import json
from lightgbm import LGBMClassifier

with open("outputs/optuna_best_params.json") as f:
    cfg = json.load(f)

clf = LGBMClassifier(
    scale_pos_weight=spw,
    random_state=42, n_jobs=-1, verbosity=-1,
    **cfg["best_params"],
)
```

## 다음 단계 (선택)

- **Threshold tuning** — F1 최대점, 운영 시 의심 threshold 결정 (15분 이내)
- **trials 확장** — 100~200 trials 로 0.67+ 영역 더 탐색 가능
- **추가 피처 엔지니어링** — 시계열 신호 (청구 간격·집중도), KCD 챕터 간 상호작용
- **PPT 슬라이드 본격 정리** — 지금까지 figures + docs 기반으로
