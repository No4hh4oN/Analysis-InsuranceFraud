# 08 · Threshold tuning (step08_threshold.py)

> 운영 시 의심 threshold 결정. 기본값 0.5 vs PR 곡선 F1 최대점 비교.

## 수행 사항

- 5-fold out-of-fold 확률(`oof`) 산출 — 각 행이 valid fold 에서 한 번씩 예측된 확률
- threshold 0.00~1.00 을 101 포인트 균등 스캔 → F1·Precision·Recall 측정
- F1 최대점 = best threshold

산출물:
- `outputs/threshold_scan.csv` — 101 포인트 metric
- `outputs/threshold_best.json` — best threshold + 점수
- `figures/threshold_pr_curve.png` — PR 곡선 + best 점
- `figures/threshold_metric_scan.png` — threshold × F1/Precision/Recall

## 이슈 / 트러블슈팅

### OOF 확률을 쓴 이유
fold 별 best threshold 평균은 fold 분산 때문에 노이즈가 큼. **전체 학습 데이터를 OOF 로 한 번에 펼친 뒤 PR 곡선 위에서 최적점**을 찾는 게 더 안정적. 새 데이터에 일반화하는 threshold 후보로 더 신뢰성 있음.

## 성능

### 기존 threshold 0.5 vs F1 최대점 0.58

| Threshold | F1 | Precision | Recall |
|---|---:|---:|---:|
| 0.50 (기존) | 0.621 | 0.588 | 0.657 |
| **0.58 (F1 최대)** | **0.629** | **0.636** | 0.622 |
| Δ | **+0.008** | **+0.048** | -0.035 |

- **F1 +0.008** — 작은 개선이지만 동시에 **Precision +4.8pp**. 운영 시 "의심 후보 중 진짜 사기 비율" 이 58.8% → 63.6% 로 향상.
- Recall 은 -3.5pp 손실. 운영 정책에 따라 trade-off 결정.

### threshold scan 그림에서 보이는 패턴

| 영역 | 정책 의미 |
|---|---|
| t < 0.3 | **Recall 우선** — 사기를 거의 다 잡지만 의심 후보가 매우 많음 (Precision < 0.4) |
| t = 0.4 ~ 0.7 | **F1 plateau** — Precision 과 Recall 이 균형 (0.58~0.63) |
| t > 0.8 | **Precision 우선** — 의심 시 진짜 사기 80%+ 이지만 사기의 대다수를 놓침 (Recall < 0.4) |

> F1 plateau 가 넓다는 것은 **threshold 선택에 모델이 강건**하다는 신호. 0.5~0.65 사이 어느 값을 쓰든 F1 0.61+ 보장.

### 베이스라인(전체 평균) 대비 의미

- 사기율 8.76% (random baseline F1≈0.16, Precision≈0.088) 대비 F1 0.629 는 7배 이상의 상대적 개선.
- PR 곡선의 AUC = PR-AUC 0.663 (Step 4).

## 정책 권고

- **균형 정책 (F1 최대)**: t = 0.58 → P 0.636, R 0.622, F1 0.629
- **Recall 우선 정책**: t ≈ 0.40 → P ≈ 0.51, R ≈ 0.75 (사기 3/4 잡고 의심 후보 늘림)
- **Precision 우선 정책**: t ≈ 0.75 → P ≈ 0.75, R ≈ 0.45 (의심 후보 적게, 정밀하게)

PPT 슬라이드용 권고는 **t = 0.58 (F1 균형)** 이지만, 실제 운영진의 조사 capacity 에 따라 t 를 좌우로 조정한다는 흐름으로 보고.

## 사용 예시

```python
import json
with open("outputs/threshold_best.json") as f:
    cfg = json.load(f)
threshold = cfg["best_threshold"]  # 0.58

# 새 데이터 예측 시
proba = pipe.predict_proba(X_new)[:, 1]
y_pred = (proba >= threshold).astype(int)
```

## 다음 단계 (선택)

본 설계서 전 8단계 종료. 남은 작업은 모두 PPT 본 작성 또는 추가 실험.

- **PPT 슬라이드 본격 정리** — 8개 docs + 14개 figures + 6개 outputs 파일을 슬라이드 구성에 배치
- **추가 피처 엔지니어링** (선택) — 시계열 신호·챕터 상호작용
