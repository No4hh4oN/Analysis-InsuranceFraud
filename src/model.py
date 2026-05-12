"""
모델 빌더와 평가지표 헬퍼

  build_model(name)  → sklearn 호환 분류기 ("lr" | "lgbm")
  evaluate(y, p)     → dict (PR-AUC / ROC-AUC / F1 / Recall@Top K%)

EDA Fig 3 (룰베이스: Top K% 청구액 의심) 와 직접 비교하기 위해 Recall@Top K% 를 평가지표에 포함
"""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    roc_auc_score,
)

from lightgbm import LGBMClassifier


# 모델 빌더
def build_model(name: str, *, scale_pos_weight: float | None = None):
    """이름으로 분류기 생성

    Parameters
    ----------
    name : {"lr", "lgbm"}
        - "lr"  : LogisticRegression (베이스라인, class_weight=balanced)
        - "lgbm": LightGBM (메인, scale_pos_weight 기반 불균형 처리)
    scale_pos_weight
        LightGBM 전용. 보통 (정상수 / 사기수) — 사기율 8.76% → 약 10.4
        None이면 10.0 기본값
    """
    if name == "lr":
        # max_iter=5000 — RobustScaler 적용 시 분산이 매우 커서 lbfgs 수렴이 느림
        # (StandardScaler 에서는 수백 iter 안에 끝남). sklearn 1.8+ 에서 n_jobs는 deprecated 라 지정 안 함.
        return LogisticRegression(
            class_weight="balanced",
            max_iter=5000,
            solver="lbfgs",
            random_state=42,
        )

    if name == "lgbm":
        return LGBMClassifier(
            scale_pos_weight=scale_pos_weight if scale_pos_weight else 10.0,
            n_estimators=500,
            learning_rate=0.05,
            num_leaves=31,
            random_state=42,
            n_jobs=-1,
            verbosity=-1,
        )

    raise ValueError(f"unknown model: {name!r} (expected 'lr' or 'lgbm')")


# 평가지표
def recall_at_topk(y_true: np.ndarray, y_proba: np.ndarray, k_pct: float) -> float:
    """
    확률 상위 K% 를 의심으로 잡았을 때의 Recall = 전체 사기 중 잡힌 비율

    EDA Fig 3(룰베이스: 청구액 상위 K%)와 같은 축이라 직접 비교 가능
    """
    y_true = np.asarray(y_true)
    y_proba = np.asarray(y_proba)
    n_top = int(len(y_true) * k_pct / 100)
    top_idx = np.argsort(-y_proba)[:n_top]
    n_fraud = y_true.sum()
    return float(y_true[top_idx].sum() / n_fraud) if n_fraud else float("nan")


def evaluate(y_true, y_proba, threshold: float = 0.5) -> dict[str, float]:
    """
    학습/검증 폴드 점수표

    - PR-AUC  : 불균형 분류 1순위
    - ROC-AUC : 관례 보고용
    - F1      : threshold 기반 — 운영 점수 결정 시 참고
    - Recall@Top10% / Top20% : EDA Fig 3 룰베이스(Recall 32.9% / 59.9%)와 비교
    """
    y_true  = np.asarray(y_true)
    y_proba = np.asarray(y_proba)
    y_pred  = (y_proba >= threshold).astype(int)
    return {
        "PR-AUC":         float(average_precision_score(y_true, y_proba)),
        "ROC-AUC":        float(roc_auc_score(y_true, y_proba)),
        "F1":             float(f1_score(y_true, y_pred)),
        "Recall@Top10%":  recall_at_topk(y_true, y_proba, 10),
        "Recall@Top20%":  recall_at_topk(y_true, y_proba, 20),
    }
