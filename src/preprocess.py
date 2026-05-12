"""
features.build_features() 결과 행렬에 붙일 표준 전처리(ColumnTransformer)

  수치 : SimpleImputer(median) → Scaler (StandardScaler ↔ RobustScaler 비교 축)
  범주 : SimpleImputer(most_frequent) → OneHotEncoder(handle_unknown="ignore")

라벨 분리 헬퍼 split_labeled() 도 함께 둔다 — DIVIDED_SET 코드의 의미와
라벨 결측 처리를 한 곳에서 결정하기 위함.
"""

from __future__ import annotations

from collections.abc import Sequence

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, RobustScaler, StandardScaler


# 전처리 파이프라인 빌더
# Scaler 비교 축 (plan.md §2.3)
SCALERS: dict[str, type] = {
    "standard": StandardScaler,
    "robust":   RobustScaler,
}


def build_preprocessor(
    num_cols: Sequence[str],
    cat_cols: Sequence[str],
    scaler: str = "robust",
) -> ColumnTransformer:
    """수치·범주 컬럼에 같은 정책을 적용하는 ColumnTransformer.

    Parameters
    ----------
    num_cols, cat_cols
        features.split_columns() 결과를 그대로 넘기면 된다.
    scaler
        "standard" | "robust" — 우리 집계 피처는 long-tail/이상치 특성이 강해
        기본값을 robust 로 둔다. 비교는 단일 인자만 바꿔 호출하면 됨.

    Returns
    -------
    sklearn ColumnTransformer (sparse=False) — 모델 앞단에 그대로 부착.
    """
    if scaler not in SCALERS:
        raise ValueError(f"scaler must be one of {list(SCALERS)}, got {scaler!r}")

    num_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  SCALERS[scaler]()),
    ])
    cat_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("ohe",     OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    return ColumnTransformer(
        transformers=[
            ("num", num_pipe, list(num_cols)),
            ("cat", cat_pipe, list(cat_cols)),
        ],
        remainder="drop",  # META 컬럼 등 명시 안 한 컬럼은 버린다 (안전장치)
        verbose_feature_names_out=False,
    )


# 라벨 / 분할 헬퍼
def split_labeled(
    X: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """라벨 보유 고객(학습)과 라벨 없는 고객(최종 제출)을 분리

    `DIVIDED_SET` 컬럼이 1=train, 2=test 로 구분되어 있지만 *진실의 원천은*
    `SIU_CUST_YN` 라벨 유무. test set은 라벨이 NaN이므로 모델 평가에는 쓰지 못하고 최종 모델로 예측을 산출해 제출하는 용도

    Returns
    -------
    X_labeled  : 학습/CV 에 쓸 피처 (DIVIDED_SET, SIU_CUST_YN 제거)
    y_labeled  : 이진 라벨 (Y=1, N=0)
    X_unlabeled: 라벨 없는 고객의 피처 (예측 산출용)
    """
    drop_cols = ["DIVIDED_SET", "SIU_CUST_YN"]
    mask = X["SIU_CUST_YN"].isin(["Y", "N"])
    X_labeled   = X.loc[mask].drop(columns=drop_cols)
    y_labeled   = (X.loc[mask, "SIU_CUST_YN"] == "Y").astype(int)
    X_unlabeled = X.loc[~mask].drop(columns=drop_cols)
    return X_labeled, y_labeled, X_unlabeled
