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
# Scaler 비교 축
SCALERS: dict[str, type] = {
    "standard": StandardScaler,
    "robust":   RobustScaler,
}

# Hybrid 모드 — 의미상 같은 그룹(금융/소득)만 StandardScaler, 나머지 RobustScaler.
# 팀 토론 반영: 동일 단위의 금융 변수는 별도 그룹으로 통일된 스케일러 적용.
STANDARD_SCALE_COLS: tuple[str, ...] = (
    "RESI_COST", "TOTALPREM", "MAX_PRM",
    "CUST_INCM", "RCBASE_HSHD_INCM", "JPBASE_HSHD_INCM",
)


def build_preprocessor(
    num_cols: Sequence[str],
    cat_cols: Sequence[str],
    scaler: str = "robust",
) -> ColumnTransformer:
    """수치·범주 컬럼에 정책을 적용하는 ColumnTransformer.

    Parameters
    ----------
    num_cols, cat_cols
        features.split_columns() 결과를 그대로 넘기면 된다.
    scaler
        "standard" | "robust" | "hybrid"
          - standard / robust : 모든 수치 컬럼에 동일 스케일러
          - hybrid            : STANDARD_SCALE_COLS 6개만 StandardScaler,
                                나머지 수치는 RobustScaler

    Returns
    -------
    sklearn ColumnTransformer (sparse=False) — 모델 앞단에 그대로 부착.
    """
    cat_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("ohe",     OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    if scaler == "hybrid":
        std_cols = [c for c in STANDARD_SCALE_COLS if c in num_cols]
        rob_cols = [c for c in num_cols if c not in std_cols]
        std_pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler",  StandardScaler()),
        ])
        rob_pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler",  RobustScaler()),
        ])
        return ColumnTransformer(
            transformers=[
                ("num_std", std_pipe, std_cols),
                ("num_rob", rob_pipe, rob_cols),
                ("cat",     cat_pipe, list(cat_cols)),
            ],
            remainder="drop",
            verbose_feature_names_out=False,
        )

    if scaler not in SCALERS:
        raise ValueError(f"scaler must be 'standard' | 'robust' | 'hybrid', got {scaler!r}")

    num_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  SCALERS[scaler]()),
    ])
    return ColumnTransformer(
        transformers=[
            ("num", num_pipe, list(num_cols)),
            ("cat", cat_pipe, list(cat_cols)),
        ],
        remainder="drop",
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
