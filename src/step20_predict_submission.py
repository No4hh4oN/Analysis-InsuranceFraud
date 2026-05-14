"""
step20_predict_submission.py
----------------------------
최종 모델(step18 정규화 앙상블)로 라벨 없는 DIVIDED_SET=2 (1,793명)의
사기자 여부를 예측 → 강의 교안 6절 형식의 제출 파일 생성.

흐름
  1. 5-fold OOF 로 F1-최적 threshold 결정 (step12 와 동일 방식)
  2. 전체 labeled 데이터로 3-모델(LR/LGBM/XGB) 재학습
  3. 라벨 없는 1,793명 예측 → weighted voting → threshold 적용
  4. outputs/4조_answer.csv  (columns = CUST_ID, 사기자여부 / 0·1)

모델·피처·가중치는 step18 과 동일 — 여기선 예측·제출만 담당.
"""

from __future__ import annotations

import json
import time

from sklearn.pipeline import Pipeline
from sklearn.metrics import average_precision_score

import pandas as pd

from io_utils import load_cust, load_claim, OUT_DIR
from features import (
    build_features, split_columns,
    add_interactions, compute_chapter_fraud_rate, apply_chapter_fraud_score,
    add_time_features, add_unused_features,
)
from preprocess import build_preprocessor, split_labeled
from step18_regularized import (
    make_model, get_oof_with_train,
    MODEL_NAMES, SCALER, SMOOTHING_ALPHA,
)
from step12_threshold_final import scan_thresholds


def apply_full_features(X_base, claim, chap_rate, prior):
    """step18 과 동일한 파생 변수 4종을 같은 순서로 적용."""
    X = add_interactions(X_base)
    X = apply_chapter_fraud_score(X, claim, chap_rate, prior)
    X = add_time_features(X, claim)
    X = add_unused_features(X, claim)
    return X


def main():
    print("[1/5] 데이터 + 베이스 피처")
    claim = load_claim()
    cust = load_cust()
    X = build_features(claim, cust)
    X_lab, y_lab, X_unlab = split_labeled(X)
    print(f"      labeled {X_lab.shape}   "
          f"제출 대상(DIVIDED_SET=2) {X_unlab.shape}")

    print("\n[2/5] 5-fold OOF — F1-최적 threshold 결정")
    probas_oof = {}
    for name in MODEL_NAMES:
        oof, _, _ = get_oof_with_train(X_lab, y_lab, claim, name)
        probas_oof[name] = oof
        print(f"      {name:>5s}  OOF PR-AUC "
              f"{average_precision_score(y_lab, oof):.4f}")

    weights = json.load(open(OUT_DIR / "step18_regularized.json"))["best_weights"]
    weighted_oof = sum(weights[n] * probas_oof[n] for n in MODEL_NAMES)
    print(f"      weighted OOF PR-AUC "
          f"{average_precision_score(y_lab, weighted_oof):.4f}   weights={weights}")

    scan = scan_thresholds(y_lab.to_numpy(), weighted_oof, n_points=101)
    best = scan.loc[scan["f1"].idxmax()]
    best_t = float(best["threshold"])
    print(f"      F1-최적 threshold = {best_t:.2f}  "
          f"(F1 {best['f1']:.4f}  P {best['precision']:.4f}  R {best['recall']:.4f})")

    print("\n[3/5] 전체 labeled 데이터로 3-모델 재학습")
    chap_rate = compute_chapter_fraud_rate(
        claim, X_lab.index, y_lab, alpha=SMOOTHING_ALPHA)
    prior = float(y_lab.mean())
    X_lab_full = apply_full_features(X_lab, claim, chap_rate, prior)
    X_unlab_full = apply_full_features(X_unlab, claim, chap_rate, prior)
    num, cat = split_columns(X_lab_full)
    spw = (y_lab == 0).sum() / (y_lab == 1).sum()

    fitted = {}
    for name in MODEL_NAMES:
        t0 = time.time()
        pipe = Pipeline([
            ("pre", build_preprocessor(num, cat, scaler=SCALER)),
            ("clf", make_model(name, spw)),
        ])
        pipe.fit(X_lab_full, y_lab)
        fitted[name] = pipe
        print(f"      {name:>5s} 학습 완료  ({time.time()-t0:.1f}s)")

    print(f"\n[4/5] 제출 대상 {len(X_unlab_full)}명 예측")
    probas_sub = {n: fitted[n].predict_proba(X_unlab_full)[:, 1]
                  for n in MODEL_NAMES}
    weighted_sub = sum(weights[n] * probas_sub[n] for n in MODEL_NAMES)
    pred_label = (weighted_sub >= best_t).astype(int)
    n_fraud = int(pred_label.sum())
    print(f"      사기 예측 {n_fraud}명 / {len(pred_label)}명  "
          f"({n_fraud / len(pred_label) * 100:.2f}%)")

    print("\n[5/5] 제출 파일 저장 (강의 교안 6절 형식)")
    result = pd.DataFrame({
        "CUST_ID":   X_unlab_full.index.to_numpy(),
        "사기자여부": pred_label,
    })
    out_path = OUT_DIR / "4조_answer.csv"
    result.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"      saved → {out_path}  ({len(result)}행)")
    print(result.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
