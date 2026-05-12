"""
Optuna 로 LightGBM 하이퍼파라미터 탐색

목적함수: 5-fold StratifiedKFold 평균 PR-AUC
전처리/모델 베이스: Step 4 best combo (standard scaler)
시도 횟수: N_TRIALS (기본 30)

산출물
  outputs/optuna_best_params.json   — 최적 파라미터 + 점수
  outputs/optuna_trials.csv         — 전 trial 점수표
  figures/optuna_history.png        — 탐색 곡선
  figures/optuna_param_importance.png  — 파라미터 중요도
"""

from __future__ import annotations

import json
import time
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import optuna
from optuna.visualization.matplotlib import (
    plot_optimization_history,
    plot_param_importances,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from lightgbm import LGBMClassifier
from sklearn.metrics import average_precision_score

from io_utils import (
    load_cust, load_claim, FIG_DIR, OUT_DIR,
    setup_korean_font, apply_plot_style, save_fig, PALETTE,
)
from features import build_features, split_columns
from preprocess import build_preprocessor, split_labeled
from model import build_model, evaluate

warnings.filterwarnings("ignore", message="X does not have valid feature names")
optuna.logging.set_verbosity(optuna.logging.WARNING)

setup_korean_font()
apply_plot_style()

N_SPLITS = 5
N_TRIALS = 30
SEED = 42
SCALER = "standard"


def objective(trial: optuna.Trial, X, y, num, cat) -> float:
    """5-fold CV 평균 PR-AUC — Optuna 목적함수"""
    params = {
        "n_estimators":      trial.suggest_int("n_estimators", 200, 800),
        "learning_rate":     trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
        "num_leaves":        trial.suggest_int("num_leaves", 15, 127),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
        "subsample":         trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree":  trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "reg_alpha":         trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
        "reg_lambda":        trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
    }

    kf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    scores = []
    for tr, va in kf.split(X, y):
        X_tr, X_va = X.iloc[tr], X.iloc[va]
        y_tr, y_va = y.iloc[tr], y.iloc[va]
        spw = (y_tr == 0).sum() / (y_tr == 1).sum()
        pipe = Pipeline([
            ("pre", build_preprocessor(num, cat, scaler=SCALER)),
            ("clf", LGBMClassifier(
                scale_pos_weight=spw,
                random_state=SEED,
                n_jobs=-1,
                verbosity=-1,
                **params,
            )),
        ])
        pipe.fit(X_tr, y_tr)
        proba = pipe.predict_proba(X_va)[:, 1]
        scores.append(average_precision_score(y_va, proba))
    return float(np.mean(scores))


def evaluate_best(params, X, y, num, cat) -> dict:
    """탐색 종료 후 best params 로 5-fold CV 풀 메트릭"""
    kf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    rows = []
    for fold, (tr, va) in enumerate(kf.split(X, y), start=1):
        X_tr, X_va = X.iloc[tr], X.iloc[va]
        y_tr, y_va = y.iloc[tr], y.iloc[va]
        spw = (y_tr == 0).sum() / (y_tr == 1).sum()
        pipe = Pipeline([
            ("pre", build_preprocessor(num, cat, scaler=SCALER)),
            ("clf", LGBMClassifier(
                scale_pos_weight=spw, random_state=SEED, n_jobs=-1,
                verbosity=-1, **params,
            )),
        ])
        pipe.fit(X_tr, y_tr)
        proba = pipe.predict_proba(X_va)[:, 1]
        sc = evaluate(y_va, proba)
        sc["fold"] = fold
        rows.append(sc)
    return pd.DataFrame(rows)


def plot_history_clean(study):
    """Optuna 기본 시각화 위에 우리 PALETTE 색 입혀서 저장"""
    ax = plot_optimization_history(study)
    fig = ax.figure
    fig.set_size_inches(11, 5)
    ax.set_title("")
    for line in ax.get_lines():
        line.set_color(PALETTE["fraud"])
        line.set_linewidth(2)
    ax.set_xlabel("Trial 번호")
    ax.set_ylabel("PR-AUC (5-fold 평균)")
    ax.grid(linestyle=":", color=PALETTE["grid"])
    save_fig("optuna_history")


def plot_importance_clean(study):
    """파라미터 중요도 — fanova 기반"""
    ax = plot_param_importances(study)
    fig = ax.figure
    fig.set_size_inches(10, 5.5)
    ax.set_title("")
    for patch in ax.patches:
        patch.set_color(PALETTE["fraud"])
        patch.set_edgecolor("white")
    ax.grid(axis="x", linestyle=":", color=PALETTE["grid"])
    save_fig("optuna_param_importance")


def main():
    print("[1/4] 데이터 + 피처")
    X = build_features(load_claim(), load_cust())
    X_lab, y_lab, _ = split_labeled(X)
    num, cat = split_columns(X)
    print(f"      labeled {X_lab.shape}")

    # 베이스라인 점수 (Step 4 기준) — 비교를 위해 다시 측정
    print("\n[2/4] baseline (Step 4 그대로) 5-fold")
    base_scores = evaluate_best(
        params=dict(
            n_estimators=500, learning_rate=0.05, num_leaves=31,
            min_child_samples=20,
        ),
        X=X_lab, y=y_lab, num=num, cat=cat,
    )
    base_pr = base_scores["PR-AUC"].mean()
    print(f"      PR-AUC {base_pr:.4f} ± {base_scores['PR-AUC'].std():.4f}")

    # Optuna 탐색
    print(f"\n[3/4] Optuna 탐색 ({N_TRIALS} trials × 5-fold)")
    t0 = time.time()
    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=SEED),
    )

    # 진행 상황 print
    def callback(study, trial):
        print(f"      trial {trial.number+1:2d}/{N_TRIALS}  "
              f"PR-AUC {trial.value:.4f}   best {study.best_value:.4f}")

    study.optimize(
        lambda t: objective(t, X_lab, y_lab, num, cat),
        n_trials=N_TRIALS,
        callbacks=[callback],
    )
    print(f"      탐색 종료 ({time.time()-t0:.0f}s)")

    # best params 로 최종 점수 측정
    print("\n[4/4] best params 평가 + 산출물 저장")
    best_scores = evaluate_best(study.best_params, X_lab, y_lab, num, cat)
    best_pr = best_scores["PR-AUC"].mean()

    # 점수 비교
    print(f"\n  {'baseline':<10s}: PR-AUC {base_pr:.4f}")
    print(f"  {'tuned':<10s}: PR-AUC {best_pr:.4f}   Δ {best_pr - base_pr:+.4f}")
    metrics = ["PR-AUC", "ROC-AUC", "F1", "Recall@Top10%", "Recall@Top20%"]
    cmp = pd.DataFrame({
        "baseline": base_scores[metrics].mean(),
        "tuned":    best_scores[metrics].mean(),
    })
    cmp["Δ"] = cmp["tuned"] - cmp["baseline"]
    print("\n=== 전 지표 비교 (5-fold 평균) ===")
    print(cmp.round(4).to_string())

    # 산출물 저장
    with open(OUT_DIR / "optuna_best_params.json", "w") as f:
        json.dump({
            "best_params":  study.best_params,
            "best_value":   study.best_value,
            "baseline_pr":  float(base_pr),
            "improvement":  float(best_pr - base_pr),
            "n_trials":     N_TRIALS,
        }, f, indent=2)
    print(f"\n  saved → outputs/optuna_best_params.json")

    trials_df = study.trials_dataframe()
    trials_df.to_csv(OUT_DIR / "optuna_trials.csv", index=False)
    print(f"  saved → outputs/optuna_trials.csv")

    plot_history_clean(study)
    plot_importance_clean(study)

    print("\n" + "=" * 70)
    print("step07 Optuna 탐색 완료.")
    print("=" * 70)


if __name__ == "__main__":
    main()
