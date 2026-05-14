"""
plot_feature_rationale.py
-------------------------
PPT 캡처용 — 파생 변수 4라운드 각각을 "왜 추가했나"로 풀어낸 근거 표.
'5-1. 변수 탐색·4라운드 누적' 슬라이드의 동반 슬라이드용.

데이터 출처
  features.py 의 add_interactions / compute_chapter_fraud_rate /
  add_time_features / add_unused_features 주석 + README 3.5/3.10/3.12

산출물
  figures/feature_rationale.png
"""

from __future__ import annotations

import matplotlib.pyplot as plt

from io_utils import setup_korean_font, apply_plot_style, save_fig, PALETTE

setup_korean_font()
apply_plot_style()


# (섹션 헤더, [(피처, 정의, 추가 근거), ...])
SECTIONS = [
    ("Interaction · 5개 · +0.16pp", [
        ("M_x_otda",
         "M_ratio × 입원일수",
         "H2×H4 — 요추·관절(주관적 진단) 진단을 장기 입원과 결합한 고객 포착"),
        ("nonC_x_paym",
         "(1-C_ratio) × 지급액",
         "H3 — 암(C)은 고액·저위험. 非암 고액 청구만 위험 신호로 분리"),
        ("risky_x_nclaim",
         "요주의병원 방문 × 청구건수",
         "H5×H1 — 요주의병원을 '자주' 가는 결합 패턴"),
        ("softinj_x_nhosp",
         "연조직손상비율 × 방문병원수",
         "H2×H1 — 요추·허리 진단을 여러 병원에 돌리는 doctor shopping"),
        ("highrisk_chap_sum",
         "M+S+J+K 챕터 비중 합",
         "H2 — EDA에서 사기율 높게 나온 챕터를 한 축으로 묶음"),
    ]),
    ("Target encoding · 1개 · +0.18pp", [
        ("chapter_fraud_score",
         "챕터별 사기율 prior 가중평균\n(Bayesian smoothed α=20, OOF)",
         "raw 챕터 비율(M_ratio…)만으론 'M=위험' prior가 암묵적. 사기율을 "
         "직접 주면 청구 적은 고객도 강한 신호. 누수 방지 위해 train fold에서만 계산"),
    ]),
    ("Time features · 7개 · +1.72pp  ★ 최대 도약", [
        ("days_acci_to_claim_mean",
         "사고 → 청구 평균 일수",
         "사기 218일 vs 정상 127일 (1.72배) — '사고 후 늦게 청구' 패턴"),
        ("days_acci_to_claim_max",
         "가장 늦게 접수한 청구",
         "2.52배 — 사고 한참 뒤 끼워 넣는 청구 이상치"),
        ("claim_span_days",
         "첫 청구 ~ 마지막 청구 기간",
         "사기 1,438일 vs 정상 736일 (1.96배) — '오래 끄는' 활동 패턴"),
        ("claim_interval_mean / std",
         "청구 간 간격 평균·불규칙성",
         "누적 합·평균으론 안 잡히는 sequential 신호 (간격·리듬)"),
        ("claim_velocity",
         "최근 1/3 기간 청구 가속도",
         "활동 말기 청구 폭증 — 적발 직전 몰아치기 포착"),
        ("max_same_day_claims",
         "같은 날 동시 청구 최대 수",
         "1.19배 — 하루에 몰아 넣는 벌크 청구"),
    ]),
    ("Behavior features · 8개 · +1.37pp  ★ 두 번째 도약", [
        ("n_doctors",
         "방문 의사 수 (CHME_LICE_NO unique)",
         "2.89배 — 의사 6.5명 vs 2.3명. n_hospital이 못 잡던 '같은 병원 내 "
         "의사 옮기기' = 진짜 doctor shopping"),
        ("n_caus_codes",
         "사고 원인 코드 다양성",
         "2.16배 — 사고 원인 4.4개 vs 2.0개. 원인을 바꿔가며 청구"),
        ("n_hosp_spec",
         "병원 종별 다양성 (종합/의원/한방…)",
         "1.78배 — 종별을 섞어 쓰는 패턴"),
        ("non_pay_ratio_mean / max",
         "비급여 비율 평균·최대",
         "0.32배 (역방향) — 보험금 노려 급여 항목 위주로 청구"),
        ("fp_change_ratio",
         "설계사 변경 청구 비율",
         "1.42배 — 설계사 변경 후 청구가 몰림"),
        ("hosp_otpa_mean / max",
         "입원기간 평균·최대",
         "max 1.19배 — 약하지만 입원 편향 보조 신호"),
    ]),
]

COL_HEADERS = ["파생 변수", "정의", "추가 근거 (EDA 시그널 · 가설)"]
COL_X = [0.015, 0.20, 0.42]      # 각 열 좌측 x (axes 좌표)
COL_W = [0.185, 0.22, 0.565]     # 각 열 폭


def _wrap_rows():
    """(종류, 내용...) 튜플의 평탄한 리스트 — 종류는 'header'|'section'|'feat'."""
    rows = [("header",)]
    for title, feats in SECTIONS:
        rows.append(("section", title))
        for f in feats:
            rows.append(("feat", *f))
    return rows


def plot_feature_rationale(fname: str = "feature_rationale"):
    rows = _wrap_rows()
    n = len(rows)

    row_h = 0.052
    fig_h = n * row_h * 13.5
    fig, ax = plt.subplots(figsize=(15.5, fig_h))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    fig.subplots_adjust(left=0.015, right=0.985, top=0.99, bottom=0.01)

    y = 1.0
    for row in rows:
        kind = row[0]

        if kind == "header":
            ax.add_patch(plt.Rectangle((0, y - row_h), 1, row_h,
                                       facecolor=PALETTE["text"], edgecolor="none"))
            for x, txt in zip(COL_X, COL_HEADERS):
                ax.text(x, y - row_h / 2, txt, ha="left", va="center",
                        fontsize=11.5, color="white", weight="bold")
            y -= row_h

        elif kind == "section":
            ax.add_patch(plt.Rectangle((0, y - row_h), 1, row_h,
                                       facecolor=PALETTE["fraud"], edgecolor="none"))
            ax.text(0.015, y - row_h / 2, row[1], ha="left", va="center",
                    fontsize=11.5, color="white", weight="bold")
            y -= row_h

        else:  # feat
            name, define, reason = row[1], row[2], row[3]
            # 정의/근거가 길면 여러 줄 — 줄 수에 맞춰 행 높이 조정
            n_lines = max(define.count("\n") + 1,
                          1 + len(reason) // 52)
            h = row_h * max(1, n_lines * 0.82)
            ax.add_patch(plt.Rectangle((0, y - h), 1, h,
                                       facecolor="white",
                                       edgecolor=PALETTE["grid"], linewidth=0.8))
            ax.text(COL_X[0], y - h / 2, name, ha="left", va="center",
                    fontsize=10, color=PALETTE["fraud"], weight="bold",
                    family="monospace")
            ax.text(COL_X[1], y - h / 2, define, ha="left", va="center",
                    fontsize=9.7, color=PALETTE["text"])
            ax.text(COL_X[2], y - h / 2, _wrap(reason, 52),
                    ha="left", va="center", fontsize=9.7, color=PALETTE["text"])
            y -= h

    save_fig(fname)
    plt.close(fig)


def _wrap(text: str, width: int) -> str:
    """공백 기준 단순 줄바꿈 — 한 줄이 width 자 넘으면 다음 줄로."""
    words = text.split(" ")
    lines, cur = [], ""
    for w in words:
        if len(cur) + len(w) + 1 > width and cur:
            lines.append(cur)
            cur = w
        else:
            cur = f"{cur} {w}".strip()
    if cur:
        lines.append(cur)
    return "\n".join(lines)


def main():
    plot_feature_rationale()
    print("saved → figures/feature_rationale.png")


if __name__ == "__main__":
    main()
