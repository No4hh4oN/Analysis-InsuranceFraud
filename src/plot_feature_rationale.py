"""
plot_feature_rationale.py
-------------------------
PPT 캡처용 — 파생 변수 4라운드의 효과(+pp) 가로 막대 + 한 줄 근거 태그.
'5-1. 변수 탐색·4라운드 누적' 슬라이드에 바로 쓸 수 있는 간소 버전.

데이터 출처
  outputs/feature_ablation.csv (Interaction/TE 효과)
  README 3.5/3.10/3.12 (Time/Behavior 효과·근거)

산출물
  figures/feature_rationale.png
"""

from __future__ import annotations

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

from io_utils import setup_korean_font, apply_plot_style, save_fig, PALETTE

setup_korean_font()
apply_plot_style()


# (라운드, 변수 수, 효과 pp, 강조 여부, 한 줄 근거)
# 효과 큰 순으로 — "변수 탐색이 ROI 크다"가 시각적으로 바로 읽히게.
ROUNDS = [
    ("Time features", 7, 1.72, True,
     "사고 후 늦게(218 vs 127일) · 오래(1,438 vs 736일) 청구하는 시간 패턴"),
    ("Behavior features", 8, 1.37, True,
     "방문 의사 수 2.89배 — 같은 병원서 의사 옮기는 진짜 doctor shopping"),
    ("Target encoding", 1, 0.18, False,
     "챕터별 사기율 prior을 직접 주입 — 청구 적은 고객도 강한 신호"),
    ("Interaction", 5, 0.16, False,
     "가설 결합을 곱으로 주입 — M챕터×입원, 비암×고액, 요주의×청구"),
]


def plot_feature_rationale(fname: str = "feature_rationale"):
    fig, ax = plt.subplots(figsize=(14, 5.6))
    fig.subplots_adjust(left=0.16, right=0.97, top=0.90, bottom=0.12)

    n = len(ROUNDS)
    ys = list(range(n))[::-1]   # 위에서부터 Time → Interaction
    bar_h = 0.46

    for y, (name, n_var, pp, hot, reason) in zip(ys, ROUNDS):
        color = PALETTE["fraud"] if hot else PALETTE["normal"]
        ax.barh(y, pp, height=bar_h, color=color,
                edgecolor="white", zorder=3)

        # 막대 끝 효과 라벨
        ax.text(pp + 0.04, y, f"+{pp:.2f}pp",
                va="center", ha="left", fontsize=12.5,
                color=PALETTE["text"], weight="bold")

        # 왼쪽: 라운드명 + 변수 수
        ax.text(-0.06, y + 0.13, name, va="center", ha="right",
                fontsize=12.5, color=PALETTE["text"], weight="bold")
        ax.text(-0.06, y - 0.16, f"{n_var}개 변수", va="center", ha="right",
                fontsize=10, color=PALETTE["subtext"])

        # 근거 태그 — 막대 아래 라운드폭 만큼, 옅은 pill
        tag_color = PALETTE["fraud"] if hot else PALETTE["subtext"]
        ax.add_patch(FancyBboxPatch(
            (0.02, y - bar_h / 2 - 0.30), 1.78, 0.30,
            boxstyle="round,pad=0.01,rounding_size=0.04",
            facecolor="white", edgecolor=tag_color,
            linewidth=1.0, zorder=2, clip_on=False))
        ax.text(0.07, y - bar_h / 2 - 0.15, reason,
                va="center", ha="left", fontsize=10.2,
                color=PALETTE["text"], zorder=4)

    ax.set_xlim(0, 2.0)
    ax.set_ylim(-0.7, n - 0.3)
    ax.set_yticks([])
    ax.set_xlabel("PR-AUC 개선 효과 (pp, baseline 대비)")
    ax.set_title("파생 변수 4라운드 — 효과와 추가 근거",
                 fontsize=14, color=PALETTE["text"], weight="bold", pad=14)
    ax.grid(axis="x", linestyle=":", color=PALETTE["grid"])
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)

    # 합계 캡션
    fig.text(0.97, 0.025, "변수 측면 누적 +3.10pp  (모델·튜닝 측면은 +1.22pp)",
             ha="right", fontsize=10.5, color=PALETTE["subtext"], weight="bold")

    save_fig(fname)
    plt.close(fig)


def main():
    plot_feature_rationale()
    print("saved → figures/feature_rationale.png")


if __name__ == "__main__":
    main()
