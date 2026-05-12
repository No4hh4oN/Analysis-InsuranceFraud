"""
io_utils.py
-----------
프로젝트 공통 유틸:
  - 데이터 로딩 (UTF-16 한글 CSV)
  - matplotlib 한글 폰트 설정
  - 그림 저장 경로 헬퍼

전처리/모델링 스크립트에서 공통으로 import 해서 쓰기 위함.
"""

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl

# 프로젝트 루트 (이 파일 기준 한 단계 위)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT
FIG_DIR = PROJECT_ROOT / "figures"
OUT_DIR = PROJECT_ROOT / "outputs"
FIG_DIR.mkdir(exist_ok=True)
OUT_DIR.mkdir(exist_ok=True)


def setup_korean_font():
    """matplotlib 한글 폰트
      - Windows : "Malgun Gothic"  (OS 기본 내장)
      - macOS   : "AppleGothic"    (OS 기본 내장)
      - Linux   : "NanumGothic" / "Noto Sans CJK KR"  (별도 설치 필요)
    """
    import matplotlib.font_manager as fm

    candidates = [
        "Malgun Gothic",         # Windows
        "AppleGothic",           # macOS
        "NanumGothic",           # Linux / 공통 배포
        "Noto Sans CJK KR",
        "Noto Sans KR",
    ]
    available = {f.name for f in fm.fontManager.ttflist}
    chosen = next((c for c in candidates if c in available), None)

    if chosen is None:
        print("[font] 경고: 한글 폰트를 찾지 못함 — 한글이 깨질 수 있음.\n"
              "       Windows는 기본 'Malgun Gothic'이 있어야 하고,\n"
              "       Linux는 'sudo apt install fonts-nanum' 권장.")
        chosen = "DejaVu Sans"
    else:
        print(f"[font] 한글 폰트: {chosen}")

    mpl.rcParams["font.family"] = chosen
    # 한글 폰트 사용 시 마이너스(−) 기호가 깨지는 문제 방지
    mpl.rcParams["axes.unicode_minus"] = False
    # 슬라이드 캡처 가독성을 위한 기본값
    mpl.rcParams["figure.dpi"] = 110
    mpl.rcParams["savefig.dpi"] = 150
    mpl.rcParams["savefig.bbox"] = "tight"
    mpl.rcParams["font.size"] = 11


def load_cust() -> pd.DataFrame:
    """고객 테이블 로드 (CUST_DATA.csv, UTF-16 LE 인코딩)"""
    return pd.read_csv(DATA_DIR / "CUST_DATA.csv", encoding="utf-16")


def load_claim() -> pd.DataFrame:
    """청구 테이블 로드 (CLAIM_DATA.csv, UTF-16 LE 인코딩)"""
    return pd.read_csv(DATA_DIR / "CLAIM_DATA.csv", encoding="utf-16")


def save_fig(name: str):
    """현재 활성 figure를 figures/<name>.png 로 저장"""
    path = FIG_DIR / f"{name}.png"
    plt.savefig(path)
    plt.close()
    print(f"  saved → {path.relative_to(PROJECT_ROOT)}")


# ──────────────────────────────────────────────────────────────────────
# PPT 캡처용 공통 스타일 — 미니멀·고가독성 차트 만들기
# ──────────────────────────────────────────────────────────────────────

# 색상 팔레트 — 파스텔 알록달록 (PPT 시안용)
# 너무 옅어 안 보이지 않도록 충분히 채도 있는 파스텔만 사용.
PALETTE = {
    "fraud":   "#EF8E8D",   # 사기/위험 — 파스텔 코랄
    "normal":  "#B6C2D1",   # 정상 — 옅은 블루-그레이
    "C":       "#8FB4D8",   # 암(C) — 파스텔 블루
    "M":       "#EF8E8D",   # 근골격계(M) — 파스텔 코랄
    "S":       "#F5C18A",   # 손상(S) — 파스텔 피치
    "muted":   "#DDE3EA",   # 강조 안 하는 회색
    "text":    "#374151",   # 본문 텍스트
    "subtext": "#8B95A3",   # 부제·캡션
    "grid":    "#F1F5F9",   # 옅은 그리드
    "rule":    "#6B7280",   # 기준선
}

# 강조할 핵심 챕터 — 이 셋만 진한 색으로 칠하고 나머지는 옅은 블루그레이.
# 메시지: C(암) = 고액·저위험, M(근골격계)·S(손상) = 저액·고위험.
HIGHLIGHT_CHAPTERS = ("C", "M", "S")

# 챕터별 고유 파스텔 색상 — 너무 흐릿하지 않게 채도 충분히 유지.
# Fig 1 산점도/Fig 2 좌우 막대 등 챕터 단위 비교 그림에서 사용.
# (강조 외 챕터는 chap_color 헬퍼에서 muted/normal 색으로 덮어쓰므로
#  여기 색이 그대로 쓰이지 않을 수 있음 — 의미·시안 검토용으로 보관.)
CHAPTER_PALETTE = {
    "A": "#F5A6B8",   # 감염성·기생충 — 파스텔 핑크
    "B": "#F5A6B8",   # A와 묶임
    "C": "#8FB4D8",   # 신생물(암) — 파스텔 블루
    "D": "#E2C0B0",   # 혈액/면역 — 파스텔 베이지
    "E": "#E8B5B0",   # 내분비·대사 — 파스텔 더스티핑크
    "F": "#C9B5DC",   # 정신/행동 — 파스텔 모브
    "G": "#B5D89F",   # 신경계 — 파스텔 연그린
    "H": "#FFC7A0",   # 눈/귀 — 파스텔 살구
    "I": "#C7B6E0",   # 순환계 — 파스텔 라벤더
    "J": "#9ECEDD",   # 호흡계 — 파스텔 스카이
    "K": "#A9D5A9",   # 소화계 — 파스텔 그린
    "L": "#F5B5B5",   # 피부 — 파스텔 살몬
    "M": "#EF8E8D",   # 근골격계 — 파스텔 코랄
    "N": "#F4D58D",   # 비뇨생식기 — 파스텔 옐로
    "O": "#D4A5D9",   # 임신/출산 — 파스텔 라일락
    "P": "#B8E0D2",   # 주산기 — 파스텔 민트
    "Q": "#D8C8B8",   # 선천기형 — 파스텔 토프
    "R": "#A8D8D8",   # 증상/징후 — 파스텔 아쿠아
    "S": "#F5C18A",   # 손상/외상 — 파스텔 피치
    "T": "#EDC8A8",   # 손상/중독 — 파스텔 베이지오렌지
    "Z": "#C0D8E8",   # 보건접촉 — 파스텔 그레이블루
}


def apply_plot_style():
    """PPT 시안 수준의 통일된 스타일을 matplotlib rcParams 에 적용.

    - 미니멀: 상·우 spines 제거, 가로 그리드만
    - 큰 글씨: PPT 슬라이드에 작아 보이지 않도록
    - 톤다운된 텍스트 색상으로 시각적 위계 명확
    제목은 그림 안에 넣지 않고 PPT 슬라이드에서 직접 단다.
    """
    mpl.rcParams.update({
        "figure.dpi": 110,
        "savefig.dpi": 160,
        "savefig.bbox": "tight",
        "savefig.facecolor": "white",
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.edgecolor": PALETTE["subtext"],
        "axes.labelcolor": PALETTE["text"],
        "axes.labelsize": 12,
        "axes.labelpad": 8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.8,
        "axes.grid": True,
        "axes.grid.axis": "y",
        "grid.color": PALETTE["grid"],
        "grid.linewidth": 0.8,
        "grid.alpha": 1.0,
        "xtick.color": PALETTE["subtext"],
        "ytick.color": PALETTE["subtext"],
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "font.size": 12,
        "legend.frameon": False,
        "legend.fontsize": 10.5,
    })


def add_footnote(fig, text: str):
    """그림 하단에 옅은 회색 캡션(데이터 출처/주석)."""
    fig.text(0.01, -0.02, text, fontsize=9, color=PALETTE["subtext"], ha="left")
