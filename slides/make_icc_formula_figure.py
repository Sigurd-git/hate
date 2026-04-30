"""Render annotated ICC(2,1) / ICC(3,1) formulas for the human-model delta study.

Each classical symbol (MS_R, MS_E, MS_C, n, k) is mapped to its concrete
meaning in this thesis: items are the 371 sentence-pair templates yielding
paired ΔF-M scores; raters are the human-aggregate vector and one LLM vector.
Output: an annotated PNG placed next to the presentation assets.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import font_manager

# Pick a Chinese-capable font for all non-math text; mathtext handles formulas.
CHINESE_FONT_CANDIDATES = [
    "Noto Sans CJK SC",
    "Noto Serif CJK SC",
    "AR PL UMing CN",
]


def resolve_chinese_font() -> str:
    """Return the first installed Chinese font from the candidate list."""
    available = {f.name for f in font_manager.fontManager.ttflist}
    for name in CHINESE_FONT_CANDIDATES:
        if name in available:
            return name
    return "DejaVu Sans"


def main() -> None:
    chinese_font = resolve_chinese_font()
    mpl.rcParams["font.family"] = chinese_font
    mpl.rcParams["axes.unicode_minus"] = False
    mpl.rcParams["mathtext.fontset"] = "cm"  # classic LaTeX math look

    fig, ax = plt.subplots(figsize=(13.0, 9.0), dpi=220)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    # --- Title ---------------------------------------------------------------
    ax.text(
        0.5,
        0.965,
        "ICC(2,1) 与 ICC(3,1) 在本研究里的具体含义",
        ha="center",
        va="top",
        fontsize=19,
        fontweight="bold",
    )
    ax.text(
        0.5,
        0.915,
        "单位：每个"
        "「评分尺度 × LLM」"
        "单元都在 n 个题项 × 2 个评分者上计算一次",
        ha="center",
        va="top",
        fontsize=12.5,
        color="#444444",
    )

    # --- ICC(2,1) formula ----------------------------------------------------
    ax.text(
        0.04,
        0.80,
        "ICC(2,1)  绝对一致（two-way random,single rater,absolute agreement）",
        ha="left",
        va="center",
        fontsize=14,
        fontweight="bold",
        color="#1f4e79",
    )
    ax.text(
        0.5,
        0.705,
        r"$\mathrm{ICC}(2,1) \;=\; "
        r"\frac{\mathrm{MS}_R - \mathrm{MS}_E}"
        r"{\mathrm{MS}_R + (k-1)\,\mathrm{MS}_E"
        r" + \frac{k}{n}\,(\mathrm{MS}_C - \mathrm{MS}_E)}$",
        ha="center",
        va="center",
        fontsize=22,
    )

    # --- ICC(3,1) formula ----------------------------------------------------
    ax.text(
        0.04,
        0.555,
        "ICC(3,1)  一致性（two-way mixed,single rater,consistency）",
        ha="left",
        va="center",
        fontsize=14,
        fontweight="bold",
        color="#7a3b10",
    )
    ax.text(
        0.5,
        0.465,
        r"$\mathrm{ICC}(3,1) \;=\; "
        r"\frac{\mathrm{MS}_R - \mathrm{MS}_E}"
        r"{\mathrm{MS}_R + (k-1)\,\mathrm{MS}_E}$",
        ha="center",
        va="center",
        fontsize=22,
    )

    # --- Symbol legend -------------------------------------------------------
    ax.text(
        0.04,
        0.345,
        "在本研究里，每个符号对应的东西：",
        ha="left",
        va="center",
        fontsize=13.5,
        fontweight="bold",
    )

    # Column layout for the glossary. Each row: (symbol, plain-language meaning)
    glossary_rows = [
        (r"$n$", "题项数——该单元下参与配对的句对模板数量 (约 371 对)"),
        (r"$k$", "评分者数 = 2（人类被试聚合向量 vs. 单个 LLM）"),
        (
            r"$\mathrm{MS}_R$",
            "题项间方差：不同句对模板在 ΔF-M 上的分散程度（『真分数』部分）",
        ),
        (
            r"$\mathrm{MS}_C$",
            "评分者间方差：人类均值 ΔF-M 与 LLM 均值 ΔF-M 的系统性位移（偏置）",
        ),
        (
            r"$\mathrm{MS}_E$",
            "残差方差：题项 × 评分者交互，即扣掉题项效应与评分者均值后的剩余不一致",
        ),
    ]

    y0 = 0.285
    row_gap = 0.045
    for i, (symbol, meaning) in enumerate(glossary_rows):
        y = y0 - i * row_gap
        ax.text(0.07, y, symbol, ha="left", va="center", fontsize=14)
        ax.text(0.175, y, meaning, ha="left", va="center", fontsize=12.5)

    # --- Bottom note: the difference between the two ICCs --------------------
    note_y = 0.045
    ax.text(
        0.5,
        note_y + 0.018,
        r"差别就在 $\frac{k}{n}(\mathrm{MS}_C - \mathrm{MS}_E)$ 这一项："
        "ICC(2,1) 会『惩罚』人类与 LLM 的均值偏置，ICC(3,1) 把它视为常数而忽略。",
        ha="center",
        va="center",
        fontsize=12,
        color="#222222",
    )
    ax.text(
        0.5,
        note_y - 0.022,
        "→ 所以 ICC(2,1) 回答『人和模型的 ΔF-M 曲线能不能直接对齐』，"
        "ICC(3,1) 回答『两者的起伏模式是否一致（忽略整体高低）』。",
        ha="center",
        va="center",
        fontsize=12,
        color="#222222",
    )

    fig.tight_layout()

    output_path = Path(__file__).parent / "icc_formula_with_study_mapping.png"
    fig.savefig(output_path, dpi=220, bbox_inches="tight", facecolor="white")
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
