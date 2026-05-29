"""Create a method-design schematic for the paper.

The figure mirrors the GPT-image visual direction but keeps the arrow geometry
deterministic: the LLM branch points to each response-scale cell center.
"""

from __future__ import annotations

import logging
from pathlib import Path as FilePath

import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, PathPatch, Rectangle
from matplotlib.path import Path as MplPath


LOGGER = logging.getLogger(__name__)

OUTPUT_DIR = FilePath("artifacts/paper_revision")
HTML_OUTPUT_DIR = FilePath("slides/html")
FIGURE_BASENAME = "fig_methods_design_schematic"


COLORS = {
    "ink": "#202833",
    "human": "#0B2A5B",
    "llm": "#0B6B35",
    "scale_fill": "#F3F6FA",
    "grid": "#202833",
}


def add_arrow(
    axis: plt.Axes,
    start: tuple[float, float],
    end: tuple[float, float],
    *,
    color: str,
    linestyle: str = "-",
    linewidth: float = 1.35,
    mutation_scale: float = 16,
    connectionstyle: str = "arc3,rad=0",
    zorder: int = 5,
) -> None:
    """Draw a single arrow with a visible head at the requested endpoint."""

    arrow = FancyArrowPatch(
        start,
        end,
        arrowstyle="-|>",
        mutation_scale=mutation_scale,
        linewidth=linewidth,
        linestyle=linestyle,
        color=color,
        shrinkA=0,
        shrinkB=0,
        connectionstyle=connectionstyle,
        zorder=zorder,
    )
    axis.add_patch(arrow)


def add_human_icon(axis: plt.Axes, center_x: float, base_y: float) -> None:
    """Draw a simple line human icon."""

    head = Circle((center_x, base_y + 0.72), 0.22, fill=False, linewidth=2.0, edgecolor=COLORS["human"])
    neck = Rectangle((center_x - 0.07, base_y + 0.43), 0.14, 0.15, fill=False, linewidth=1.8, edgecolor=COLORS["human"])

    shoulder_path = MplPath(
        [
            (center_x - 0.48, base_y + 0.02),
            (center_x - 0.48, base_y + 0.35),
            (center_x - 0.22, base_y + 0.43),
            (center_x, base_y + 0.43),
            (center_x + 0.22, base_y + 0.43),
            (center_x + 0.48, base_y + 0.35),
            (center_x + 0.48, base_y + 0.02),
        ],
        [
            MplPath.MOVETO,
            MplPath.CURVE4,
            MplPath.CURVE4,
            MplPath.CURVE4,
            MplPath.CURVE4,
            MplPath.CURVE4,
            MplPath.CURVE4,
        ],
    )
    shoulder = PathPatch(shoulder_path, fill=False, linewidth=2.0, edgecolor=COLORS["human"], capstyle="round")

    axis.add_patch(head)
    axis.add_patch(neck)
    axis.add_patch(shoulder)


def add_robot_icon(axis: plt.Axes, center_x: float, base_y: float) -> None:
    """Draw a simple line robot icon."""

    body = FancyBboxPatch(
        (center_x - 0.38, base_y + 0.12),
        0.76,
        0.58,
        boxstyle="round,pad=0.02,rounding_size=0.13",
        fill=False,
        linewidth=2.0,
        edgecolor=COLORS["llm"],
    )
    left_ear = FancyBboxPatch(
        (center_x - 0.52, base_y + 0.29),
        0.12,
        0.22,
        boxstyle="round,pad=0.01,rounding_size=0.04",
        fill=False,
        linewidth=1.8,
        edgecolor=COLORS["llm"],
    )
    right_ear = FancyBboxPatch(
        (center_x + 0.40, base_y + 0.29),
        0.12,
        0.22,
        boxstyle="round,pad=0.01,rounding_size=0.04",
        fill=False,
        linewidth=1.8,
        edgecolor=COLORS["llm"],
    )
    axis.add_patch(body)
    axis.add_patch(left_ear)
    axis.add_patch(right_ear)
    axis.add_patch(Circle((center_x - 0.16, base_y + 0.43), 0.055, fill=False, linewidth=1.9, edgecolor=COLORS["llm"]))
    axis.add_patch(Circle((center_x + 0.16, base_y + 0.43), 0.055, fill=False, linewidth=1.9, edgecolor=COLORS["llm"]))
    axis.plot([center_x - 0.10, center_x + 0.10], [base_y + 0.26, base_y + 0.26], color=COLORS["llm"], lw=2.0)
    axis.plot([center_x, center_x], [base_y + 0.70, base_y + 0.86], color=COLORS["llm"], lw=1.9)
    axis.add_patch(Circle((center_x, base_y + 0.92), 0.06, fill=False, linewidth=1.9, edgecolor=COLORS["llm"]))


def draw_schematic() -> plt.Figure:
    """Build and return the method-design schematic."""

    sns.set_theme(style="white")
    figure, axis = plt.subplots(figsize=(13.5, 7.0), dpi=220)
    axis.set_xlim(0, 14)
    axis.set_ylim(0, 7)
    axis.axis("off")

    # Top corpus block.
    corpus_box = FancyBboxPatch(
        (4.7, 5.82),
        4.6,
        0.72,
        boxstyle="round,pad=0.06,rounding_size=0.25",
        linewidth=1.8,
        edgecolor=COLORS["ink"],
        facecolor="#FFFFFF",
        zorder=2,
    )
    axis.add_patch(corpus_box)
    axis.text(
        7.0,
        6.18,
        "371 minimal-pair corpus",
        ha="center",
        va="center",
        fontsize=20,
        color="#050505",
        fontweight="medium",
    )

    # Branch arrows from corpus to rater types.
    human_x = 3.775
    robot_x = 10.475
    add_arrow(axis, (4.7, 6.18), (human_x, 4.95), color=COLORS["ink"], linewidth=1.6, connectionstyle="angle3,angleA=180,angleB=90")
    add_arrow(axis, (9.3, 6.18), (robot_x, 4.95), color=COLORS["ink"], linewidth=1.6, connectionstyle="angle3,angleA=0,angleB=90")

    add_human_icon(axis, human_x, 3.75)
    axis.text(human_x, 3.67, "Human", ha="center", va="top", fontsize=16.5, color=COLORS["human"], fontweight="medium")

    add_robot_icon(axis, robot_x, 3.78)
    axis.text(robot_x, 3.67, "LLM", ha="center", va="top", fontsize=16.5, color=COLORS["llm"], fontweight="medium")

    # Matrix.
    table_x0 = 2.1
    table_y0 = 0.90
    cell_width = 3.35
    cell_height = 0.98
    table_width = cell_width * 3
    table_height = cell_height

    axis.add_patch(Rectangle((table_x0, table_y0), table_width, table_height, facecolor=COLORS["scale_fill"], edgecolor="none", zorder=0))

    for column_index in range(4):
        x_position = table_x0 + column_index * cell_width
        axis.plot([x_position, x_position], [table_y0, table_y0 + table_height], color=COLORS["grid"], lw=1.15, zorder=1)
    for row_index in range(2):
        y_position = table_y0 + row_index * table_height
        axis.plot([table_x0, table_x0 + table_width], [y_position, y_position], color=COLORS["grid"], lw=1.15, zorder=1)

    for label, x_position in zip(["3pt", "7pt", "Slider"], [table_x0 + 0.5 * cell_width, table_x0 + 1.5 * cell_width, table_x0 + 2.5 * cell_width], strict=True):
        axis.text(x_position, table_y0 - 0.34, label, ha="center", va="center", fontsize=16.5, color="#050505")

    scale_centers = [(table_x0 + (index + 0.5) * cell_width, table_y0 + 0.5 * cell_height) for index in range(3)]

    # Human participants are randomly assigned to one response scale.
    selected_human_cell = scale_centers[0]
    human_path = MplPath(
        [
            (human_x, 3.27),
            selected_human_cell,
        ],
        [MplPath.MOVETO, MplPath.LINETO],
    )
    human_arrow = FancyArrowPatch(
        path=human_path,
        arrowstyle="-|>",
        mutation_scale=16,
        linewidth=1.35,
        linestyle=(0, (4, 4)),
        color=COLORS["human"],
        zorder=6,
    )
    axis.add_patch(human_arrow)
    axis.text(human_x - 0.35, 2.28, "random assignment", ha="right", va="center", fontsize=12.8, color=COLORS["human"])

    # LLM rates all response scales. Endpoints are exact centers.
    llm_origin = (robot_x, 3.42)
    for end in scale_centers:
        add_arrow(
            axis,
            llm_origin,
            end,
            color=COLORS["llm"],
            linewidth=1.22,
            mutation_scale=15,
            zorder=5,
        )

    figure.tight_layout(pad=0.35)
    return figure


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    HTML_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    figure = draw_schematic()
    pdf_path = OUTPUT_DIR / f"{FIGURE_BASENAME}.pdf"
    png_path = OUTPUT_DIR / f"{FIGURE_BASENAME}.png"
    html_png_path = HTML_OUTPUT_DIR / f"{FIGURE_BASENAME}.png"

    figure.savefig(pdf_path, bbox_inches="tight")
    figure.savefig(png_path, bbox_inches="tight")
    figure.savefig(html_png_path, bbox_inches="tight")
    plt.close(figure)

    LOGGER.info("Saved %s", pdf_path)
    LOGGER.info("Saved %s", png_path)
    LOGGER.info("Saved %s", html_png_path)


if __name__ == "__main__":
    main()
