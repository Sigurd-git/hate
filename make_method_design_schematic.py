"""Create the method-design schematic for the manuscript."""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, PathPatch, Rectangle
from matplotlib.path import Path as MplPath


LOGGER = logging.getLogger(__name__)

OUTPUT_DIR = Path("artifacts/paper_revision")
HTML_OUTPUT_DIR = Path("slides/html")
FIGURE_BASENAME = "fig_methods_design_schematic"
# cas-sc review mode reports \textwidth = 468.3324 pt = 6.48 in.
LATEX_LINEWIDTH_IN = 6.48
LATEX_BODY_FONT_PT = 10.0
SANS_SERIF_VISUAL_MATCH = 0.88


COLORS = {
    "ink": "#202833",
    "muted": "#5F6B7A",
    "human": "#0B2A5B",
    "llm": "#0B6B35",
    "line": "#26313D",
    "box": "#FFFFFF",
    "corpus_fill": "#F7F8FA",
    "human_fill": "#EEF4FF",
    "llm_fill": "#EEF8F2",
    "scale_fill": "#F4F6F8",
    "domain_fill": "#FFF7EA",
}


def source_font_size(figure_width_in: float) -> float:
    """Return source font size that renders near body size after LaTeX scaling."""

    return LATEX_BODY_FONT_PT * figure_width_in / LATEX_LINEWIDTH_IN


def visual_body_font_size(figure_width_in: float) -> float:
    """Match sans-serif figure text to the visual size of the serif body text."""

    return source_font_size(figure_width_in) * SANS_SERIF_VISUAL_MATCH


def configure_plotting() -> None:
    sns.set_theme(style="white")
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Noto Sans CJK SC", "Noto Sans CJK JP", "DejaVu Sans"],
            "mathtext.fontset": "dejavusans",
            "mathtext.default": "regular",
            "mathtext.rm": "DejaVu Sans",
            "mathtext.it": "DejaVu Sans:italic",
            "mathtext.bf": "DejaVu Sans:bold",
            "axes.unicode_minus": False,
        }
    )


def add_arrow(
    axis: plt.Axes,
    start: tuple[float, float],
    end: tuple[float, float],
    *,
    color: str = COLORS["line"],
    linestyle: str = "-",
    linewidth: float = 1.25,
    mutation_scale: float = 14,
    connectionstyle: str = "arc3,rad=0",
    zorder: int = 4,
) -> None:
    """Draw an arrow between two points."""

    axis.add_patch(
        FancyArrowPatch(
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
    )


def add_box(
    axis: plt.Axes,
    center: tuple[float, float],
    size: tuple[float, float],
    title: str,
    detail: str | None = None,
    *,
    facecolor: str = COLORS["box"],
    edgecolor: str = COLORS["line"],
    title_color: str = COLORS["ink"],
    detail_color: str = COLORS["muted"],
    title_size: float = 11.0,
    detail_size: float = 8.7,
    linewidth: float = 1.15,
    rounding: float = 0.12,
) -> None:
    """Draw a rounded labelled box."""

    x_center, y_center = center
    width, height = size
    box = FancyBboxPatch(
        (x_center - width / 2, y_center - height / 2),
        width,
        height,
        boxstyle=f"round,pad=0.03,rounding_size={rounding}",
        facecolor=facecolor,
        edgecolor=edgecolor,
        linewidth=linewidth,
        zorder=2,
    )
    axis.add_patch(box)
    if detail:
        axis.text(
            x_center,
            y_center + 0.22,
            title,
            ha="center",
            va="center",
            fontsize=title_size,
            color=title_color,
            fontweight="semibold",
        )
        axis.text(
            x_center,
            y_center - 0.22,
            detail,
            ha="center",
            va="center",
            fontsize=detail_size,
            color=detail_color,
            linespacing=1.25,
        )
    else:
        axis.text(
            x_center,
            y_center,
            title,
            ha="center",
            va="center",
            fontsize=title_size,
            color=title_color,
            fontweight="semibold",
            linespacing=1.2,
        )


def add_human_icon(axis: plt.Axes, center: tuple[float, float], scale: float = 1.0) -> None:
    """Draw a simple line human icon."""

    center_x, base_y = center
    head_radius = 0.14 * scale
    head = Circle((center_x, base_y + 0.34 * scale), head_radius, fill=False, linewidth=1.8, edgecolor=COLORS["human"])
    neck = Rectangle((center_x - 0.045 * scale, base_y + 0.18 * scale), 0.09 * scale, 0.08 * scale, fill=False, linewidth=1.6, edgecolor=COLORS["human"])
    shoulder_path = MplPath(
        [
            (center_x - 0.28 * scale, base_y),
            (center_x - 0.28 * scale, base_y + 0.18 * scale),
            (center_x - 0.13 * scale, base_y + 0.20 * scale),
            (center_x, base_y + 0.20 * scale),
            (center_x + 0.13 * scale, base_y + 0.20 * scale),
            (center_x + 0.28 * scale, base_y + 0.18 * scale),
            (center_x + 0.28 * scale, base_y),
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
    axis.add_patch(head)
    axis.add_patch(neck)
    axis.add_patch(PathPatch(shoulder_path, fill=False, linewidth=1.8, edgecolor=COLORS["human"], capstyle="round"))


def add_robot_icon(axis: plt.Axes, center: tuple[float, float], scale: float = 1.0) -> None:
    """Draw a simple line robot icon."""

    center_x, base_y = center
    body_width = 0.50 * scale
    body_height = 0.38 * scale
    body = FancyBboxPatch(
        (center_x - body_width / 2, base_y + 0.08 * scale),
        body_width,
        body_height,
        boxstyle=f"round,pad=0.01,rounding_size={0.08 * scale}",
        fill=False,
        linewidth=1.8,
        edgecolor=COLORS["llm"],
    )
    axis.add_patch(body)
    axis.add_patch(Circle((center_x - 0.10 * scale, base_y + 0.28 * scale), 0.035 * scale, fill=False, linewidth=1.7, edgecolor=COLORS["llm"]))
    axis.add_patch(Circle((center_x + 0.10 * scale, base_y + 0.28 * scale), 0.035 * scale, fill=False, linewidth=1.7, edgecolor=COLORS["llm"]))
    axis.plot([center_x - 0.07 * scale, center_x + 0.07 * scale], [base_y + 0.17 * scale, base_y + 0.17 * scale], color=COLORS["llm"], lw=1.8)
    axis.plot([center_x, center_x], [base_y + 0.46 * scale, base_y + 0.58 * scale], color=COLORS["llm"], lw=1.7)
    axis.add_patch(Circle((center_x, base_y + 0.63 * scale), 0.04 * scale, fill=False, linewidth=1.7, edgecolor=COLORS["llm"]))


def add_scale_strip(axis: plt.Axes, center: tuple[float, float], width: float, height: float) -> list[tuple[float, float]]:
    """Draw three response-scale cells and return their centers."""

    x_center, y_center = center
    x_left = x_center - width / 2
    cell_width = width / 3
    centers: list[tuple[float, float]] = []
    labels = ["3pt", "7pt", "Slider"]
    for index, label in enumerate(labels):
        x = x_left + index * cell_width
        axis.add_patch(
            Rectangle(
                (x, y_center - height / 2),
                cell_width,
                height,
                facecolor=COLORS["scale_fill"],
                edgecolor=COLORS["line"],
                linewidth=1.0,
                zorder=1,
            )
        )
        cell_center = (x + cell_width / 2, y_center)
        centers.append(cell_center)
        axis.text(cell_center[0], y_center, label, ha="center", va="center", fontsize=9.3, color=COLORS["ink"])
    return centers


def add_small_icon_label(axis: plt.Axes, center: tuple[float, float], label: str, kind: str) -> None:
    """Draw a compact rater icon with a label beneath it."""

    icon_x, icon_y = center
    if kind == "human":
        add_human_icon(axis, (icon_x, icon_y - 0.06), scale=0.66)
        color = COLORS["human"]
    else:
        add_robot_icon(axis, (icon_x, icon_y - 0.12), scale=0.70)
        color = COLORS["llm"]
    axis.text(icon_x, icon_y - 0.64, label, ha="center", va="center", fontsize=10.5, color=color, fontweight="semibold")


def add_rater_badge(axis: plt.Axes, center: tuple[float, float], title: str, detail: str, kind: str) -> None:
    """Draw a compact rater badge at the start of a flow row."""

    x_center, y_center = center
    if kind == "human":
        color = COLORS["human"]
        facecolor = "#F8FBFF"
    else:
        color = COLORS["llm"]
        facecolor = "#F8FCF9"

    width = 1.52
    height = 0.72
    badge = FancyBboxPatch(
        (x_center - width / 2, y_center - height / 2),
        width,
        height,
        boxstyle="round,pad=0.03,rounding_size=0.12",
        facecolor=facecolor,
        edgecolor=color,
        linewidth=1.15,
        zorder=2,
    )
    axis.add_patch(badge)

    axis.text(x_center, y_center + 0.09, title, ha="center", va="center", fontsize=9.8, color=color, fontweight="semibold")
    axis.text(x_center, y_center - 0.14, detail, ha="center", va="center", fontsize=7.7, color=COLORS["muted"])


def draw_schematic() -> plt.Figure:
    """Build and return the method-design schematic."""

    configure_plotting()
    figure_width = 15.4
    font_size = visual_body_font_size(figure_width)
    figure, axis = plt.subplots(figsize=(figure_width, 5.25), dpi=220)
    axis.set_xlim(0, 15.2)
    axis.set_ylim(0, 5.35)
    axis.axis("off")

    human_y = 3.62
    llm_y = 1.62
    corpus_center = (1.62, 2.62)
    corpus_width = 2.92
    corpus_height = 1.62

    add_box(
        axis,
        center=corpus_center,
        size=(corpus_width, corpus_height),
        title="371 matched pairs\nfemale + male targets\n10 attack domains",
        detail=None,
        facecolor=COLORS["corpus_fill"],
        edgecolor=COLORS["line"],
        title_color=COLORS["ink"],
        title_size=font_size * 0.82,
        linewidth=1.55,
        rounding=0.16,
    )

    box_specs = [
        (4.84, 2.80),
        (8.35, 2.95),
        (12.00, 3.08),
    ]
    row_configs = [
        {
            "y": human_y,
            "color": COLORS["human"],
            "cells": [
                ("779 human raters\none scale\none target version", COLORS["human_fill"]),
                ("Aggressiveness\nratings\n3pt / 7pt / slider", COLORS["box"]),
                ("Human F-M gaps\noverall / female\n/ male", COLORS["domain_fill"]),
            ],
        },
        {
            "y": llm_y,
            "color": COLORS["llm"],
            "cells": [
                ("Nine LLMs\nall scales\nboth target versions", COLORS["llm_fill"]),
                ("Aggressiveness\nratings\nsame scale formats", COLORS["box"]),
                ("Model comparison\naggregate / domain\n/ item", COLORS["domain_fill"]),
            ],
        },
    ]

    corpus_right_edge = corpus_center[0] + corpus_width / 2
    first_box_left_edge = box_specs[0][0] - box_specs[0][1] / 2
    branch_x = corpus_right_edge + (first_box_left_edge - corpus_right_edge) * 0.38
    axis.plot([corpus_right_edge + 0.02, branch_x], [corpus_center[1], corpus_center[1]], color=COLORS["line"], lw=1.1, zorder=3)
    axis.plot([branch_x, branch_x], [llm_y, human_y], color=COLORS["line"], lw=1.1, zorder=3)

    for row_config in row_configs:
        y_position = row_config["y"]
        row_color = row_config["color"]

        previous_right_edge: float | None = None
        for index, (title, facecolor) in enumerate(row_config["cells"]):
            x_center, width = box_specs[index]
            height = 1.34
            left_edge = x_center - width / 2
            right_edge = x_center + width / 2

            add_box(
                axis,
                center=(x_center, y_position),
                size=(width, height),
                title=title,
                detail=None,
                facecolor=facecolor,
                edgecolor=row_color,
                title_color=row_color,
                title_size=font_size * 0.82,
                linewidth=1.3,
                rounding=0.13,
            )

            if previous_right_edge is None:
                add_arrow(axis, (branch_x, y_position), (left_edge - 0.05, y_position), color=row_color, linewidth=1.25, mutation_scale=16)
            else:
                add_arrow(axis, (previous_right_edge + 0.04, y_position), (left_edge - 0.04, y_position), color=row_color, linewidth=1.25, mutation_scale=16)
            previous_right_edge = right_edge

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
