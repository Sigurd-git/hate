"""Create the method-design schematic for the manuscript."""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, PathPatch, Rectangle
from matplotlib.path import Path as MplPath


LOGGER = logging.getLogger(__name__)

OUTPUT_DIR = Path("artifacts/paper_revision")
HTML_OUTPUT_DIR = Path("slides/html")
FIGURE_BASENAME = "fig_methods_design_schematic"


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
            y_center + 0.13,
            title,
            ha="center",
            va="center",
            fontsize=title_size,
            color=title_color,
            fontweight="semibold",
        )
        axis.text(
            x_center,
            y_center - 0.15,
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

    sns.set_theme(style="white")
    figure, axis = plt.subplots(figsize=(15.4, 4.65), dpi=220)
    axis.set_xlim(0, 15.2)
    axis.set_ylim(0, 4.9)
    axis.axis("off")

    human_y = 3.45
    llm_y = 1.45
    corpus_center = (1.62, 2.45)
    corpus_width = 2.78
    corpus_height = 1.38

    add_box(
        axis,
        center=corpus_center,
        size=(corpus_width, corpus_height),
        title="371 minimal-pair corpus",
        detail="371 paired templates\nfemale + male versions\ndomain labels retained",
        facecolor=COLORS["corpus_fill"],
        edgecolor=COLORS["line"],
        title_color=COLORS["ink"],
        title_size=11.2,
        detail_size=7.9,
        linewidth=1.45,
        rounding=0.16,
    )

    box_specs = [
        ("assignment", 4.58, 2.46),
        ("sample", 7.22, 2.34),
        ("target", 9.96, 2.44),
        ("output", 12.98, 2.52),
    ]
    row_configs = [
        {
            "y": human_y,
            "color": COLORS["human"],
            "cells": [
                ("Human: one scale", "sessions randomly assigned\n3pt, 7pt, or slider", COLORS["human_fill"]),
                ("75 unique templates", "randomly sampled per session\nnot domain-stratified", COLORS["box"]),
                ("One target version", "female or male version\napproximately balanced", COLORS["box"]),
                ("75 trials", "trial-level ratings", COLORS["domain_fill"]),
            ],
        },
        {
            "y": llm_y,
            "color": COLORS["llm"],
            "cells": [
                ("LLM: all scales", "each model completed\n3pt, 7pt, and slider", COLORS["llm_fill"]),
                ("Full corpus", "371 templates per scale\nno subsampling", COLORS["box"]),
                ("Both target versions", "female and male versions\n742 sentences per scale", COLORS["box"]),
                ("Item ratings", "model-by-scale ratings", COLORS["domain_fill"]),
            ],
        },
    ]

    for row_config in row_configs:
        y_position = row_config["y"]
        row_color = row_config["color"]
        corpus_right_edge = corpus_center[0] + corpus_width / 2

        previous_right_edge: float | None = None
        for index, (title, detail, facecolor) in enumerate(row_config["cells"]):
            _, x_center, width = box_specs[index]
            height = 0.80
            left_edge = x_center - width / 2
            right_edge = x_center + width / 2

            add_box(
                axis,
                center=(x_center, y_position),
                size=(width, height),
                title=title,
                detail=detail,
                facecolor=facecolor,
                edgecolor=row_color,
                title_color=row_color,
                title_size=9.8,
                detail_size=7.7,
                linewidth=1.2,
                rounding=0.13,
            )

            if previous_right_edge is None:
                start_y = corpus_center[1] + 0.28 if y_position > corpus_center[1] else corpus_center[1] - 0.28
                add_arrow(
                    axis,
                    (corpus_right_edge + 0.03, start_y),
                    (left_edge - 0.05, y_position),
                    color=row_color,
                    linewidth=1.12,
                    connectionstyle="arc3,rad=0.10" if y_position > corpus_center[1] else "arc3,rad=-0.10",
                )
            else:
                add_arrow(axis, (previous_right_edge + 0.04, y_position), (left_edge - 0.04, y_position), color=row_color, linewidth=1.12)
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
