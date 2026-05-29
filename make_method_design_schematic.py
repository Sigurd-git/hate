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


def draw_schematic() -> plt.Figure:
    """Build and return the method-design schematic."""

    sns.set_theme(style="white")
    figure, axis = plt.subplots(figsize=(13.8, 7.6), dpi=220)
    axis.set_xlim(0, 14)
    axis.set_ylim(0, 8)
    axis.axis("off")

    # Shared material source.
    add_box(
        axis,
        center=(7.0, 7.15),
        size=(5.25, 0.78),
        title="371 minimal-pair templates",
        detail="Chinese attack templates; two target-gender versions; domain labels retained",
        facecolor=COLORS["corpus_fill"],
        title_size=15.2,
        detail_size=9.3,
        linewidth=1.55,
        rounding=0.18,
    )

    # Branch headers.
    human_x = 3.75
    llm_x = 10.25
    add_arrow(axis, (5.25, 6.76), (human_x, 6.35), color=COLORS["line"], linewidth=1.35, connectionstyle="angle3,angleA=180,angleB=90")
    add_arrow(axis, (8.75, 6.76), (llm_x, 6.35), color=COLORS["line"], linewidth=1.35, connectionstyle="angle3,angleA=0,angleB=90")

    add_human_icon(axis, (human_x, 5.78), scale=1.18)
    axis.text(human_x, 5.55, "Human sessions", ha="center", va="center", fontsize=13.6, color=COLORS["human"], fontweight="semibold")
    add_robot_icon(axis, (llm_x, 5.72), scale=1.22)
    axis.text(llm_x, 5.55, "Nine LLMs", ha="center", va="center", fontsize=13.6, color=COLORS["llm"], fontweight="semibold")

    # Human branch.
    add_box(
        axis,
        center=(human_x, 4.72),
        size=(3.8, 0.72),
        title="Balanced assignment to one response scale",
        detail="each participant used 3pt, 7pt, or slider for all trials",
        facecolor=COLORS["human_fill"],
        edgecolor=COLORS["human"],
        title_color=COLORS["human"],
    )
    human_scale_centers = add_scale_strip(axis, center=(human_x, 3.90), width=3.2, height=0.36)
    axis.text(human_x - 1.95, 4.02, "random\nassignment", ha="right", va="center", fontsize=8.8, color=COLORS["human"], linespacing=1.05)
    for scale_center in human_scale_centers:
        add_arrow(
            axis,
            (human_x, 4.36),
            (scale_center[0], scale_center[1] + 0.18),
            color=COLORS["human"],
            linestyle=(0, (4, 4)),
            linewidth=1.05,
            mutation_scale=11,
            zorder=5,
        )

    add_box(
        axis,
        center=(human_x, 2.92),
        size=(3.8, 0.78),
        title="75 unique templates sampled per session",
        detail="sampled from the full pool; not stratified by attack domain",
        facecolor=COLORS["box"],
        edgecolor=COLORS["human"],
        title_color=COLORS["human"],
    )
    add_box(
        axis,
        center=(human_x, 1.82),
        size=(3.8, 0.82),
        title="One target version per sampled template",
        detail="female or male version assigned at random;\ntarget counts approximately balanced",
        facecolor=COLORS["box"],
        edgecolor=COLORS["human"],
        title_color=COLORS["human"],
        detail_size=8.3,
    )
    add_box(
        axis,
        center=(human_x, 0.76),
        size=(3.8, 0.68),
        title="75 trial-level ratings",
        detail="response value + item id + target version + domain labels",
        facecolor=COLORS["domain_fill"],
        edgecolor=COLORS["human"],
        title_color=COLORS["human"],
        detail_size=8.2,
    )
    add_arrow(axis, (human_x, 3.72), (human_x, 3.33), color=COLORS["human"], linewidth=1.25)
    add_arrow(axis, (human_x, 2.53), (human_x, 2.23), color=COLORS["human"], linewidth=1.25)
    add_arrow(axis, (human_x, 1.41), (human_x, 1.10), color=COLORS["human"], linewidth=1.25)

    # LLM branch.
    add_box(
        axis,
        center=(llm_x, 4.72),
        size=(3.8, 0.72),
        title="Each model completed all response scales",
        detail="3pt, 7pt, and slider ratings were run for every LLM",
        facecolor=COLORS["llm_fill"],
        edgecolor=COLORS["llm"],
        title_color=COLORS["llm"],
    )
    llm_scale_centers = add_scale_strip(axis, center=(llm_x, 3.90), width=3.2, height=0.36)
    for scale_center in llm_scale_centers:
        add_arrow(axis, (llm_x, 4.36), (scale_center[0], scale_center[1] + 0.18), color=COLORS["llm"], linewidth=1.05, mutation_scale=11, zorder=5)

    add_box(
        axis,
        center=(llm_x, 2.92),
        size=(3.8, 0.80),
        title="Full corpus rated under each scale",
        detail="371 templates x 2 target-gender versions = 742 sentences",
        facecolor=COLORS["box"],
        edgecolor=COLORS["llm"],
        title_color=COLORS["llm"],
    )
    add_box(
        axis,
        center=(llm_x, 1.82),
        size=(3.8, 0.82),
        title="Both female and male versions observed",
        detail="no template subsampling; paired ratings available for every item",
        facecolor=COLORS["box"],
        edgecolor=COLORS["llm"],
        title_color=COLORS["llm"],
    )
    add_box(
        axis,
        center=(llm_x, 0.76),
        size=(3.8, 0.68),
        title="Model-by-scale item ratings",
        detail="paired by template, target version, model, and scale",
        facecolor=COLORS["domain_fill"],
        edgecolor=COLORS["llm"],
        title_color=COLORS["llm"],
        detail_size=8.2,
    )
    add_arrow(axis, (llm_x, 3.72), (llm_x, 3.33), color=COLORS["llm"], linewidth=1.25)
    add_arrow(axis, (llm_x, 2.52), (llm_x, 2.23), color=COLORS["llm"], linewidth=1.25)
    add_arrow(axis, (llm_x, 1.41), (llm_x, 1.10), color=COLORS["llm"], linewidth=1.25)

    # A subtle note connecting domains to the human branch.
    axis.text(
        7.0,
        0.22,
        "Attack domains are retained as item-level labels rather than forced to be balanced within each human session.",
        ha="center",
        va="center",
        fontsize=8.7,
        color=COLORS["muted"],
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
