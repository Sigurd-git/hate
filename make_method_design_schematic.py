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


def draw_schematic() -> plt.Figure:
    """Build and return the method-design schematic."""

    sns.set_theme(style="white")
    figure, axis = plt.subplots(figsize=(15.2, 5.1), dpi=220)
    axis.set_xlim(0, 16)
    axis.set_ylim(0, 5.6)
    axis.axis("off")

    # Shared material source.
    add_box(
        axis,
        center=(1.85, 3.05),
        size=(2.95, 1.38),
        title="371 minimal-pair corpus",
        detail="371 paired templates\n742 target-version sentences\ndomain labels retained",
        facecolor=COLORS["corpus_fill"],
        title_size=12.0,
        detail_size=8.3,
        linewidth=1.55,
        rounding=0.18,
    )

    # Human row.
    human_y = 4.22
    llm_y = 1.96
    add_arrow(axis, (3.33, 3.30), (4.02, human_y), color=COLORS["line"], linewidth=1.25, connectionstyle="arc3,rad=0.16")
    add_small_icon_label(axis, (4.35, human_y + 0.16), "Human sessions", "human")

    add_box(
        axis,
        center=(6.15, human_y),
        size=(2.36, 0.78),
        title="One response scale",
        detail="balanced assignment\n3pt, 7pt, or slider",
        facecolor=COLORS["human_fill"],
        edgecolor=COLORS["human"],
        title_color=COLORS["human"],
        title_size=10.2,
        detail_size=8.1,
    )
    add_arrow(axis, (4.70, human_y), (4.97, human_y), color=COLORS["human"], linestyle=(0, (4, 4)), linewidth=1.0)
    axis.text(5.38, human_y + 0.56, "random assignment", ha="center", va="center", fontsize=8.1, color=COLORS["human"])
    add_box(
        axis,
        center=(8.75, human_y),
        size=(2.36, 0.78),
        title="75 unique templates",
        detail="sampled per session\nnot domain-stratified",
        facecolor=COLORS["box"],
        edgecolor=COLORS["human"],
        title_color=COLORS["human"],
        title_size=10.2,
        detail_size=8.1,
    )
    add_box(
        axis,
        center=(11.35, human_y),
        size=(2.38, 0.78),
        title="One target version",
        detail="female or male version\nrandom, approximately balanced",
        facecolor=COLORS["box"],
        edgecolor=COLORS["human"],
        title_color=COLORS["human"],
        title_size=10.2,
        detail_size=7.9,
    )
    add_box(
        axis,
        center=(14.10, human_y),
        size=(2.00, 0.74),
        title="75 trials",
        detail="trial-level data",
        facecolor=COLORS["domain_fill"],
        edgecolor=COLORS["human"],
        title_color=COLORS["human"],
        title_size=9.5,
        detail_size=7.6,
    )
    add_arrow(axis, (7.33, human_y), (7.57, human_y), color=COLORS["human"], linewidth=1.15)
    add_arrow(axis, (9.93, human_y), (10.16, human_y), color=COLORS["human"], linewidth=1.15)
    add_arrow(axis, (12.54, human_y), (13.10, human_y), color=COLORS["human"], linewidth=1.15)

    # LLM row.
    add_arrow(axis, (3.33, 2.82), (4.02, llm_y), color=COLORS["line"], linewidth=1.25, connectionstyle="arc3,rad=-0.16")
    add_small_icon_label(axis, (4.35, llm_y + 0.16), "Nine LLMs", "llm")
    add_box(
        axis,
        center=(6.15, llm_y),
        size=(2.36, 0.78),
        title="All response scales",
        detail="each model completed\n3pt, 7pt, and slider",
        facecolor=COLORS["llm_fill"],
        edgecolor=COLORS["llm"],
        title_color=COLORS["llm"],
        title_size=10.2,
        detail_size=8.1,
    )
    add_box(
        axis,
        center=(8.75, llm_y),
        size=(2.36, 0.78),
        title="Full corpus per scale",
        detail="371 templates x 2 versions\n= 742 sentences",
        facecolor=COLORS["box"],
        edgecolor=COLORS["llm"],
        title_color=COLORS["llm"],
        title_size=10.2,
        detail_size=8.1,
    )
    add_box(
        axis,
        center=(11.35, llm_y),
        size=(2.38, 0.78),
        title="Both target versions",
        detail="female and male ratings\navailable for every item",
        facecolor=COLORS["box"],
        edgecolor=COLORS["llm"],
        title_color=COLORS["llm"],
        title_size=10.2,
        detail_size=7.9,
    )
    add_box(
        axis,
        center=(14.10, llm_y),
        size=(2.00, 0.74),
        title="Item ratings",
        detail="model-by-scale",
        facecolor=COLORS["domain_fill"],
        edgecolor=COLORS["llm"],
        title_color=COLORS["llm"],
        title_size=9.5,
        detail_size=7.6,
    )
    add_arrow(axis, (7.33, llm_y), (7.57, llm_y), color=COLORS["llm"], linewidth=1.15)
    add_arrow(axis, (9.93, llm_y), (10.16, llm_y), color=COLORS["llm"], linewidth=1.15)
    add_arrow(axis, (12.54, llm_y), (13.10, llm_y), color=COLORS["llm"], linewidth=1.15)

    # A subtle note connecting domains to the human branch.
    axis.text(
        8.35,
        0.62,
        "Attack domains retained as item-level labels.",
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
