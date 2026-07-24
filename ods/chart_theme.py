"""Shared chart palette so every figure matches the app's green/cream theme."""

from __future__ import annotations

import plotly.graph_objects as go

ACCENT = "#15803d"
ACCENT_SOFT = "#1f9a4c"
AREA_FILL = "rgba(21, 128, 61, .22)"
PAPER = "#fffdf6"
INK = "#23392b"
GRID = "rgba(20, 83, 45, .12)"
GRID_STRONG = "rgba(20, 83, 45, .16)"
HOVER_BG = "#16281e"
HOVER_FG = "#f7f3e7"

# Slice/series colors for categorical charts (pies, grouped boxes): greens
# first, then warm creams and golds so neighboring slices stay distinct.
CATEGORICAL = (
    "#15803d",
    "#8fbf9f",
    "#0d5c2b",
    "#c9b671",
    "#54a06d",
    "#7d8a6a",
    "#2f4c3a",
    "#e0d3a4",
)


def style_figure(
    figure: go.Figure,
    *,
    x_title: str | None = None,
    y_title: str | None = None,
    height: int = 340,
    show_legend: bool = False,
) -> go.Figure:
    """Apply the shared layout: cream paper, green grid, quiet axes."""
    figure.update_layout(
        title={"text": ""},
        height=height,
        margin={"l": 48, "r": 18, "t": 18, "b": 56},
        paper_bgcolor=PAPER,
        plot_bgcolor=PAPER,
        font={"color": INK},
        hoverlabel={"bgcolor": HOVER_BG, "font_color": HOVER_FG},
        showlegend=show_legend,
        legend={"orientation": "h", "yanchor": "bottom", "y": -0.35},
    )
    if x_title is not None:
        figure.update_xaxes(title=x_title, gridcolor=GRID, automargin=True)
    if y_title is not None:
        figure.update_yaxes(title=y_title, gridcolor=GRID_STRONG, automargin=True)
    return figure
