"""Plot and tree rendering for the NiceGUI explorer."""

from __future__ import annotations

import io

import numpy as np
import plotly.graph_objects as go
from matplotlib import colormaps
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.colors import Colormap, LinearSegmentedColormap, to_hex
from matplotlib.figure import Figure
from sklearn.decomposition import PCA

from ..plot_tree_patch import plot_tree_patched
from ..utilities import colormap_from_str, custom_axes_limit, rule_print_inline
from .models import InteractPlot, InteractPoint

SIZE_CANDIDATE = 9.0
SIZE_SELECTED = 15.0
PLOT_WIDTH = 620
PLOT_HEIGHT = 500


def _feature_names_for_clf(clf) -> list[str]:
    if hasattr(clf, "feature_names_in_") and clf.feature_names_in_ is not None:
        return [str(name) for name in clf.feature_names_in_]
    return [f"X{i}" for i in range(clf.n_features_in_)]


def print_tree_rule(tree_index: str, render_context: dict) -> None:
    """Print the selected rule to preserve the existing console feedback."""
    clf = render_context["clf"]
    sample = render_context["sample"]
    rule_print_inline(clf[int(tree_index)], sample)


def render_tree_image(tree_index: str, render_context: dict) -> tuple[bytes, str]:
    """Render one selected tree as PNG bytes and return its display title."""
    clf = render_context["clf"]
    sample_index = render_context.get("sample_index")
    if sample_index is None:
        sample_index = render_context["sample"].index[0]
    max_depth = render_context.get("max_depth")
    feature_names = render_context.get("feature_names") or _feature_names_for_clf(clf)

    numeric_tree_index = int(tree_index)
    tree = clf[numeric_tree_index]

    if max_depth is None:
        plot_leaves = tree.tree_.n_leaves
        plot_depth = tree.tree_.max_depth
    else:
        plot_leaves = max(tree.tree_.n_leaves, 2 ** (max_depth - 1))
        plot_depth = min(tree.tree_.max_depth, max_depth)

    smart_height = max(4, 1.5 * (plot_depth + 1))
    if clf.n_outputs_ > 3:
        smart_height *= 0.92 + 0.08 * clf.n_outputs_
    smart_width = max(10, float(plot_leaves), 2 * smart_height)

    figure = Figure(figsize=(smart_width, smart_height), dpi=90)
    FigureCanvasAgg(figure)
    try:
        axis = figure.subplots()
        plot_tree_patched(
            tree,
            max_depth=max_depth,
            feature_names=feature_names,
            fontsize=8,
            ax=axis,
        )
        title = f"Tree {numeric_tree_index} for sample index {sample_index}"
        axis.set_title(title, fontsize=10 + int(1.3 * plot_depth))
        figure.tight_layout()

        image_buffer = io.BytesIO()
        figure.savefig(image_buffer, bbox_inches="tight", format="png")
        return image_buffer.getvalue(), title
    finally:
        figure.clear()


def _matplotlib_colorscale(colormap: Colormap, steps: int = 32) -> list[tuple[float, str]]:
    """Convert a Matplotlib colormap to Plotly's colorscale format."""
    positions = np.linspace(0.0, 1.0, steps)
    return [(float(position), to_hex(colormap(position))) for position in positions]


def _discrete_colorscale(colormap: Colormap, count: int) -> list[tuple[float, str]]:
    """Build a stepped Plotly colorscale with one color per cluster."""
    if count < 1:
        raise ValueError("At least one cluster is required")

    color_positions = np.linspace(0.0, 1.0, count) if count > 1 else np.asarray([0.5])
    scale: list[tuple[float, str]] = []
    for index, color_position in enumerate(color_positions):
        color = to_hex(colormap(float(color_position)))
        scale.extend([(index / count, color), ((index + 1) / count, color)])
    return scale


def _point_hover_text(point: InteractPoint, clustered: bool) -> str:
    if clustered:
        return f"Tree {point.name} | Cluster {point.cluster_memb}"
    return f"Tree {point.name} | Value {point.value}"


def _add_points_trace(
    figure: go.Figure,
    points: list[InteractPoint],
    selected: bool,
    clustered: bool,
) -> None:
    if not points:
        return

    figure.add_trace(
        go.Scatter(
            x=[float(point.pos[0]) for point in points],
            y=[float(point.pos[1]) for point in points],
            mode="markers",
            marker={
                "symbol": "star" if selected else "circle",
                "size": SIZE_SELECTED if selected else SIZE_CANDIDATE,
                "color": [point.color_value for point in points],
                "coloraxis": "coloraxis",
                "line": {"width": 1, "color": "DarkSlateGrey"},
            },
            text=[_point_hover_text(point, clustered) for point in points],
            customdata=[point.name for point in points],
            hovertemplate="%{text}<extra></extra>",
            name="Selected trees" if selected else "Candidate trees",
        )
    )


def build_plotly_figure(interactplot: InteractPlot) -> go.Figure:
    """Build a self-contained Plotly figure, including its colorbar."""
    figure = go.Figure()

    _add_points_trace(
        figure,
        [point for point in interactplot.points if not point.selected],
        selected=False,
        clustered=interactplot.clustered,
    )
    _add_points_trace(
        figure,
        [point for point in interactplot.points if point.selected],
        selected=True,
        clustered=interactplot.clustered,
    )

    colorbar: dict[str, object] = {
        "title": {"text": interactplot.colorbar_title},
        "thickness": 18,
    }
    if interactplot.colorbar_tickvals is not None:
        colorbar["tickvals"] = interactplot.colorbar_tickvals
    if interactplot.colorbar_ticktext is not None:
        colorbar["ticktext"] = interactplot.colorbar_ticktext

    figure.update_layout(
        title=interactplot.title,
        xaxis_title=interactplot.xlabel,
        yaxis_title=interactplot.ylabel,
        clickmode="event+select",
        width=PLOT_WIDTH,
        height=PLOT_HEIGHT,
        legend={"yanchor": "top", "y": 0.99, "xanchor": "left", "x": 0.01},
        margin={"l": 40, "r": 90, "t": 50, "b": 40},
        coloraxis={
            "colorscale": interactplot.colorscale,
            "cmin": interactplot.color_min,
            "cmax": interactplot.color_max,
            "colorbar": colorbar,
        },
    )
    return figure


def prepare_interactive_plots(
    plot_data_bunch,
    kmeans,
    input_method,
    colormap: str | LinearSegmentedColormap | None = None,
    clusterplots: tuple[bool, ...] = (True, False),
) -> list[InteractPlot]:
    """Project candidate trees and prepare NiceGUI's Plotly figures."""
    projection_input = np.asarray(plot_data_bunch.proj_data, dtype=float)
    if projection_input.ndim != 2 or min(projection_input.shape) < 2:
        raise ValueError(
            "Interactive plots require at least two trees and two representation features; "
            f"got shape {projection_input.shape}"
        )

    projected_data = PCA(n_components=2).fit_transform(projection_input)
    cluster_labels = np.asarray(kmeans.labels_, dtype=int)
    tree_indices = list(plot_data_bunch.index)
    point_count = len(tree_indices)
    if projected_data.shape[0] != point_count:
        raise ValueError(
            "Projection rows and tree indices must have matching lengths; "
            f"got {projected_data.shape[0]} and {point_count}"
        )
    if cluster_labels.shape != (point_count,):
        raise ValueError(
            "Cluster labels and tree indices must have matching lengths; "
            f"got {cluster_labels.size} and {point_count}"
        )
    unique_clusters = sorted(int(label) for label in np.unique(cluster_labels))
    cluster_numbers = {label: index + 1 for index, label in enumerate(unique_clusters)}
    final_tree_indices = set(input_method.final_trees_idx)

    pred_values = np.asarray(plot_data_bunch.pred, dtype=float).reshape(-1)
    loss_values = np.asarray(plot_data_bunch.loss, dtype=float).reshape(-1)
    rf_prediction = np.asarray(plot_data_bunch.rf_pred, dtype=float)
    scalar_prediction = rf_prediction.size == 1
    continuous_color_values = pred_values if scalar_prediction else loss_values
    if continuous_color_values.size != point_count:
        value_kind = "Prediction" if scalar_prediction else "Loss"
        raise ValueError(
            f"{value_kind} values and tree indices must have matching lengths; "
            f"got {continuous_color_values.size} and {point_count}"
        )

    cluster_colormap = colormaps["viridis"]
    value_colormap: Colormap = colormap_from_str(colormap)
    plots: list[InteractPlot] = []

    for clustered in clusterplots:
        if clustered:
            color_values = np.asarray(
                [cluster_numbers[label] for label in cluster_labels], dtype=float
            )
            hover_values = [str(cluster_numbers[label]) for label in cluster_labels]
            colorscale = _discrete_colorscale(cluster_colormap, len(unique_clusters))
            color_min = 0.5
            color_max = len(unique_clusters) + 0.5
            title = "Cluster colors"
            colorbar_title = "Cluster"
            tickvals = [float(value) for value in range(1, len(unique_clusters) + 1)]
            ticktext = [f"cl.{cluster_numbers[label]}" for label in unique_clusters]
        else:
            color_values = continuous_color_values
            if scalar_prediction:
                scalar_rf_prediction = float(rf_prediction.squeeze())
                color_min, color_max = custom_axes_limit(
                    float(np.min(color_values)),
                    float(np.max(color_values)),
                    scalar_rf_prediction,
                    plot_data_bunch.set_up == "binary",
                )
                colorbar_title = f"Prediction<br>RF: {scalar_rf_prediction:.3g}"
            else:
                color_min, color_max = custom_axes_limit(
                    float(np.min(color_values)),
                    float(np.max(color_values)),
                    force_in=None,
                    is_binary=False,
                )
                colorbar_title = f"{input_method.fidelity_measure} loss"
            hover_values = [f"{value:.3f}" for value in color_values]
            colorscale = _matplotlib_colorscale(value_colormap)
            title = "Prediction colors"
            tickvals = None
            ticktext = None

        points = [
            InteractPoint(
                name=str(tree_index),
                pos=projected_data[index],
                color_value=float(color_values[index]),
                selected=tree_index in final_tree_indices,
                cluster_memb=hover_values[index] if clustered else None,
                value=None if clustered else hover_values[index],
            )
            for index, tree_index in enumerate(tree_indices)
        ]
        plots.append(
            InteractPlot(
                name="clusters" if clustered else "predictions",
                points=points,
                clustered=clustered,
                title=title,
                colorbar_title=colorbar_title,
                colorscale=colorscale,
                color_min=float(color_min),
                color_max=float(color_max),
                colorbar_tickvals=tickvals,
                colorbar_ticktext=ticktext,
            )
        )

    return plots
