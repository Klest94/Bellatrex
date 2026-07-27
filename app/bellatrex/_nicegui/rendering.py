"""Rendering helpers for the NiceGUI frontend."""

from __future__ import annotations

import io
import os
from typing import TYPE_CHECKING
from uuid import uuid4

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colormaps
from matplotlib.axes import Axes
from matplotlib.colorbar import Colorbar
from matplotlib.colors import BoundaryNorm, LinearSegmentedColormap
from matplotlib.figure import Figure
from matplotlib.ticker import FuncFormatter
from sklearn.decomposition import PCA

from ..utilities import colormap_from_str, custom_axes_limit, custom_formatter, rule_print_inline
from .models import InteractPlot, InteractPoint

if TYPE_CHECKING:
    import plotly.graph_objects as go

SIZE_SIMPLE = 9.0
SIZE_SELECTED = 15.0
PLOT_WIDTH = 520
PLOT_HEIGHT = 430
COLORBAR_FIGSIZE = (1.94, 4.84)
COLORBAR_DPI = 110
CLUSTER_COLORBAR_CAX_RECT = (0.40, 0.06, 0.18, 0.88)
PREDICTION_COLORBAR_CAX_RECT = (0.50, 0.06, 0.18, 0.88)


def _feature_names_for_clf(clf) -> list[str]:
    if hasattr(clf, "feature_names_in_") and clf.feature_names_in_ is not None:
        return list(clf.feature_names_in_)
    return [f"X{i}" for i in range(clf.n_features_in_)]


def _create_colorbar_figure(
    cax_rect: tuple[float, float, float, float],
) -> tuple[Figure, Axes]:
    """Create a fixed canvas with an explicitly positioned colorbar axis."""
    figure = plt.figure(figsize=COLORBAR_FIGSIZE, dpi=COLORBAR_DPI)
    axis = figure.add_axes(cax_rect)
    return figure, axis


def print_tree_rule(tree_index, render_context: dict) -> None:
    """Print the rule for the selected tree to preserve click feedback."""
    clf = render_context["clf"]
    sample = render_context["sample"]
    rule_print_inline(clf[int(tree_index)], sample)


def render_tree_image(tree_index, render_context: dict) -> tuple[bytes, str]:
    from ..plot_tree_patch import plot_tree_patched

    clf = render_context["clf"]
    sample = render_context["sample"]
    sample_index = render_context.get("sample_index", sample.index[0])
    max_depth = render_context.get("max_depth")
    feature_names = render_context.get("feature_names") or _feature_names_for_clf(clf)

    tree_index = int(tree_index)
    the_tree = clf[tree_index]

    if max_depth is not None:
        real_plot_leaves = max(the_tree.tree_.n_leaves, 2 ** (max_depth - 1))
        real_plot_depth = min(the_tree.tree_.max_depth, max_depth)
    else:
        real_plot_leaves = the_tree.tree_.n_leaves
        real_plot_depth = the_tree.tree_.max_depth

    smart_width = max(10, 1.0 * real_plot_leaves)
    smart_height = max(4, 1.5 * (real_plot_depth + 1))
    if clf.n_outputs_ > 3:
        smart_height = smart_height * (0.92 + 0.08 * clf.n_outputs_)
    smart_width = max(smart_width, 2 * smart_height)

    figure, axis = plt.subplots(figsize=(smart_width, smart_height), dpi=90)
    plot_tree_patched(
        the_tree,
        max_depth=max_depth,
        feature_names=feature_names,
        fontsize=8,
        ax=axis,
    )
    title = f"Tree {tree_index} for sample index {sample_index}"
    axis.set_title(title, fontsize=10 + int(1.3 * real_plot_depth))
    figure.tight_layout()

    image_buffer = io.BytesIO()
    figure.savefig(image_buffer, bbox_inches="tight", format="png")
    plt.close(figure)
    return image_buffer.getvalue(), title


def write_tree_image(tree_png: bytes, tree_name: str, temp_files_dir: str) -> tuple[str, str]:
    image_name = f"tree_{tree_name}_{uuid4().hex}.png"
    image_path = os.path.abspath(os.path.join(temp_files_dir, image_name))
    with open(image_path, "wb") as image_file:
        image_file.write(tree_png)
    if not os.path.exists(image_path) or os.path.getsize(image_path) == 0:
        raise OSError(f"Could not write tree image for {tree_name}.")
    return image_name, image_path


def _is_scalar_like(value: object) -> bool:
    return np.asarray(value).size == 1


def _rgba_to_plotly(rgba: list[float]) -> str:
    return (
        f"rgba({int(rgba[0])},{int(rgba[1])},{int(rgba[2])},"
        f"{float(rgba[3]) / 255.0:.2f})"
    )


def _point_hover_text(point: InteractPoint, clustered: bool) -> str:
    if clustered:
        return f"Tree {point.name} | Cluster {point.cluster_memb}"
    return f"Tree {point.name} | Pred {point.value}"


def _add_points_trace(
    figure: go.Figure,
    points: list[InteractPoint],
    symbol: str,
    size: float,
    trace_name: str,
    clustered: bool,
    plotly_go: object,
) -> None:
    if not points:
        return

    scatter = getattr(plotly_go, "Scatter")
    figure.add_trace(
        scatter(
            x=[float(point.pos[0]) for point in points],
            y=[float(point.pos[1]) for point in points],
            mode="markers",
            marker=dict(
                symbol=symbol,
                size=size,
                color=[_rgba_to_plotly(point.color) for point in points],
                line=dict(width=1, color="DarkSlateGrey"),
            ),
            text=[_point_hover_text(point, clustered) for point in points],
            customdata=[point.name for point in points],
            hoverinfo="text",
            name=trace_name,
            showlegend=False,
        )
    )


def _add_legend_trace(
    figure: go.Figure,
    symbol: str,
    size: float,
    trace_name: str,
    fill_color: str = "rgba(220,220,220,1.0)",
    plotly_go: object | None = None,
) -> None:
    if plotly_go is None:
        raise ValueError("plotly_go must be provided when adding legend traces")

    scatter = getattr(plotly_go, "Scatter")
    figure.add_trace(
        scatter(
            x=[None],
            y=[None],
            mode="markers",
            marker=dict(
                symbol=symbol,
                size=size,
                color=fill_color,
                line=dict(width=1, color="black"),
            ),
            hoverinfo="skip",
            name=trace_name,
        )
    )


def build_plotly_figure(interactplot: InteractPlot) -> go.Figure:
    import plotly.graph_objects as go  # pylint: disable=import-outside-toplevel

    figure = go.Figure()
    candidate_points = [point for point in interactplot.points if point.shape != "star"]
    selected_points = [point for point in interactplot.points if point.shape == "star"]

    _add_points_trace(
        figure,
        candidate_points,
        "circle",
        SIZE_SIMPLE,
        "Candidate trees",
        interactplot.clustered,
        go,
    )
    _add_points_trace(
        figure,
        selected_points,
        "star",
        SIZE_SELECTED,
        "Selected trees",
        interactplot.clustered,
        go,
    )
    _add_legend_trace(figure, "circle", SIZE_SIMPLE, "Candidate trees", plotly_go=go)
    _add_legend_trace(figure, "star", SIZE_SELECTED, "Selected trees", plotly_go=go)

    figure.update_layout(
        title="Cluster colors" if interactplot.clustered else "Prediction colors",
        xaxis_title=interactplot.xlabel,
        yaxis_title=interactplot.ylabel,
        clickmode="event+select",
        width=PLOT_WIDTH,
        height=PLOT_HEIGHT,
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
        margin=dict(l=40, r=20, t=50, b=40),
    )
    return figure


def build_colorbar_paths(plots, temp_files_dir: str) -> list[str]:
    return [
        os.path.abspath(os.path.join(temp_files_dir, f"temp_colourbar{i}.png"))
        for i in range(len(plots))
    ]


def plot_with_interface(
    plot_data_bunch,
    kmeans,
    input_method,
    temp_files_dir,
    max_depth=None,
    colormap=None,
    clusterplots=(True, False),
):
    """Prepare plot data for the NiceGUI interactive frontend."""
    _ = max_depth

    pca_fitted = PCA(n_components=2).fit(plot_data_bunch.proj_data)
    plottable_data = pca_fitted.transform(plot_data_bunch.proj_data)

    cluster_memb = kmeans.labels_
    final_tree_indices = set(input_method.final_trees_idx)
    is_final_candidate = [
        plot_data_bunch.index[i] in final_tree_indices for i in range(len(plot_data_bunch.index))
    ]
    custom_sizes = [SIZE_SELECTED if is_final else SIZE_SIMPLE for is_final in is_final_candidate]
    custom_shapes = ["star" if is_final else "circle" for is_final in is_final_candidate]

    plots = []
    color_map_left = colormaps["viridis"]
    color_map_left = LinearSegmentedColormap.from_list(
        "Custom cmap", [color_map_left(i) for i in range(color_map_left.N)], color_map_left.N
    )
    color_map_right = colormap_from_str(colormap)
    pred_values = np.atleast_1d(np.asarray(plot_data_bunch.pred).squeeze())
    loss_values = np.atleast_1d(np.asarray(plot_data_bunch.loss).squeeze())
    rf_pred_is_scalar = _is_scalar_like(plot_data_bunch.rf_pred)
    scalar_rf_pred = (
        float(np.asarray(plot_data_bunch.rf_pred).squeeze()) if rf_pred_is_scalar else None
    )

    for plotindex, clustered in enumerate(clusterplots):
        if clustered:
            fig, ax = _create_colorbar_figure(CLUSTER_COLORBAR_CAX_RECT)

            freqs = np.bincount(cluster_memb)
            if np.min(freqs) == 0:
                raise KeyError(
                    "There are empty clusters, the scatter and colorbar could differ in color shade"
                )
            norm_bins = np.array([0] + list(np.cumsum(freqs)))
            labels = [f"cl.{int(i + 1):d}" for i in np.unique(cluster_memb)]

            norm = BoundaryNorm(norm_bins, color_map_left.N)
            tickz = (norm_bins[:-1] + (norm_bins[1:] - norm_bins[:-1]) / 2).tolist()

            cb = Colorbar(
                ax,
                cmap=color_map_left,
                norm=norm,
                spacing="proportional",
                ticks=tickz,
                boundaries=norm_bins.tolist(),
                format="%1i",
            )
            cb.ax.set_yticklabels(labels)
            ax.yaxis.set_ticks_position("left")
            norms = [norm(norm_bins[cluster_memb[i]]) for i in range(len(cluster_memb))]
        else:
            fig, ax = _create_colorbar_figure(PREDICTION_COLORBAR_CAX_RECT)
            if rf_pred_is_scalar:
                if scalar_rf_pred is None:
                    raise ValueError("A scalar random-forest prediction was expected.")
                is_binary = plot_data_bunch.set_up == "binary"
                v_min, v_max = custom_axes_limit(
                    np.asarray(pred_values).min(),
                    np.asarray(pred_values).max(),
                    scalar_rf_pred,
                    is_binary,
                )
                norm_preds = BoundaryNorm(np.linspace(v_min, v_max, 256), color_map_right.N)
                pred_tick = round(scalar_rf_pred, 3)
                cb = Colorbar(ax, cmap=color_map_right, norm=norm_preds)
                ax.set_title("RF pred: " + str(pred_tick), fontsize=8, pad=6)
                cb.ax.plot([0, 1], [pred_values] * 2, color="grey", linewidth=1)
                cb.ax.plot([0.02, 0.98], [pred_tick] * 2, color="black", linewidth=2.5, marker="P")
            else:
                v_min, v_max = custom_axes_limit(
                    np.asarray(loss_values).min(),
                    np.asarray(loss_values).max(),
                    force_in=None,
                    is_binary=False,
                )
                norm_preds = BoundaryNorm(np.linspace(v_min, v_max, 256), color_map_right.N)
                cb = Colorbar(ax, cmap=color_map_right, norm=norm_preds)
                ax.set_title(str(input_method.fidelity_measure) + " loss", fontsize=8, pad=6)
                cb.ax.plot([0, 1], [loss_values] * 2, color="grey", linewidth=1)

            ticks_to_plot = ax.get_yticks()
            if np.abs(np.min(ticks_to_plot)) < 1e-3 and np.abs(np.max(ticks_to_plot)) > 1e-2:
                min_index = np.argmin(ticks_to_plot)
                ticks_to_plot[min_index] = 0
                ax.set_yticks(ticks_to_plot)

            ax.yaxis.set_major_formatter(FuncFormatter(custom_formatter))
            ax.minorticks_off()
            if rf_pred_is_scalar:
                norms = [norm_preds(float(pred_values[i])) for i in range(len(cluster_memb))]
            else:
                norms = [norm_preds(loss_values[i]) for i in range(len(cluster_memb))]

        fig.savefig(
            os.path.join(temp_files_dir, f"temp_colourbar{plotindex}.png"),
        )
        plt.close(fig)

        cmap_gui = color_map_left if clustered else color_map_right
        colours = [[channel * 255 for channel in cmap_gui(norms[i])] for i in range(len(norms))]

        points = []
        for j, index in enumerate(plot_data_bunch.index):
            point = InteractPoint(
                index,
                plottable_data[j],
                colours[j],
                custom_sizes[j],
                custom_shapes[j],
            )
            if clustered:
                point.cluster_memb = str(cluster_memb[j] + 1)
            else:
                if isinstance(pred_values[j], (float, np.floating)):
                    point.value = f"{float(pred_values[j]):.3f}"
                elif isinstance(loss_values[j], (float, np.floating)):
                    point.value = f"{float(loss_values[j]):.3f}"
                else:
                    raise ValueError(
                        f"Expected numeric prediction/loss values, got {type(loss_values[j])} instead"
                    )
            points.append(point)

        plots.append(InteractPlot(plotindex, points, clustered=clustered, sample_idx=plotindex))

    return plots
