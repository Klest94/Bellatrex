"""NiceGUI page composition for the interactive Bellatrex explorer."""

from __future__ import annotations

import base64
from collections.abc import Callable
from numbers import Integral, Real

from nicegui import ui
from nicegui.events import GenericEventArguments

from .models import InteractPlot, RenderContext
from .rendering import build_plotly_figure, print_tree_rule, render_tree_image

TreeRenderer = Callable[[str, RenderContext], tuple[bytes, str]]
RulePrinter = Callable[[str, RenderContext], None]

_TRANSPARENT_PIXEL = "data:image/gif;base64,R0lGODlhAQABAAD/ACwAAAAAAQABAAACADs="


def extract_click_selection(args: object) -> tuple[str | None, str]:
    """Read the small event payload emitted by the Plotly click handler."""
    if not isinstance(args, dict):
        return None, "Select a tree point to inspect it"

    raw_tree_name = args.get("tree_name")
    if isinstance(raw_tree_name, (list, tuple)):
        raw_tree_name = raw_tree_name[0] if raw_tree_name else None

    tree_name: str | None = None
    if isinstance(raw_tree_name, Integral):
        tree_name = str(int(raw_tree_name))
    elif isinstance(raw_tree_name, Real) and float(raw_tree_name).is_integer():
        tree_name = str(int(float(raw_tree_name)))
    elif isinstance(raw_tree_name, str):
        try:
            tree_name = str(int(raw_tree_name.strip()))
        except ValueError:
            pass

    text = str(args.get("text") or "Select a tree point to inspect it")
    return tree_name if tree_name is not None and int(tree_name) >= 0 else None, text


def _image_data_url(image_png: bytes) -> str:
    return "data:image/png;base64," + base64.b64encode(image_png).decode("ascii")


def build_explorer_page(
    plots: list[InteractPlot],
    render_context: RenderContext | None = None,
    *,
    tree_renderer: TreeRenderer = render_tree_image,
    rule_printer: RulePrinter = print_tree_rule,
) -> None:
    """Compose the explorer page; kept separate from ``ui.run`` for easy tests."""
    tree_cache: dict[str, tuple[str, str]] = {}
    tree_names = {point.name for plot in plots for point in plot.points}

    with (
        ui.dialog().props("maximized") as tree_dialog,
        ui.card().style("width:100vw; height:100vh; max-width:none; padding:1rem;"),
    ):
        with ui.row().style("width:100%; align-items:center; justify-content:space-between;"):
            tree_title = (
                ui.label("Selected tree")
                .classes("text-xl font-semibold")
                .style("flex:1; min-width:0;")
                .mark("tree-title")
            )
            ui.button(icon="close", on_click=tree_dialog.close).props("flat round").mark(
                "close-tree"
            )
        tree_subtitle = (
            ui.label().classes("text-sm text-gray-500").style("width:100%;").mark("tree-subtitle")
        )
        with ui.scroll_area().style(
            "width:100%; height:calc(100vh - 7rem); border:1px solid #e5e7eb; "
            "border-radius:0.25rem; background:#f3f4f6;"
        ):
            tree_image = (
                ui.element("img")
                .props(f'src={_TRANSPARENT_PIXEL} alt="Selected tree diagram"')
                .style("max-width:none; background:white;")
                .mark("tree-image")
            )
            tree_image.set_visibility(False)

    sample_index = render_context.get("sample_index", "unknown") if render_context else "unknown"
    page = ui.column().style("width:100%; gap:0;")
    with page:
        ui.label(f"Bellatrex visual explanation, sample {sample_index}").classes(
            "text-2xl font-bold p-4"
        ).style("width:100%;").mark("explorer-title")
        info_label = (
            ui.label("Click a point to inspect the corresponding tree")
            .classes("text-sm italic text-gray-500 px-4 pb-2")
            .style("width:100%;")
            .mark("explorer-status")
        )

    def open_tree(tree_name: str, subtitle: str) -> None:
        if render_context is None:
            ui.notify("Tree rendering context is unavailable.", color="warning")
            return

        try:
            rule_printer(tree_name, render_context)
            if tree_name not in tree_cache:
                tree_png, title = tree_renderer(tree_name, render_context)
                tree_cache[tree_name] = (_image_data_url(tree_png), title)
            image_source, title = tree_cache[tree_name]
        except (KeyError, IndexError, OSError, TypeError, ValueError) as exc:
            ui.notify(f"Could not render tree {tree_name}: {exc}", color="negative")
            return

        tree_title.set_text(title)
        tree_subtitle.set_text(subtitle)
        tree_image.props(f"src={image_source}")
        tree_image.set_visibility(True)
        info_label.set_text(f"Opened {title}")
        tree_dialog.open()

    def on_plotly_click(event: GenericEventArguments) -> None:
        tree_name, text = extract_click_selection(event.args)
        info_label.set_text(text)
        if tree_name in tree_names:
            open_tree(tree_name, text)

    with page:
        with ui.row().classes("w-full flex-wrap items-start justify-center gap-4 px-2"):
            for index, interactplot in enumerate(plots):
                plot_element = (
                    ui.plotly(build_plotly_figure(interactplot))
                    .classes("max-w-full")
                    .mark(f"explorer-plot-{index}")
                )
                plot_element.on(
                    "plotly_click",
                    on_plotly_click,
                    js_handler=(
                        "(event) => {"
                        " const point = event?.points?.[0] || {};"
                        " emit({tree_name: point.customdata, text: point.text});"
                        "}"
                    ),
                )


def run_explorer(
    plots: list[InteractPlot],
    render_context: RenderContext | None = None,
    *,
    port: int | None = None,
    show: bool = True,
) -> None:
    """Run the browser-based NiceGUI explorer in the current process."""
    ui.run(
        lambda: build_explorer_page(plots, render_context),
        title="Bellatrex",
        host="127.0.0.1",
        port=port,
        show=show,
        native=False,
        reload=False,
    )
