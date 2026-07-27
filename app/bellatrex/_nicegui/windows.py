import os
import re
import sys

from nicegui import app as ng_app
from nicegui import ui

from .cache import TreeCacheEntry, TreeRenderCache
from .rendering import (
    build_colorbar_paths,
    build_plotly_figure,
    print_tree_rule,
    render_tree_image,
    write_tree_image,
)
from .runtime import (
    build_main_window_payload,
    build_tree_window_payload,
    cleanup_temp_artifacts,
    detect_native_window_support,
    ensure_nicegui_screen_test_port,
    find_free_port,
    prepare_session_temp_dir,
    prepare_tree_window_temp_dir,
    run_subprocess_app,
)

MAIN_WINDOW_SIZE = (1440, 720)
TREE_WINDOW_SIZE = (1440, 900)

TREE_WINDOW_HEAD_CSS = """
<style>
    html, body {
        height: 100%;
        margin: 0;
        overflow: hidden;
    }
    .nicegui-content {
        height: 100vh;
        overflow: hidden;
    }
</style>
"""

TREE_WINDOW_LAYOUT_STYLE = (
    "width:100%; height:100%; gap:0.75rem; padding:1rem; " "box-sizing:border-box; overflow:hidden;"
)
TREE_WINDOW_VIEWPORT_STYLE = (
    "flex:1 1 auto; min-height:0; min-width:0; width:100%; overflow-x:auto; "
    "overflow-y:auto; border:1px solid #e5e7eb; border-radius:4px; "
    "background:#f5f5f5; padding:12px; box-sizing:border-box;"
)
TREE_WINDOW_IMAGE_WRAPPER_STYLE = (
    "display:inline-block; width:max-content; height:max-content; background:white;"
)
TREE_WINDOW_IMAGE_STYLE = "display:block; width:auto; height:auto; max-width:none; max-height:none;"
MAIN_PLOTS_CONTAINER_STYLE = (
    "display:flex; flex-direction:row; align-items:flex-start; "
    "gap:2rem; overflow-x:auto; padding:0 2rem 1rem 1rem;"
)
PLOT_PAIR_STYLE = (
    "display:flex; flex-direction:row; align-items:flex-start; gap:4px; flex-shrink:0;"
)
COLORBAR_CONTAINER_STYLE = (
    "height:484px; width:134px; display:flex; align-items:center; "
    "justify-content:flex-start; padding-top:-50px; padding-right:4px; flex-shrink:0;"
)
COLORBAR_IMAGE_STYLE = "height:440px; width:134px; display:block; flex-shrink:0;"


def _extract_click_selection(args: object) -> tuple[str, object]:
    text = "Point selected"

    if isinstance(args, dict):
        if "tree_name" in args:
            return args.get("text") or text, args.get("tree_name")

        points = args.get("points")
        if isinstance(points, list) and points and isinstance(points[0], dict):
            point = points[0]
            return point.get("text") or text, point.get("customdata")

        return text, None

    if isinstance(args, list) and args and isinstance(args[0], dict):
        point = args[0]
        return point.get("text") or text, point.get("customdata")

    return text, None


def _normalize_tree_name(raw_tree_name: object) -> str | None:
    if raw_tree_name is None:
        return None

    tree_name = raw_tree_name
    if isinstance(tree_name, (list, tuple)):
        if not tree_name:
            return None
        tree_name = tree_name[0]
    elif isinstance(tree_name, dict):
        preferred_keys = ("tree_name", "customdata", "name", "text")
        for key in preferred_keys:
            if key in tree_name and tree_name[key] is not None:
                tree_name = tree_name[key]
                break
        else:
            for value in tree_name.values():
                if value is not None:
                    tree_name = value
                    break

    if tree_name is None:
        return None

    if isinstance(tree_name, (list, tuple)):
        if not tree_name:
            return None
        tree_name = tree_name[0]

    if isinstance(tree_name, str):
        match = re.search(r"-?\d+", tree_name)
        return match.group(0) if match else None

    if isinstance(tree_name, dict):
        return None

    if isinstance(tree_name, (int, float)):
        return str(int(tree_name))
    return None


def _build_tree_cache_entry(
    tree_name: str,
    render_context: dict,
    temp_files_dir: str,
) -> TreeCacheEntry:
    tree_png, title = render_tree_image(tree_name, render_context)
    image_name, image_path = write_tree_image(tree_png, tree_name, temp_files_dir)
    return TreeCacheEntry(
        tree_name=str(tree_name),
        image_name=image_name,
        image_path=image_path,
        image_source=f"/bellatrex_tmp/{image_name}",
        title=title,
    )


def _run_tree_window_app(
    image_name: str,
    title: str,
    subtitle: str | None,
    native: bool,
    port: int,
    temp_files_dir: str,
    payload_path: str | None = None,
) -> None:
    ensure_nicegui_screen_test_port(port)
    ng_app.add_static_files("/bellatrex_tmp", temp_files_dir)
    tree_source = f"/bellatrex_tmp/{image_name}"
    tree_image_path = os.path.abspath(os.path.join(temp_files_dir, image_name))

    auto_close = os.getenv("BELLATREX_GUI_AUTO_CLOSE_SECONDS")
    try:
        auto_close_seconds = float(auto_close) if auto_close else 2.0
    except ValueError:
        auto_close_seconds = 2.0

    @ui.page("/")
    def tree_window_page() -> None:
        ui.add_head_html(TREE_WINDOW_HEAD_CSS)
        with ui.column().style(TREE_WINDOW_LAYOUT_STYLE):
            ui.label(title).classes("text-xl font-semibold")
            if subtitle:
                ui.label(subtitle).classes("text-sm text-gray-500")
            ui.link("Open in browser", tree_source, new_tab=True).classes("text-sm")
            with ui.element("div").style(TREE_WINDOW_VIEWPORT_STYLE):
                with ui.element("div").style(TREE_WINDOW_IMAGE_WRAPPER_STYLE):
                    ui.element("img").props(f'src="{tree_source}" alt="{title}"').style(
                        TREE_WINDOW_IMAGE_STYLE
                    )
        if auto_close_seconds > 0:
            ui.timer(auto_close_seconds, lambda: ng_app.shutdown(), once=True)

    ng_app.on_shutdown(
        lambda: cleanup_temp_artifacts(
            temp_files_dir,
            tree_image_paths=[tree_image_path],
            payload_path=payload_path,
        )
    )

    ui.run(
        native=native,
        port=port,
        reload=False,
        title=title,
        show=not native,
        window_size=TREE_WINDOW_SIZE,
    )


def _run_nicegui_app(
    plots,
    colorbar_paths,
    render_context: dict | None,
    native: bool,
    port: int,
    temp_files_dir: str,
    payload_path: str | None = None,
) -> None:
    ensure_nicegui_screen_test_port(port)
    ng_app.add_static_files("/bellatrex_tmp", temp_files_dir)
    tree_cache = TreeRenderCache()

    auto_close = os.getenv("BELLATREX_GUI_AUTO_CLOSE_SECONDS")
    try:
        auto_close_seconds = float(auto_close) if auto_close else 0.0
    except ValueError:
        auto_close_seconds = 0.0

    @ui.page("/")
    def main_page() -> None:
        sample_indices = [str(plot.sample_idx) for plot in plots]
        sample_index = sample_indices[0] if sample_indices else "unknown"
        ui.label(f"Bellatrex visual explanation, sample {sample_index}").classes(
            "text-2xl font-bold p-4"
        )
        info_label = ui.label("Click on a point to open the corresponding tree").classes(
            "text-sm italic text-gray-500 px-4 pb-2"
        )

        def _open_tree_window(tree_name: str, subtitle: str | None = None) -> None:
            if render_context is None:
                ui.notify("Tree rendering context is unavailable.", color="warning")
                return

            try:
                print_tree_rule(tree_name, render_context)
                entry, _ = tree_cache.get_or_create(
                    tree_name,
                    lambda normalized_tree_name: _build_tree_cache_entry(
                        normalized_tree_name,
                        render_context,
                        temp_files_dir,
                    ),
                )
            except (KeyError, IndexError, OSError, TypeError, ValueError) as exc:
                ui.notify(f"Could not render tree {tree_name}: {exc}", color="negative")
                return

            if native:
                try:
                    child_temp_dir, child_image_name, _ = prepare_tree_window_temp_dir(
                        temp_files_dir,
                        entry.image_path,
                    )
                    tree_payload_path = build_tree_window_payload(
                        image_name=child_image_name,
                        title=entry.title,
                        subtitle=subtitle,
                        native=True,
                        port=find_free_port(),
                        temp_files_dir=child_temp_dir,
                    )
                    run_subprocess_app(tree_payload_path, blocking=False)
                except OSError as exc:
                    ui.notify(f"Could not open tree window: {exc}", color="negative")
                    return
            else:
                ui.run_javascript(
                    f"window.open({entry.image_source!r}, '_blank', "
                    "'noopener,noreferrer,width=1400,height=900');"
                )

            if subtitle:
                info_label.set_text(f"{subtitle} | Opened {entry.title}")
            else:
                info_label.set_text(f"Opened {entry.title}")

        with ui.element("div").style(MAIN_PLOTS_CONTAINER_STYLE):
            for idx, interactplot in enumerate(plots):
                with ui.element("div").style(PLOT_PAIR_STYLE):
                    with ui.element("div").style("flex-shrink:0;"):
                        plot_elem = ui.plotly(build_plotly_figure(interactplot))

                        def _on_plotly_click(event, lbl=info_label) -> None:
                            text, raw_tree_name = _extract_click_selection(
                                getattr(event, "args", None)
                            )
                            lbl.set_text(text)
                            tree_name = _normalize_tree_name(raw_tree_name)
                            if tree_name is not None:
                                _open_tree_window(tree_name, text)

                        plot_elem.on(
                            "plotly_click",
                            _on_plotly_click,
                            js_handler=(
                                "(event) => {"
                                " const p = event?.points?.[0] || {};"
                                " emit({tree_name: p.customdata, text: p.text});"
                                "}"
                            ),
                        )

                    cb_path = colorbar_paths[idx] if idx < len(colorbar_paths) else None
                    if cb_path and os.path.exists(cb_path):
                        cb_source = (
                            f"/bellatrex_tmp/{os.path.basename(cb_path)}"
                            f"?v={int(os.path.getmtime(cb_path))}"
                        )
                        with ui.element("div").style(COLORBAR_CONTAINER_STYLE):
                            ui.image(cb_source).props("fit=contain").style(COLORBAR_IMAGE_STYLE)
                    else:
                        ui.label("Colorbar missing").classes("text-xs text-red-600")
        if auto_close_seconds > 0:
            ui.timer(auto_close_seconds, lambda: ng_app.shutdown(), once=True)

    ng_app.on_shutdown(
        lambda: cleanup_temp_artifacts(
            temp_files_dir,
            colorbar_paths=colorbar_paths,
            tree_image_paths=tree_cache.image_paths(),
            payload_path=payload_path,
        )
    )

    ui.run(
        native=native,
        port=port,
        reload=False,
        title="Bellatrex",
        show=not native,
        window_size=MAIN_WINDOW_SIZE,
    )


def launch_nicegui_window(
    plots,
    temp_files_dir: str,
    blocking=None,
    render_context: dict | None = None,
) -> None:
    """Open the Bellatrex NiceGUI explorer.

    ``blocking=False`` does not delete files before the next caller continues.
    Instead, each launch gets its own session directory, so overlapping
    iterations can safely reuse tree ids without collisions.
    """
    from multiprocessing import get_context

    if blocking is None:
        # Prefer non-blocking by default so callers that simply want to
        # "open a window" don't block program execution unexpectedly.
        blocking = False

    native = detect_native_window_support()
    colorbar_paths = build_colorbar_paths(plots, temp_files_dir)
    session_temp_files_dir, session_colorbar_paths = prepare_session_temp_dir(
        temp_files_dir,
        colorbar_paths,
    )
    port = find_free_port()
    ensure_nicegui_screen_test_port(port)

    if sys.platform.startswith("win"):
        payload_path = build_main_window_payload(
            plots,
            session_colorbar_paths,
            render_context,
            native,
            port,
            session_temp_files_dir,
        )
        try:
            run_subprocess_app(payload_path, blocking)
        finally:
            if blocking:
                cleanup_temp_artifacts(
                    session_temp_files_dir,
                    colorbar_paths=session_colorbar_paths,
                    payload_path=payload_path,
                )
        return

    proc = get_context("spawn").Process(
        target=_run_nicegui_app,
        args=(
            plots,
            session_colorbar_paths,
            render_context,
            native,
            port,
            session_temp_files_dir,
        ),
        daemon=False,
    )
    proc.start()

    if blocking:
        proc.join()
        cleanup_temp_artifacts(
            session_temp_files_dir,
            colorbar_paths=session_colorbar_paths,
        )
        if proc.exitcode not in (0, None):
            raise RuntimeError(f"NiceGUI window process exited with code {proc.exitcode}.")
