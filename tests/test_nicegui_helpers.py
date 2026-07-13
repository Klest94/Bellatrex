import os
import pickle
import sys
import types

import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")
pytest.importorskip("nicegui", reason="Install Bellatrex[gui] to run GUI tests")

pytestmark = pytest.mark.gui

try:
    from app.bellatrex import nicegui_plots_code
    from app.bellatrex._nicegui.cache import TreeCacheEntry, TreeRenderCache
    from app.bellatrex._nicegui.models import InteractPlot, InteractPoint
    from app.bellatrex._nicegui.rendering import build_plotly_figure, plot_with_interface
    from app.bellatrex._nicegui.runtime import (
        build_main_window_payload,
        cleanup_temp_artifacts,
        ensure_nicegui_screen_test_port,
        prepare_session_temp_dir,
        prepare_tree_window_temp_dir,
    )
except ImportError:
    from bellatrex import nicegui_plots_code
    from bellatrex._nicegui.cache import TreeCacheEntry, TreeRenderCache
    from bellatrex._nicegui.models import InteractPlot, InteractPoint
    from bellatrex._nicegui.rendering import build_plotly_figure, plot_with_interface
    from bellatrex._nicegui.runtime import (
        build_main_window_payload,
        cleanup_temp_artifacts,
        ensure_nicegui_screen_test_port,
        prepare_session_temp_dir,
        prepare_tree_window_temp_dir,
    )


class DummyKMeans:
    def __init__(self, labels):
        self.labels_ = np.asarray(labels)


class DummyInputMethod:
    def __init__(self):
        self.final_trees_idx = [1, 3]
        self.fidelity_measure = "dummy"


class DummyPlotDataBunch:
    def __init__(self):
        self.proj_data = np.array(
            [
                [0.1, 0.2, 0.3],
                [0.3, 0.2, 0.1],
                [0.2, 0.4, 0.6],
                [0.5, 0.1, 0.2],
            ]
        )
        self.index = [0, 1, 2, 3]
        self.pred = np.array([0.2, 0.4, 0.6, 0.8])
        self.loss = np.array([0.8, 0.6, 0.4, 0.2])
        self.rf_pred = 0.5
        self.set_up = "binary"


def test_plot_with_interface_returns_expected_plot_objects(tmp_path):
    plots = plot_with_interface(
        DummyPlotDataBunch(),
        DummyKMeans([0, 1, 0, 1]),
        DummyInputMethod(),
        temp_files_dir=str(tmp_path),
    )

    assert len(plots) == 2
    assert all(isinstance(plot, InteractPlot) for plot in plots)
    assert all(len(plot.points) == 4 for plot in plots)
    assert (tmp_path / "temp_colourbar0.png").exists()
    assert (tmp_path / "temp_colourbar1.png").exists()


def test_build_plotly_figure_uses_local_plotly_import(monkeypatch):
    fake_plotly = types.ModuleType("plotly")
    fake_go = types.ModuleType("plotly.graph_objects")

    class FakeFigure:
        def __init__(self):
            self.traces = []
            self.layout = {}

        def add_trace(self, trace):
            self.traces.append(trace)

        def update_layout(self, **kwargs):
            self.layout.update(kwargs)

    def fake_scatter(**kwargs):
        return kwargs

    fake_go.Figure = FakeFigure
    fake_go.Scatter = fake_scatter
    fake_plotly.graph_objects = fake_go

    monkeypatch.setitem(sys.modules, "plotly", fake_plotly)
    monkeypatch.setitem(sys.modules, "plotly.graph_objects", fake_go)

    plot = InteractPlot(
        name="0",
        clustered=True,
        points=[
            InteractPoint(
                name="1",
                pos=np.asarray([0.1, 0.2]),
                color=[0, 0, 255, 255],
                size=9.0,
                shape="circle",
                cluster_memb="1",
            ),
            InteractPoint(
                name="2",
                pos=np.asarray([0.3, 0.4]),
                color=[255, 0, 0, 255],
                size=15.0,
                shape="star",
                cluster_memb="2",
            ),
        ],
    )
    figure = build_plotly_figure(plot)

    assert len(figure.traces) == 4
    assert figure.traces[0]["customdata"] == ["1"]
    assert figure.traces[1]["customdata"] == ["2"]
    assert figure.layout["title"] == "Cluster colors"


def test_tree_render_cache_reuses_existing_entry():
    cache = TreeRenderCache()
    calls = {"count": 0}

    def build_entry(tree_name: str) -> TreeCacheEntry:
        calls["count"] += 1
        return TreeCacheEntry(
            tree_name=tree_name,
            image_name=f"{tree_name}.png",
            image_path=f"/tmp/{tree_name}.png",
            image_source=f"/bellatrex_tmp/{tree_name}.png",
            title=f"Tree {tree_name}",
        )

    first_entry, first_created = cache.get_or_create("41", build_entry)
    second_entry, second_created = cache.get_or_create("41", build_entry)

    assert first_created is True
    assert second_created is False
    assert first_entry is second_entry
    assert calls["count"] == 1


def test_tree_render_cache_stores_distinct_trees():
    cache = TreeRenderCache()

    left_entry, _ = cache.get_or_create(
        "41",
        lambda tree_name: TreeCacheEntry(
            tree_name=tree_name,
            image_name=f"{tree_name}.png",
            image_path=f"/tmp/{tree_name}.png",
            image_source=f"/bellatrex_tmp/{tree_name}.png",
            title=f"Tree {tree_name}",
        ),
    )
    right_entry, _ = cache.get_or_create(
        "42",
        lambda tree_name: TreeCacheEntry(
            tree_name=tree_name,
            image_name=f"{tree_name}.png",
            image_path=f"/tmp/{tree_name}.png",
            image_source=f"/bellatrex_tmp/{tree_name}.png",
            title=f"Tree {tree_name}",
        ),
    )

    assert left_entry is not right_entry
    assert set(cache.image_paths()) == {"/tmp/41.png", "/tmp/42.png"}


def test_prepare_session_temp_dir_isolates_iterations(tmp_path):
    first_colorbar = tmp_path / "temp_colourbar0.png"
    first_colorbar.write_text("first", encoding="utf8")
    session_one_dir, session_one_paths = prepare_session_temp_dir(
        str(tmp_path), [str(first_colorbar)]
    )

    second_colorbar = tmp_path / "temp_colourbar0.png"
    second_colorbar.write_text("second", encoding="utf8")
    session_two_dir, session_two_paths = prepare_session_temp_dir(
        str(tmp_path), [str(second_colorbar)]
    )

    assert session_one_dir != session_two_dir
    assert session_one_paths[0] != session_two_paths[0]
    assert open(session_one_paths[0], encoding="utf8").read() == "first"
    assert open(session_two_paths[0], encoding="utf8").read() == "second"


def test_child_window_temp_copy_survives_main_session_cleanup(tmp_path):
    session_dir = tmp_path / "bellatrex_session_parent"
    session_dir.mkdir()
    session_image = session_dir / "tree_41.png"
    session_image.write_bytes(b"png-bytes")

    child_dir, _, child_image_path = prepare_tree_window_temp_dir(
        str(session_dir),
        str(session_image),
    )
    cleanup_temp_artifacts(
        str(session_dir),
        tree_image_paths=[str(session_image)],
    )

    assert not session_dir.exists()
    assert child_dir != str(session_dir)
    assert child_image_path != str(session_image)
    assert child_image_path.endswith("tree_41.png")
    assert open(child_image_path, "rb").read() == b"png-bytes"


def test_public_facade_exports_stable_functions():
    assert callable(nicegui_plots_code.plot_with_interface)
    assert callable(nicegui_plots_code.launch_nicegui_window)


def test_main_window_payload_round_trip(tmp_path):
    payload_path = build_main_window_payload(
        plots=["plot"],
        colorbar_paths=["colorbar.png"],
        render_context={"sample_index": 7},
        native=False,
        port=8123,
        temp_files_dir=str(tmp_path),
    )

    with open(payload_path, "rb") as handle:
        payload = pickle.load(handle)

    assert payload["kind"] == "main_window"
    assert payload["plots"] == ["plot"]
    assert payload["colorbar_paths"] == ["colorbar.png"]
    assert payload["render_context"]["sample_index"] == 7
    assert payload["port"] == 8123


def test_ensure_nicegui_screen_test_port_sets_default(monkeypatch):
    monkeypatch.delenv("NICEGUI_SCREEN_TEST_PORT", raising=False)

    ensure_nicegui_screen_test_port(4567)

    assert os.environ["NICEGUI_SCREEN_TEST_PORT"] == "4567"


def test_ensure_nicegui_screen_test_port_preserves_existing(monkeypatch):
    monkeypatch.setenv("NICEGUI_SCREEN_TEST_PORT", "9999")

    ensure_nicegui_screen_test_port(4567)

    assert os.environ["NICEGUI_SCREEN_TEST_PORT"] == "9999"
