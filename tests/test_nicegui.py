import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier

try:
    from nicegui.testing.user_simulation import user_simulation
except ModuleNotFoundError as exc:
    if exc.name != "nicegui.testing.user_simulation":
        raise
    user_simulation = None

try:
    from app.bellatrex._nicegui import explorer
    from app.bellatrex._nicegui.explorer import build_explorer_page, extract_click_selection
    from app.bellatrex._nicegui.models import InteractPlot, InteractPoint
    from app.bellatrex._nicegui.rendering import (
        build_plotly_figure,
        prepare_interactive_plots,
        render_tree_image,
    )
except ImportError:
    from bellatrex._nicegui import explorer
    from bellatrex._nicegui.explorer import build_explorer_page, extract_click_selection
    from bellatrex._nicegui.models import InteractPlot, InteractPoint
    from bellatrex._nicegui.rendering import (
        build_plotly_figure,
        prepare_interactive_plots,
        render_tree_image,
    )

pytestmark = pytest.mark.gui
requires_simulator = pytest.mark.skipif(
    user_simulation is None,
    reason="Install Bellatrex[dev] for NiceGUI user simulation tests",
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


def _prepared_plots() -> list[InteractPlot]:
    return prepare_interactive_plots(
        DummyPlotDataBunch(),
        DummyKMeans([0, 1, 0, 1]),
        DummyInputMethod(),
    )


def test_prepare_interactive_plots_is_in_memory():
    plots = _prepared_plots()

    assert [plot.name for plot in plots] == ["clusters", "predictions"]
    assert all(len(plot.points) == 4 for plot in plots)
    assert {point.name for point in plots[0].points if point.selected} == {"1", "3"}
    assert plots[0].colorbar_ticktext == ["cl.1", "cl.2"]
    assert plots[1].colorbar_title == "Prediction<br>RF: 0.5"


def test_prepare_interactive_plots_uses_loss_for_multi_output():
    data = DummyPlotDataBunch()
    data.rf_pred = np.asarray([0.1, 0.2])

    (plot,) = prepare_interactive_plots(
        data,
        DummyKMeans([0, 1, 0, 1]),
        DummyInputMethod(),
        clusterplots=(False,),
    )

    assert [point.color_value for point in plot.points] == pytest.approx(data.loss)
    assert plot.colorbar_title == "dummy loss"


def test_prepare_interactive_plots_rejects_mismatched_cluster_labels():
    with pytest.raises(ValueError, match="matching lengths"):
        prepare_interactive_plots(
            DummyPlotDataBunch(),
            DummyKMeans([0, 1]),
            DummyInputMethod(),
        )


def test_prepare_interactive_plots_rejects_mismatched_prediction_values():
    data = DummyPlotDataBunch()
    data.pred = np.asarray([0.2, 0.4])

    with pytest.raises(ValueError, match="Prediction values.*matching lengths"):
        prepare_interactive_plots(
            data,
            DummyKMeans([0, 1, 0, 1]),
            DummyInputMethod(),
        )


def test_interact_point_requires_two_coordinates():
    with pytest.raises(ValueError, match=r"shape \(2,\)"):
        InteractPoint(name="0", pos=np.asarray([0.1]), color_value=0.5)


def test_plotly_figure_contains_both_marker_groups_and_its_colorbar():
    cluster_plot = _prepared_plots()[0]

    figure = build_plotly_figure(cluster_plot)

    assert len(figure.data) == 2
    assert list(figure.data[0].customdata) == ["0", "2"]
    assert list(figure.data[1].customdata) == ["1", "3"]
    assert figure.data[0].marker.symbol == "circle"
    assert figure.data[1].marker.symbol == "star"
    assert tuple(figure.layout.coloraxis.colorbar.ticktext) == ("cl.1", "cl.2")


def test_render_tree_image_uses_a_headless_canvas():
    features = pd.DataFrame(
        [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]],
        columns=["left", "right"],
    )
    forest = RandomForestClassifier(n_estimators=1, max_depth=2, random_state=0).fit(
        features, [0, 0, 1, 1]
    )

    image, title = render_tree_image(
        "0",
        {"clf": forest, "sample": features.iloc[[0]], "sample_index": 7},
    )

    assert image.startswith(b"\x89PNG\r\n\x1a\n")
    assert title == "Tree 0 for sample index 7"


@pytest.mark.parametrize(
    ("args", "expected"),
    [
        ({"tree_name": "41", "text": "Tree 41"}, ("41", "Tree 41")),
        ({"tree_name": [7]}, (None, "Select a tree point to inspect it")),
        ({"tree_name": "not-a-tree"}, (None, "Select a tree point to inspect it")),
        (None, (None, "Select a tree point to inspect it")),
        ({"tree_name": 7}, (None, "Select a tree point to inspect it")),
        ({"tree_name": -1}, (None, "Select a tree point to inspect it")),
        ({"tree_name": 1.5}, (None, "Select a tree point to inspect it")),
        ({"tree_name": float("inf")}, (None, "Select a tree point to inspect it")),
        ({"tree_name": []}, (None, "Select a tree point to inspect it")),
    ],
)
def test_extract_click_selection(args, expected):
    assert extract_click_selection(args) == expected


@requires_simulator
@pytest.mark.anyio
async def test_explorer_page_opens_and_reuses_an_in_page_tree_dialog():
    render_calls: list[str] = []
    rule_calls: list[str] = []

    def render_tree(tree_name: str, _context: dict) -> tuple[bytes, str]:
        render_calls.append(tree_name)
        return b"png", f"Tree {tree_name} for sample index 7"

    async with user_simulation(
        root=lambda: build_explorer_page(
            _prepared_plots(),
            {"sample_index": 7},
            tree_renderer=render_tree,
            rule_printer=lambda tree_name, _context: rule_calls.append(tree_name),
        )
    ) as user:
        await user.open("/")
        await user.should_see("Bellatrex visual explanation, sample 7")

        user.find(marker="explorer-plot-0").trigger(
            "plotly_click",
            {"tree_name": "1", "text": "Tree 1 | Cluster 2"},
        )
        await user.should_see("Tree 1 for sample index 7")
        await user.should_see(marker="tree-image")

        user.find(marker="close-tree").click()
        user.find(marker="explorer-plot-0").trigger(
            "plotly_click",
            {"tree_name": "1", "text": "Tree 1 | Cluster 2"},
        )

    assert render_calls == ["1"]
    assert rule_calls == ["1", "1"]


def test_run_explorer_uses_browser_mode(monkeypatch):
    captured = {}

    def fake_run(root, **kwargs):
        captured["root"] = root
        captured.update(kwargs)

    monkeypatch.setattr(explorer.ui, "run", fake_run)

    explorer.run_explorer(_prepared_plots(), {"sample_index": 3}, show=False, port=8123)

    assert callable(captured["root"])
    assert captured["host"] == "127.0.0.1"
    assert captured["native"] is False
    assert captured["reload"] is False
    assert captured["show"] is False
    assert captured["port"] == 8123


def test_prepare_interactive_plots_rejects_mismatched_projection_rows():
    data = DummyPlotDataBunch()
    data.proj_data = data.proj_data[:2]
    with pytest.raises(ValueError, match="Projection rows.*matching lengths"):
        prepare_interactive_plots(data, DummyKMeans([0, 1, 0, 1]), DummyInputMethod())


@requires_simulator
@pytest.mark.anyio
async def test_explorer_ignores_unknown_tree_and_recovers_after_render_failure():
    render_calls = []

    def render_tree(tree_name, _context):
        render_calls.append(tree_name)
        if len(render_calls) == 1:
            raise ValueError("render failed")
        return b"png", f"Tree {tree_name}"

    async with user_simulation(
        root=lambda: build_explorer_page(
            _prepared_plots(),
            {"sample_index": 7},
            tree_renderer=render_tree,
            rule_printer=lambda *_: None,
        )
    ) as user:
        await user.open("/")
        plot = user.find(marker="explorer-plot-0")
        plot.trigger("plotly_click", {"tree_name": "999"})
        assert render_calls == []
        plot.trigger("plotly_click", {"tree_name": "1"})
        await user.should_see("Could not render tree 1: render failed")
        await user.should_not_see(marker="tree-image")
        plot.trigger("plotly_click", {"tree_name": "1"})
        await user.should_see(marker="tree-image")
        await user.should_see("Opened Tree 1")
    assert render_calls == ["1", "1"]


@requires_simulator
@pytest.mark.anyio
async def test_explorer_cache_is_scoped_to_each_page():
    render_calls = []

    def render_tree(tree_name, _context):
        render_calls.append(tree_name)
        return b"png", f"Tree {tree_name}"

    async with user_simulation(
        root=lambda: build_explorer_page(
            _prepared_plots(),
            {"sample_index": 7},
            tree_renderer=render_tree,
            rule_printer=lambda *_: None,
        )
    ) as user:
        for _ in range(2):
            await user.open("/")
            user.find(marker="explorer-plot-0").trigger("plotly_click", {"tree_name": "1"})
            await user.should_see("Opened Tree 1")
    assert render_calls == ["1", "1"]
