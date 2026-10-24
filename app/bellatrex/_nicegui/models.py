"""Framework-independent data passed to the NiceGUI explorer."""

from dataclasses import dataclass, field
from typing import Any, TypedDict

import numpy as np
from numpy.typing import NDArray


class RenderContext(TypedDict, total=False):
    """Values needed to render a tree selected in the explorer."""

    clf: Any
    sample: Any
    sample_index: object
    max_depth: int | None
    feature_names: list[str] | None


@dataclass(slots=True)
class InteractPoint:
    """One tree represented in an interactive scatter plot."""

    name: str
    pos: NDArray[np.float64]
    color_value: float
    selected: bool = False
    cluster_memb: str | None = None
    value: str | None = None

    def __post_init__(self) -> None:
        self.name = str(self.name)
        self.pos = np.asarray(self.pos, dtype=float)
        if self.pos.shape != (2,):
            raise ValueError(f"Point position must have shape (2,), got {self.pos.shape}")


@dataclass(slots=True)
class InteractPlot:
    """Data and display metadata for one Plotly scatter plot."""

    name: str
    points: list[InteractPoint]
    clustered: bool = False
    title: str = ""
    colorbar_title: str = ""
    colorscale: list[tuple[float, str]] = field(default_factory=list)
    color_min: float = 0.0
    color_max: float = 1.0
    colorbar_tickvals: list[float] | None = None
    colorbar_ticktext: list[str] | None = None
    xlabel: str = "PC1"
    ylabel: str = "PC2"

    def __post_init__(self) -> None:
        self.name = str(self.name)
