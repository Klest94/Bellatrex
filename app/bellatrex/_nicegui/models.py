"""Framework-independent data passed to the NiceGUI explorer."""

from dataclasses import dataclass, field

import numpy as np


@dataclass(slots=True)
class InteractPoint:
    """One tree represented in an interactive scatter plot."""

    name: str
    pos: np.ndarray
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
