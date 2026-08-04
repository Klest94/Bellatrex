"""Data models used by the NiceGUI interactive explorer."""

from dataclasses import dataclass

import numpy as np


@dataclass(slots=True)
class InteractPoint:
    """Container for one clickable point in the NiceGUI scatter plot."""

    name: str
    pos: np.ndarray
    color: list[float]
    size: float
    shape: str
    cluster_memb: str | None = None
    value: str | None = None

    def __post_init__(self) -> None:
        self.name = str(self.name)


@dataclass(slots=True)
class InteractPlot:
    """Data prepared for one Plotly scatter plus its companion colorbar."""

    name: str
    points: list[InteractPoint]
    clustered: bool = False
    xlabel: str = "PC1"
    ylabel: str = "PC2"
    sample_idx: object = None

    def __post_init__(self) -> None:
        self.name = str(self.name)
