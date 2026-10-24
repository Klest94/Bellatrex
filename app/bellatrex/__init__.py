# -*- coding: utf-8 -*-
"""
bellatrex package initializer.
@author: Klest Dedja
"""

try:
    from .__version__ import __version__
except ImportError:
    __version__ = "unknown"


# Lazy imports keep import-time overhead low and avoid issues during editable installs.
def __getattr__(name):
    if name == "BellatrexExplain":
        from .bellatrex_explain import BellatrexExplain  # pylint: disable=import-outside-toplevel

        return BellatrexExplain
    if name == "pack_trained_ensemble":
        from .wrapper_class import pack_trained_ensemble  # pylint: disable=import-outside-toplevel

        return pack_trained_ensemble
    if name == "predict_helper":
        from .utilities import predict_helper  # pylint: disable=import-outside-toplevel

        return predict_helper
    if name == "TaskType":
        from .utilities import TaskType  # pylint: disable=import-outside-toplevel

        return TaskType
    raise AttributeError(f"module {__name__} has no attribute {name}")


__all__ = ["BellatrexExplain", "pack_trained_ensemble", "predict_helper", "TaskType"]
