import os

import matplotlib
import pytest

# Force Matplotlib to use the "Agg" backend, which is a non-interactive backend suitable for testing environments without a display server.
os.environ.setdefault("MPLBACKEND", "Agg")
matplotlib.use("Agg", force=True)


@pytest.fixture(scope="module")
def anyio_backend():
    """NiceGUI's semantic test user runs on asyncio."""
    return "asyncio"
