"""Check that preparing plots does not load the web page library."""

import subprocess
import sys


def test_plot_modules_do_not_load_nicegui():
    subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; "
            "import bellatrex._nicegui.models; "
            "import bellatrex._nicegui.rendering; "
            "assert 'nicegui' not in sys.modules",
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=60,
    )
