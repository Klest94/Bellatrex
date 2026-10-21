"""Execute both tutorials against an installed wheel, outside the source checkout.

Install the wheel, nbclient and ipykernel first, then run this script.
"""

import os
import subprocess
import sys
import tempfile
from pathlib import Path

import bellatrex
import nbformat
from nbclient import NotebookClient


def check_rules(directory: Path) -> None:
    """Require a real exported explanation, rather than just successful imports."""
    if not any(
        "Bellatrex prediction:" in path.read_text(encoding="utf-8")
        for path in directory.glob("*.txt")
    ):
        raise AssertionError(f"Tutorial did not export an explanation in {directory}")


def main() -> None:
    repository = Path(__file__).resolve().parents[1]
    if Path(bellatrex.__file__).resolve().is_relative_to(repository):
        raise AssertionError("Install the built wheel before running this smoke check.")

    with tempfile.TemporaryDirectory(prefix="bellatrex-tutorial-") as temporary:
        root = Path(temporary)
        env = os.environ.copy()
        env.pop("PYTHONPATH", None)
        env.update(MPLBACKEND="Agg", BELLATREX_TUTORIAL_HEADLESS="1")

        script_work = root / "script"
        script_work.mkdir()
        env["BELLATREX_EXPLAIN_DIR"] = str(script_work / "rules")
        subprocess.run(
            [sys.executable, str(repository / "tutorial.py"), "--headless"],
            cwd=script_work,
            env=env,
            check=True,
            timeout=180,
        )
        check_rules(script_work / "rules")
        print("PASS: tutorial.py trained a forest and exported an explanation.")

        notebook_work = root / "notebook"
        notebook_work.mkdir()
        env["BELLATREX_EXPLAIN_DIR"] = str(notebook_work / "rules")
        notebook = nbformat.read(repository / "tutorial.ipynb", as_version=4)
        client = NotebookClient(
            notebook,
            timeout=180,
            kernel_name="python3",
            resources={"metadata": {"path": str(notebook_work)}},
        )
        # Use this interpreter and its installed wheel, even if another environment
        # registered the machine's default python3 kernelspec.
        client.km = client.create_kernel_manager()
        client.km.kernel_spec.argv = [
            sys.executable,
            "-m",
            "ipykernel_launcher",
            "-f",
            "{connection_file}",
        ]
        client.execute(env=env)
        check_rules(notebook_work / "rules")
        print("PASS: tutorial.ipynb executed in a fresh kernel and exported an explanation.")


if __name__ == "__main__":
    main()
