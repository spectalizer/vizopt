"""Convert all notebooks to Python scripts and run pyright on them.

Usage:
    uv run python scripts/convert_all_notebooks_to_py.py
    uv run python scripts/convert_all_notebooks_to_py.py --no-cleanup
"""

import argparse
import subprocess
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(
        description="Convert notebooks to .py and run pyright."
    )
    parser.add_argument(
        "--no-cleanup",
        action="store_true",
        help="Keep the generated .py files after running pyright.",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).parent.parent
    notebooks = sorted((repo_root / "notebooks").rglob("*.ipynb"))
    if not notebooks:
        print("No notebooks found.")
        return

    py_files = []
    for nb_path in notebooks:
        subprocess.run(
            ["jupyter", "nbconvert", "--to", "script", str(nb_path)],
            check=True,
        )
        py_files.append(nb_path.with_suffix(".py"))

    result = subprocess.run(["pyright"] + [str(p) for p in py_files])

    if not args.no_cleanup:
        for p in py_files:
            p.unlink(missing_ok=True)

    sys.exit(result.returncode)


if __name__ == "__main__":
    main()
