"""Convert a Jupyter notebook to a Markdown page for Zensical (mkdocs-compatible).

Usage:
    python scripts/nb_to_md.py examples/nested_circles.ipynb docs/examples/nested-circles.md
    python scripts/nb_to_md.py --execute examples/nested_circles.ipynb docs/examples/nested-circles.md

Image outputs are saved to a sibling `images/` directory next to the output file
and referenced with relative paths in the markdown.
"""

import argparse
import base64
import re
import sys
from pathlib import Path


def _save_image(data: dict, image_dir: Path, image_prefix: str, idx: int) -> str | None:
    """Save an image output to disk and return a markdown image tag, or None.

        Prefers SVG, falls back to PNG then JPEG.

        `%config InlineBackend.figure_formats = ['svg']` at the top of the notebook
        should be used to generate SVG outputs.

        Args:
            data: The output data dict from a notebook cell.
            image_dir: Directory to save images to.
            image_prefix: Prefix for image filenames.
            idx: Index for image filenames.

        Returns:
            Markdown image tag, e.g.:
            `![output](images/nested-circles_0.svg)
    `
    """
    if "image/svg+xml" in data:
        svg = data["image/svg+xml"]
        if isinstance(svg, list):
            svg = "".join(svg)
        image_dir.mkdir(parents=True, exist_ok=True)
        filename = f"{image_prefix}_{idx}.svg"
        (image_dir / filename).write_text(svg, encoding="utf-8")
        return f"![output](images/{filename})"

    img_data = data.get("image/png") or data.get("image/jpeg")
    if not img_data:
        return None
    ext = "png" if "image/png" in data else "jpg"
    img_bytes = base64.b64decode(img_data)
    image_dir.mkdir(parents=True, exist_ok=True)
    filename = f"{image_prefix}_{idx}.{ext}"
    (image_dir / filename).write_bytes(img_bytes)
    return f"![output](images/{filename})"


def cell_to_md(
    cell: dict, image_dir: Path, image_prefix: str, image_counter: list
) -> str:
    """Convert a single notebook cell to a markdown string."""
    cell_type = cell["cell_type"]
    source = "".join(cell["source"])

    if cell_type == "markdown":
        return source.strip()

    if cell_type == "code":
        if not source.strip():
            return ""
        parts = [f"```python\n{source.rstrip()}\n```"]

        for output in cell.get("outputs", []):
            output_type = output.get("output_type", "")

            if output_type in ("execute_result", "display_data"):
                text = output.get("text", [])
                if isinstance(text, list):
                    text = "".join(text)
                if text.strip():
                    parts.append(f"```\n{text.rstrip()}\n```")

            img_tag = _save_image(
                output.get("data", {}), image_dir, image_prefix, image_counter[0]
            )
            if img_tag:
                image_counter[0] += 1
                parts.append(img_tag)

        return "\n\n".join(parts)

    return ""  # ignore 'raw' cells


def execute_notebook(nb_path: Path) -> dict:
    """Execute a notebook and return the resulting notebook dict with outputs."""
    import nbformat
    from nbconvert.preprocessors import ExecutePreprocessor

    nb = nbformat.read(nb_path, as_version=4)
    ep = ExecutePreprocessor(timeout=600, kernel_name="python3")
    ep.preprocess(nb, {"metadata": {"path": str(nb_path.parent)}})
    return nb


def notebook_to_md(nb_path: Path, out_path: Path, execute: bool = False) -> None:
    """Convert a notebook file to a Markdown file."""
    if execute:
        print(f"Executing: {nb_path}")
        nb = execute_notebook(nb_path)
        cells = nb.cells
    else:
        import json

        nb = json.loads(nb_path.read_text(encoding="utf-8"))
        cells = nb.get("cells", [])

    image_dir = out_path.parent / "images"
    image_prefix = re.sub(r"[^a-z0-9_]", "_", nb_path.stem.lower())
    image_counter = [0]

    blocks = []
    for cell in cells:
        md = cell_to_md(dict(cell), image_dir, image_prefix, image_counter)
        if md:
            blocks.append(md)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n\n".join(blocks) + "\n", encoding="utf-8")
    print(f"Written: {out_path}")
    if image_counter[0]:
        print(f"Saved {image_counter[0]} image(s) to {image_dir}/")


def main():
    """Entry point for the nb_to_md script."""
    parser = argparse.ArgumentParser(
        description="Convert a Jupyter notebook to Markdown for Zensical."
    )
    parser.add_argument("notebook", type=Path, help="Path to the .ipynb file")
    parser.add_argument("output", type=Path, help="Path for the output .md file")
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Execute the notebook before converting (requires nbconvert)",
    )
    args = parser.parse_args()

    if not args.notebook.exists():
        print(f"Error: notebook not found: {args.notebook}", file=sys.stderr)
        sys.exit(1)

    notebook_to_md(args.notebook, args.output, execute=args.execute)


if __name__ == "__main__":
    main()
