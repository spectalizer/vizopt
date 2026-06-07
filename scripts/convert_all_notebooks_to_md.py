"""Convert all notebooks in the examples/ folder to Markdown.

Usage:
    uv run python scripts/convert_all_notebooks.py
    uv run python scripts/convert_all_notebooks.py --execute
"""

import argparse
import re
import sys
from pathlib import Path

# Allow importing nb_to_md from the same directory
sys.path.insert(0, str(Path(__file__).parent))
from nb_to_md import notebook_to_md


def _extract_title_and_desc(md_path: Path) -> tuple[str, str]:
    """Return (title, first_paragraph) from a markdown file."""
    text = md_path.read_text(encoding="utf-8")
    title = ""
    desc = ""
    for line in text.splitlines():
        stripped = line.strip()
        if not title and re.match(r"^#{1,3} ", stripped):
            title = re.sub(r"^#{1,3} ", "", stripped)
        elif title and stripped and not stripped.startswith("#") and not stripped.startswith("```"):
            desc = stripped
            break
    return title or md_path.stem, desc


def _write_index(docs_dir: Path, md_files: list[Path]) -> None:
    """Write docs/examples/index.md listing all example pages."""
    lines = ["# Examples\n"]
    for md_path in md_files:
        title, desc = _extract_title_and_desc(md_path)
        entry = f"- [{title}]({md_path.name})"
        if desc:
            entry += f" — {desc}"
        lines.append(entry)
    index_path = docs_dir / "index.md"
    index_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Written: {index_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Convert all example notebooks to Markdown."
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Execute each notebook before converting (requires nbconvert)",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).parent.parent
    examples_dir = repo_root / "notebooks" / "examples"
    docs_dir = repo_root / "docs" / "examples"

    notebooks = sorted(examples_dir.glob("*.ipynb"))
    if not notebooks:
        print("No notebooks found in examples/")
        return

    md_files = []
    for nb_path in notebooks:
        out_path = docs_dir / nb_path.with_suffix(".md").name
        notebook_to_md(nb_path, out_path, execute=args.execute)
        md_files.append(out_path)

    _write_index(docs_dir, md_files)


if __name__ == "__main__":
    main()
