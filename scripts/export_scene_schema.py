"""Export the JSON Schema of vizopt.scene.Scene for the frontend.

The frontend generates its TypeScript types from this file (with
json-schema-to-typescript), so re-run this script whenever the models in
`src/vizopt/scene.py` change.

Usage:
    uv run python scripts/export_scene_schema.py
    uv run python scripts/export_scene_schema.py --out path/to/scene.schema.json
"""

import argparse
import json
from pathlib import Path

from vizopt.scene import Scene

DEFAULT_OUT = (
    Path(__file__).resolve().parent.parent
    / "frontend"
    / "src"
    / "protocol"
    / "scene.schema.json"
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    schema = Scene.model_json_schema()
    args.out.write_text(json.dumps(schema, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
