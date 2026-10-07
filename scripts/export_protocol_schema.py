"""Export the JSON Schema of the live-server protocol for the frontend.

The schema covers every WebSocket message and the `Scene` they carry. The
frontend generates its TypeScript types from it (`npm run codegen` in
`frontend/` runs this script, then json-schema-to-typescript), so re-run it
whenever `src/vizopt/scene.py` or `src/vizopt/server/protocol.py` change;
`tests/test_server.py` fails while the committed schema is out of date.

Usage:
    uv run python scripts/export_protocol_schema.py
    uv run python scripts/export_protocol_schema.py --out path/to/protocol.schema.json
"""

import argparse
import json
from pathlib import Path

from vizopt.server.protocol import protocol_json_schema

DEFAULT_OUT = (
    Path(__file__).resolve().parent.parent
    / "frontend"
    / "src"
    / "protocol"
    / "protocol.schema.json"
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(protocol_json_schema(), indent=2) + "\n"
    args.out.write_text(text, encoding="utf-8")
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
