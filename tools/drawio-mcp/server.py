"""MCP server exposing the draw.io builder as tools.

Run over stdio; register it in `.mcp.json` (see README.md in this folder).

The server is a thin shell: all the work lives in ``drawio_builder``, which is
importable on its own. That split is deliberate -- a diagram can be generated
from a plain script or a test without an MCP client in the loop, and the tools
here stay small enough to read.
"""
from __future__ import annotations

import json
import os
import sys
from typing import Any, Dict, Optional

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from mcp.server.mcpserver import MCPServer

import drawio_builder as db

mcp = MCPServer(
    name="drawio",
    instructions=(
        "Creates and edits draw.io (.drawio) diagrams from a declarative spec. "
        "Describe a diagram semantically -- which box sits in which column and "
        "row, what connects to what -- and the server computes the pixel "
        "geometry and emits valid mxGraph XML.\n\n"
        "Call `drawio_vocabulary` first to see the available roles and anchors. "
        "Use `drawio_write` to create or overwrite a file, `drawio_validate` to "
        "check an existing one, and `drawio_read` to get a file back as XML "
        "before editing it.\n\n"
        "Files are written as uncompressed XML so they diff cleanly in git."
    ),
)


def _ok(payload: Dict[str, Any]) -> str:
    return json.dumps({"ok": True, **payload}, ensure_ascii=False, indent=2)


def _err(message: str, hint: str = "") -> str:
    out: Dict[str, Any] = {"ok": False, "error": message}
    if hint:
        out["hint"] = hint
    return json.dumps(out, ensure_ascii=False, indent=2)


@mcp.tool(
    title="Describe the diagram vocabulary",
    description=(
        "Lists the node, group, edge and label roles a spec may use, plus the "
        "edge anchor names. Call this before writing a spec: an unknown role is "
        "rejected rather than silently falling back to a default."
    ),
)
def drawio_vocabulary() -> str:
    roles = db.describe_roles()
    return _ok({
        "roles": roles,
        "spec_shape": {
            "title": "page name",
            "layout": {
                "col_width": 180, "row_height": 80,
                "col_gap": 46, "row_gap": 34, "origin": [40, 40],
            },
            "nodes": [{
                "id": "unique", "label": "bold line", "sub": "smaller line",
                "col": 0, "row": 0, "role": "default",
                "colspan": 1, "rowspan": 1,
                "dx": 0, "dy": 0, "width": None, "height": None,
            }],
            "groups": [{
                "id": "unique", "label": "container title",
                "members": ["node ids"], "role": "default",
                "pad": [14, 30, 14, 14],
            }],
            "edges": [{
                "from": "node id", "to": "node id", "label": "",
                "role": "default", "exit": "right", "entry": "left",
                "waypoints": [[320, 400]],
            }],
            "labels": [{
                "text": "free text", "col": 0, "row": 0,
                "role": "caption", "dx": 0, "dy": 0,
            }],
        },
        "notes": [
            "col/row are grid coordinates, not pixels; the server converts them.",
            "A group is sized to fit its members, so it stays correct when they move.",
            "Dangling edge endpoints are rejected: draw.io opens such a file as a "
            "blank page with no error, which is painful to debug.",
        ],
    })


@mcp.tool(
    title="Write a .drawio file",
    description=(
        "Builds a diagram from a spec and writes it to disk as uncompressed "
        "mxGraph XML. The file is validated before writing, so a malformed "
        "spec fails loudly instead of producing a file that opens blank. "
        "Overwrites an existing file at that path."
    ),
)
def drawio_write(path: str, spec: Dict[str, Any]) -> str:
    """Create or overwrite a .drawio file.

    Args:
        path: destination path; `.drawio` is appended when missing.
        spec: the diagram spec. See `drawio_vocabulary` for its shape.
    """
    if not path:
        return _err("path is required")
    if not path.lower().endswith((".drawio", ".xml")):
        path += ".drawio"
    try:
        result = db.write(spec, path)
    except db.SpecError as exc:
        return _err(str(exc), "call drawio_vocabulary to see valid roles and fields")
    except OSError as exc:
        return _err(f"could not write {path}: {exc}")
    return _ok(result)


@mcp.tool(
    title="Render a spec without writing",
    description=(
        "Builds the mxGraph XML for a spec and returns it, touching no files. "
        "Use to preview or to pipe the XML somewhere else."
    ),
)
def drawio_render(spec: Dict[str, Any]) -> str:
    try:
        xml = db.render(spec)
    except db.SpecError as exc:
        return _err(str(exc), "call drawio_vocabulary to see valid roles and fields")
    return _ok({"xml": xml, "bytes": len(xml.encode("utf-8"))})


@mcp.tool(
    title="Validate a .drawio file",
    description=(
        "Checks that a file on disk is well-formed mxGraph XML with the "
        "mandatory root cells and no edges pointing at missing shapes."
    ),
)
def drawio_validate(path: str) -> str:
    if not os.path.exists(path):
        return _err(f"no such file: {path}")
    try:
        return _ok(db.validate_file(path))
    except db.SpecError as exc:
        return _err(str(exc))
    except OSError as exc:
        return _err(f"could not read {path}: {exc}")


@mcp.tool(
    title="Read a .drawio file",
    description=(
        "Returns the raw XML of an existing diagram, plus a shape and edge "
        "inventory. Read a file before editing it so the edit builds on what "
        "is actually there."
    ),
)
def drawio_read(path: str, include_xml: bool = True) -> str:
    if not os.path.exists(path):
        return _err(f"no such file: {path}")
    try:
        with open(path, encoding="utf-8") as fh:
            xml = fh.read()
        summary = db.validate_xml(xml)
    except db.SpecError as exc:
        return _err(f"{path} is not a usable draw.io file: {exc}")
    except OSError as exc:
        return _err(f"could not read {path}: {exc}")

    payload: Dict[str, Any] = {"path": os.path.abspath(path), **summary}
    if include_xml:
        payload["xml"] = xml
    return _ok(payload)


if __name__ == "__main__":
    mcp.run(transport="stdio")
