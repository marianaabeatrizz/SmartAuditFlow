# drawio MCP

Generates and edits draw.io (`.drawio`) diagrams from a declarative spec.

## Why it exists

A `.drawio` file is plain mxGraph XML, so a diagram *can* be hand-written. What
makes that painful is the geometry: every shape carries absolute pixel
coordinates, and getting them wrong gives overlapping boxes, edges crossing
labels, or columns that drift out of alignment. This server takes the semantic
description — which box sits in which column and row, what connects to what —
and computes the geometry.

It also refuses to write a file draw.io would open as a blank page, which is
the single most confusing failure mode of hand-written mxGraph.

## Layout

| File | Role |
|---|---|
| `drawio_builder.py` | The library. Importable on its own, no MCP needed. |
| `server.py` | Thin MCP wrapper exposing the library as five tools. |
| `test_server.py` | Drives the server through a real MCP client over stdio. |

The split is deliberate: a diagram can be generated from a plain script or a
test without an MCP client in the loop.

## Tools

| Tool | Does |
|---|---|
| `drawio_vocabulary` | Lists the node/group/edge/label roles and anchors. Call first. |
| `drawio_write` | Builds a spec and writes a `.drawio` file. |
| `drawio_render` | Returns the XML without touching disk. |
| `drawio_validate` | Checks an existing file is well-formed and has no dangling edges. |
| `drawio_read` | Returns a file's XML plus a shape/edge inventory. |

## Spec

```json
{
  "title": "Page name",
  "layout": {"col_width": 180, "row_height": 80,
             "col_gap": 46, "row_gap": 34, "origin": [40, 40]},
  "nodes": [
    {"id": "plan", "label": "Planner", "sub": "delegates",
     "col": 1, "row": 1, "role": "accent"}
  ],
  "groups": [
    {"id": "panel", "label": "JUDGES", "members": ["j1", "j2"], "role": "accent"}
  ],
  "edges": [
    {"from": "plan", "to": "panel", "label": "dispatches",
     "role": "accent", "exit": "right", "entry": "left"}
  ],
  "labels": [{"text": "EXECUTION", "col": 1, "row": 0, "role": "caption"}]
}
```

`col`/`row` are grid coordinates, not pixels. A group is sized to fit its
members, so it stays correct when they move. Wrap several specs in
`{"pages": [...]}` to get one file with a tab per figure.

Roles: `default`, `accent`, `absent` (dashed, for something that does not
exist), `danger`, `ok`, `muted`, `badge`, `badge_accent`.

## Running

Registered in the repo's `.mcp.json`. **Claude Code loads MCP servers at
startup, so it only appears after a restart.** Until then the library works
directly:

```python
import drawio_builder as db
db.write(spec, "docs/diagram.drawio")
```

Tests:

```bash
smart-contract-audit/.venv/Scripts/python.exe tools/drawio-mcp/test_server.py
```

## Known rough edge

The `mcp` SDK is installed in `smart-contract-audit/.venv`, which belongs to
the audit application — this tool borrows it rather than owning a venv. That
couples the two. To decouple: `python -m venv tools/drawio-mcp/.venv`, install
`mcp`, and point `.mcp.json` at it.
