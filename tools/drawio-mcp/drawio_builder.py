"""Build draw.io (.drawio / mxGraph) files from a declarative diagram spec.

A .drawio file is plain mxGraph XML, so it can be written directly -- but the
XML carries pixel geometry for every shape, which is tedious and easy to get
subtly wrong (overlapping boxes, edges that cross labels, columns that drift
out of alignment). This module takes the semantic description -- which box sits
in which column and row, what connects to what -- and computes the geometry.

Output is **uncompressed** XML. draw.io also accepts a deflate+base64 payload,
but uncompressed diffs cleanly in git, which matters more here.

The spec:

    {
      "title": "Page name",
      "layout": {"col_width":180,"row_height":80,"col_gap":46,"row_gap":34,
                 "origin":[40,40]},
      "nodes": [
        {"id":"a","label":"Planner","sub":"delegates","col":1,"row":2,
         "role":"accent","colspan":1,"rowspan":1}
      ],
      "groups": [
        {"id":"panel","label":"Judging panel","members":["j1","j2"],
         "role":"accent","pad":[14,30,14,14]}
      ],
      "edges": [
        {"from":"a","to":"b","label":"dispatches","role":"accent",
         "style":"orthogonal","exit":"right","entry":"left"}
      ],
      "labels": [
        {"text":"STAGE","col":1,"row":0,"role":"caption"}
      ]
    }

Grid coordinates are integers; the builder turns them into pixels. Nodes may
span columns/rows. Groups are drawn as a container sized to fit their members,
so a panel of judges stays correct when its members move.
"""
from __future__ import annotations

import html
import uuid
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

# --- palette -----------------------------------------------------------------
# Kept deliberately small and named by role rather than by colour, so a diagram
# can be restyled without touching the spec.
PALETTE = {
    "ink": "#14191A",
    "soft": "#4C5758",
    "faint": "#8A9495",
    "line": "#C3CBCA",
    "paper": "#FFFFFF",
    "accent": "#0E6E6E",
    "accent_fill": "#DCEBE9",
    "danger": "#9B2C22",
    "danger_fill": "#F7E6E3",
    "ok": "#1C6B45",
    "ok_fill": "#DFEDE5",
    "warn": "#8A5416",
    "warn_fill": "#F6E9D8",
}

NODE_ROLES: Dict[str, str] = {
    # the ordinary box
    "default": (
        "rounded=1;whiteSpace=wrap;html=1;arcSize=8;"
        f"fillColor={PALETTE['paper']};strokeColor={PALETTE['soft']};"
        f"fontColor={PALETTE['ink']};strokeWidth=1.4;"
        "verticalAlign=middle;align=center;fontSize=13;spacing=6;"
    ),
    # the thing being added or argued for
    "accent": (
        "rounded=1;whiteSpace=wrap;html=1;arcSize=8;"
        f"fillColor={PALETTE['accent_fill']};strokeColor={PALETTE['accent']};"
        f"fontColor={PALETTE['accent']};strokeWidth=2.4;"
        "verticalAlign=middle;align=center;fontSize=13;spacing=6;"
    ),
    # something absent / removed: dashed outline, no fill
    "absent": (
        "rounded=1;whiteSpace=wrap;html=1;arcSize=8;dashed=1;dashPattern=6 4;"
        f"fillColor=none;strokeColor={PALETTE['danger']};"
        f"fontColor={PALETTE['danger']};strokeWidth=1.4;"
        "verticalAlign=middle;align=center;fontSize=13;spacing=6;"
    ),
    "danger": (
        "rounded=1;whiteSpace=wrap;html=1;arcSize=8;"
        f"fillColor={PALETTE['danger_fill']};strokeColor={PALETTE['danger']};"
        f"fontColor={PALETTE['danger']};strokeWidth=1.4;"
        "verticalAlign=middle;align=center;fontSize=13;spacing=6;"
    ),
    "ok": (
        "rounded=1;whiteSpace=wrap;html=1;arcSize=8;"
        f"fillColor={PALETTE['ok_fill']};strokeColor={PALETTE['ok']};"
        f"fontColor={PALETTE['ok']};strokeWidth=1.8;"
        "verticalAlign=middle;align=center;fontSize=13;spacing=6;"
    ),
    # de-emphasised: the "before" state in a comparison
    "muted": (
        "rounded=1;whiteSpace=wrap;html=1;arcSize=8;"
        f"fillColor={PALETTE['paper']};strokeColor={PALETTE['line']};"
        f"fontColor={PALETTE['faint']};strokeWidth=1.2;"
        "verticalAlign=middle;align=center;fontSize=13;spacing=6;"
    ),
    # a small pill, for change markers between two rows
    "badge": (
        "rounded=1;whiteSpace=wrap;html=1;arcSize=50;"
        f"fillColor={PALETTE['paper']};strokeColor={PALETTE['soft']};"
        f"fontColor={PALETTE['soft']};strokeWidth=1;"
        "verticalAlign=middle;align=center;fontSize=10;"
    ),
    "badge_accent": (
        "rounded=1;whiteSpace=wrap;html=1;arcSize=50;"
        f"fillColor={PALETTE['accent_fill']};strokeColor={PALETTE['accent']};"
        f"fontColor={PALETTE['accent']};strokeWidth=1.6;"
        "verticalAlign=middle;align=center;fontSize=10;fontStyle=1;"
    ),
}

GROUP_ROLES: Dict[str, str] = {
    "default": (
        "rounded=1;whiteSpace=wrap;html=1;arcSize=6;"
        f"fillColor=none;strokeColor={PALETTE['soft']};strokeWidth=1.6;"
        f"fontColor={PALETTE['soft']};verticalAlign=top;align=center;"
        "fontSize=11;fontStyle=1;spacingTop=4;container=0;"
    ),
    "accent": (
        "rounded=1;whiteSpace=wrap;html=1;arcSize=6;"
        f"fillColor={PALETTE['accent_fill']};strokeColor={PALETTE['accent']};"
        f"strokeWidth=2.4;fontColor={PALETTE['accent']};"
        "verticalAlign=top;align=center;fontSize=11;fontStyle=1;"
        "spacingTop=4;container=0;"
    ),
    "absent": (
        "rounded=1;whiteSpace=wrap;html=1;arcSize=6;dashed=1;dashPattern=6 4;"
        f"fillColor=none;strokeColor={PALETTE['danger']};strokeWidth=1.4;"
        f"fontColor={PALETTE['danger']};verticalAlign=top;align=center;"
        "fontSize=11;fontStyle=1;spacingTop=4;container=0;"
    ),
}

EDGE_ROLES: Dict[str, str] = {
    "default": (
        "edgeStyle=orthogonalEdgeStyle;rounded=1;html=1;jettySize=auto;"
        f"strokeColor={PALETTE['soft']};strokeWidth=1.4;"
        f"fontColor={PALETTE['soft']};fontSize=11;endArrow=blockThin;endFill=1;"
    ),
    "accent": (
        "edgeStyle=orthogonalEdgeStyle;rounded=1;html=1;jettySize=auto;"
        f"strokeColor={PALETTE['accent']};strokeWidth=1.8;"
        f"fontColor={PALETTE['accent']};fontSize=11;endArrow=blockThin;endFill=1;"
    ),
    "straight": (
        "edgeStyle=none;rounded=0;html=1;"
        f"strokeColor={PALETTE['soft']};strokeWidth=1.3;"
        f"fontColor={PALETTE['soft']};fontSize=11;endArrow=blockThin;endFill=1;"
    ),
    "straight_accent": (
        "edgeStyle=none;rounded=0;html=1;"
        f"strokeColor={PALETTE['accent']};strokeWidth=1.5;"
        f"fontColor={PALETTE['accent']};fontSize=11;endArrow=blockThin;endFill=1;"
    ),
    "dashed": (
        "edgeStyle=orthogonalEdgeStyle;rounded=1;html=1;dashed=1;dashPattern=6 4;"
        f"strokeColor={PALETTE['soft']};strokeWidth=1.4;"
        f"fontColor={PALETTE['soft']};fontSize=11;endArrow=blockThin;endFill=1;"
    ),
    "muted": (
        "edgeStyle=orthogonalEdgeStyle;rounded=1;html=1;"
        f"strokeColor={PALETTE['line']};strokeWidth=1.2;"
        f"fontColor={PALETTE['faint']};fontSize=11;endArrow=blockThin;endFill=1;"
    ),
    "danger_dashed": (
        "edgeStyle=none;rounded=0;html=1;dashed=1;dashPattern=6 4;"
        f"strokeColor={PALETTE['danger']};strokeWidth=1.3;"
        f"fontColor={PALETTE['danger']};fontSize=11;endArrow=blockThin;endFill=1;"
    ),
    "ok": (
        "edgeStyle=none;rounded=0;html=1;"
        f"strokeColor={PALETTE['ok']};strokeWidth=1.9;"
        f"fontColor={PALETTE['ok']};fontSize=11;endArrow=blockThin;endFill=1;"
    ),
}

LABEL_ROLES: Dict[str, str] = {
    "caption": (
        "text;html=1;align=center;verticalAlign=middle;"
        f"fontColor={PALETTE['faint']};fontSize=11;fontStyle=1;letterSpacing=1;"
    ),
    "note": (
        "text;html=1;align=left;verticalAlign=middle;"
        f"fontColor={PALETTE['faint']};fontSize=11;"
    ),
    "note_accent": (
        "text;html=1;align=left;verticalAlign=middle;"
        f"fontColor={PALETTE['accent']};fontSize=11;fontStyle=1;"
    ),
    "title": (
        "text;html=1;align=left;verticalAlign=middle;"
        f"fontColor={PALETTE['ink']};fontSize=16;fontStyle=1;"
    ),
    "rowlabel": (
        "text;html=1;align=left;verticalAlign=middle;"
        f"fontColor={PALETTE['faint']};fontSize=11;fontStyle=1;letterSpacing=1;"
    ),
    "rowlabel_accent": (
        "text;html=1;align=left;verticalAlign=middle;"
        f"fontColor={PALETTE['accent']};fontSize=11;fontStyle=1;letterSpacing=1;"
    ),
}

# Anchor points for explicit edge attachment, as draw.io exit/entry fractions.
ANCHORS: Dict[str, Tuple[float, float]] = {
    "left": (0.0, 0.5),
    "right": (1.0, 0.5),
    "top": (0.5, 0.0),
    "bottom": (0.5, 1.0),
    "topleft": (0.0, 0.0),
    "topright": (1.0, 0.0),
    "bottomleft": (0.0, 1.0),
    "bottomright": (1.0, 1.0),
}


class SpecError(ValueError):
    """The spec is malformed. Raised with the offending id where known."""


@dataclass
class Layout:
    col_width: int = 180
    row_height: int = 80
    col_gap: int = 46
    row_gap: int = 34
    origin: Tuple[int, int] = (40, 40)

    def x(self, col: int) -> int:
        return self.origin[0] + col * (self.col_width + self.col_gap)

    def y(self, row: int) -> int:
        return self.origin[1] + row * (self.row_height + self.row_gap)

    def w(self, colspan: int = 1) -> int:
        return colspan * self.col_width + (colspan - 1) * self.col_gap

    def h(self, rowspan: int = 1) -> int:
        return rowspan * self.row_height + (rowspan - 1) * self.row_gap


@dataclass
class Box:
    """A placed rectangle, in pixels."""
    id: str
    x: int
    y: int
    w: int
    h: int
    style: str
    value: str
    parent: str = "1"


@dataclass
class Diagram:
    title: str = "Page-1"
    layout: Layout = field(default_factory=Layout)
    boxes: List[Box] = field(default_factory=list)
    edges: List[Dict[str, Any]] = field(default_factory=list)


def _label(label: str = "", sub: str = "", bold_label: bool = True) -> str:
    """Compose an mxCell value, escaping text but allowing our own markup.

    draw.io renders the value as HTML when the style carries html=1, so the
    caller's text must be escaped first or a stray '&' silently breaks the file.
    """
    parts = []
    if label:
        esc = html.escape(label, quote=False)
        parts.append(f"<b>{esc}</b>" if bold_label else esc)
    if sub:
        parts.append(
            f'<span style="font-size:11px;">{html.escape(sub, quote=False)}</span>'
        )
    return "<br>".join(parts)


def build(spec: Dict[str, Any]) -> Diagram:
    """Turn a spec into placed geometry. Raises SpecError on bad input."""
    if not isinstance(spec, dict):
        raise SpecError("spec must be an object")

    lay_in = spec.get("layout") or {}
    layout = Layout(
        col_width=int(lay_in.get("col_width", 180)),
        row_height=int(lay_in.get("row_height", 80)),
        col_gap=int(lay_in.get("col_gap", 46)),
        row_gap=int(lay_in.get("row_gap", 34)),
        origin=tuple(lay_in.get("origin", (40, 40))),  # type: ignore[arg-type]
    )

    diagram = Diagram(title=str(spec.get("title", "Page-1")), layout=layout)
    placed: Dict[str, Box] = {}

    # --- nodes ---------------------------------------------------------------
    for node in spec.get("nodes", []) or []:
        nid = str(node.get("id", "")).strip()
        if not nid:
            raise SpecError("every node needs an id")
        if nid in placed:
            raise SpecError(f"duplicate node id: {nid!r}")

        role = str(node.get("role", "default"))
        if role not in NODE_ROLES:
            raise SpecError(
                f"node {nid!r}: unknown role {role!r}. "
                f"Known: {', '.join(sorted(NODE_ROLES))}"
            )

        colspan = max(1, int(node.get("colspan", 1)))
        rowspan = max(1, int(node.get("rowspan", 1)))
        box = Box(
            id=nid,
            x=layout.x(int(node.get("col", 0))) + int(node.get("dx", 0)),
            y=layout.y(int(node.get("row", 0))) + int(node.get("dy", 0)),
            w=int(node.get("width", layout.w(colspan))),
            h=int(node.get("height", layout.h(rowspan))),
            style=NODE_ROLES[role] + str(node.get("extra_style", "")),
            value=_label(
                str(node.get("label", "")),
                str(node.get("sub", "")),
                bool(node.get("bold", True)),
            ),
        )
        placed[nid] = box

    # --- groups --------------------------------------------------------------
    # Drawn before their members so draw.io paints the container underneath;
    # mxGraph z-order follows document order.
    groups: List[Box] = []
    for group in spec.get("groups", []) or []:
        gid = str(group.get("id", "")).strip()
        if not gid:
            raise SpecError("every group needs an id")
        members = [str(m) for m in (group.get("members") or [])]
        missing = [m for m in members if m not in placed]
        if missing:
            raise SpecError(f"group {gid!r} references unknown nodes: {missing}")
        if not members:
            raise SpecError(f"group {gid!r} has no members")

        role = str(group.get("role", "default"))
        if role not in GROUP_ROLES:
            raise SpecError(
                f"group {gid!r}: unknown role {role!r}. "
                f"Known: {', '.join(sorted(GROUP_ROLES))}"
            )

        # pad = [left, top, right, bottom]; top defaults larger to clear the title
        pad = group.get("pad", [14, 30, 14, 14])
        pl, pt, pr, pb = (list(pad) + [14, 30, 14, 14])[:4]

        xs = [placed[m].x for m in members]
        ys = [placed[m].y for m in members]
        xe = [placed[m].x + placed[m].w for m in members]
        ye = [placed[m].y + placed[m].h for m in members]

        groups.append(Box(
            id=gid,
            x=min(xs) - int(pl),
            y=min(ys) - int(pt),
            w=(max(xe) - min(xs)) + int(pl) + int(pr),
            h=(max(ye) - min(ys)) + int(pt) + int(pb),
            style=GROUP_ROLES[role] + str(group.get("extra_style", "")),
            value=html.escape(str(group.get("label", "")), quote=False),
        ))

    # --- free labels ---------------------------------------------------------
    labels: List[Box] = []
    for i, lab in enumerate(spec.get("labels", []) or []):
        role = str(lab.get("role", "note"))
        if role not in LABEL_ROLES:
            raise SpecError(
                f"label #{i}: unknown role {role!r}. "
                f"Known: {', '.join(sorted(LABEL_ROLES))}"
            )
        lid = str(lab.get("id") or f"lbl{i}")
        labels.append(Box(
            id=lid,
            x=layout.x(int(lab.get("col", 0))) + int(lab.get("dx", 0)),
            y=layout.y(int(lab.get("row", 0))) + int(lab.get("dy", 0)),
            w=int(lab.get("width", layout.col_width)),
            h=int(lab.get("height", 20)),
            style=LABEL_ROLES[role] + str(lab.get("extra_style", "")),
            value=html.escape(str(lab.get("text", "")), quote=False),
        ))

    # groups first (painted under), then nodes, then labels on top
    diagram.boxes = groups + list(placed.values()) + labels

    # --- edges ---------------------------------------------------------------
    for i, edge in enumerate(spec.get("edges", []) or []):
        src = str(edge.get("from", ""))
        dst = str(edge.get("to", ""))
        known = set(placed) | {g.id for g in groups}
        for end, name in ((src, "from"), (dst, "to")):
            if end not in known:
                raise SpecError(
                    f"edge #{i}: {name}={end!r} is not a node or group id"
                )
        role = str(edge.get("role", "default"))
        if role not in EDGE_ROLES:
            raise SpecError(
                f"edge #{i}: unknown role {role!r}. "
                f"Known: {', '.join(sorted(EDGE_ROLES))}"
            )

        style = EDGE_ROLES[role]
        exit_a = edge.get("exit")
        entry_a = edge.get("entry")
        if exit_a:
            if exit_a not in ANCHORS:
                raise SpecError(f"edge #{i}: unknown exit {exit_a!r}")
            ex, ey = ANCHORS[exit_a]
            style += f"exitX={ex};exitY={ey};exitDx=0;exitDy=0;"
        if entry_a:
            if entry_a not in ANCHORS:
                raise SpecError(f"edge #{i}: unknown entry {entry_a!r}")
            nx, ny = ANCHORS[entry_a]
            style += f"entryX={nx};entryY={ny};entryDx=0;entryDy=0;"
        style += str(edge.get("extra_style", ""))

        diagram.edges.append({
            "id": str(edge.get("id") or f"e{i}"),
            "source": src,
            "target": dst,
            "style": style,
            "value": html.escape(str(edge.get("label", "")), quote=False),
            "waypoints": edge.get("waypoints") or [],
        })

    return diagram


def to_xml(diagram: Diagram, page_size: Optional[Tuple[int, int]] = None) -> str:
    """Serialise a single diagram to uncompressed mxGraph XML."""
    return to_xml_multi([diagram], [page_size] if page_size else None)


def to_xml_multi(
    diagrams: List[Diagram],
    page_sizes: Optional[List[Optional[Tuple[int, int]]]] = None,
) -> str:
    """Serialise several diagrams as pages of one .drawio file.

    draw.io shows each <diagram> as a tab, which keeps a set of related figures
    in one file instead of scattering them across several.
    """
    if not diagrams:
        raise SpecError("no diagrams to write")

    mxfile = ET.Element("mxfile", {
        "host": "Claude",
        "modified": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.000Z"),
        "agent": "drawio_builder.py",
        "version": "24.7.17",
        "type": "device",
    })
    for i, diagram in enumerate(diagrams):
        size = (page_sizes[i] if page_sizes and i < len(page_sizes) else None)
        _append_page(mxfile, diagram, size)

    ET.indent(mxfile, space="  ")
    return '<?xml version="1.0" encoding="UTF-8"?>\n' + ET.tostring(
        mxfile, encoding="unicode"
    )


def _append_page(
    mxfile: ET.Element,
    diagram: Diagram,
    page_size: Optional[Tuple[int, int]] = None,
) -> None:
    width, height = page_size or _page_size(diagram)

    dia = ET.SubElement(mxfile, "diagram", {
        "id": str(uuid.uuid4()),
        "name": diagram.title,
    })
    model = ET.SubElement(dia, "mxGraphModel", {
        "dx": "1200", "dy": "800", "grid": "1", "gridSize": "10",
        "guides": "1", "tooltips": "1", "connect": "1", "arrows": "1",
        "fold": "1", "page": "1", "pageScale": "1",
        "pageWidth": str(width), "pageHeight": str(height),
        "math": "0", "shadow": "0",
    })
    root = ET.SubElement(model, "root")
    ET.SubElement(root, "mxCell", {"id": "0"})
    ET.SubElement(root, "mxCell", {"id": "1", "parent": "0"})

    for box in diagram.boxes:
        cell = ET.SubElement(root, "mxCell", {
            "id": box.id,
            "value": box.value,
            "style": box.style,
            "vertex": "1",
            "parent": box.parent,
        })
        ET.SubElement(cell, "mxGeometry", {
            "x": str(box.x), "y": str(box.y),
            "width": str(box.w), "height": str(box.h),
            "as": "geometry",
        })

    for edge in diagram.edges:
        cell = ET.SubElement(root, "mxCell", {
            "id": edge["id"],
            "value": edge["value"],
            "style": edge["style"],
            "edge": "1",
            "parent": "1",
            "source": edge["source"],
            "target": edge["target"],
        })
        geom = ET.SubElement(cell, "mxGeometry", {
            "relative": "1", "as": "geometry",
        })
        if edge["waypoints"]:
            array = ET.SubElement(geom, "Array", {"as": "points"})
            for point in edge["waypoints"]:
                ET.SubElement(array, "mxPoint", {
                    "x": str(int(point[0])), "y": str(int(point[1])),
                })


def _page_size(diagram: Diagram) -> Tuple[int, int]:
    """Page big enough for the content, rounded up, with a margin."""
    if not diagram.boxes:
        return (850, 1100)
    right = max(b.x + b.w for b in diagram.boxes) + 40
    bottom = max(b.y + b.h for b in diagram.boxes) + 40
    return (max(850, right), max(500, bottom))


def render(spec: Dict[str, Any]) -> str:
    """Render one spec, or a {"pages": [...]} wrapper, to XML."""
    return to_xml_multi([build(s) for s in _as_pages(spec)])


def _as_pages(spec: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Accept either a single spec or {"pages": [spec, spec, ...]}."""
    if isinstance(spec, dict) and isinstance(spec.get("pages"), list):
        if not spec["pages"]:
            raise SpecError('"pages" is empty')
        return spec["pages"]
    return [spec]


def write(spec: Dict[str, Any], path: str) -> Dict[str, Any]:
    """Build, verify and write. Returns a summary for the caller."""
    pages = _as_pages(spec)
    diagrams = [build(s) for s in pages]
    xml = to_xml_multi(diagrams)
    validate_xml(xml)  # never write a file draw.io would refuse to open
    import os

    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(xml)
    return {
        "path": os.path.abspath(path),
        "bytes": len(xml.encode("utf-8")),
        "pages": [
            {"name": d.title, "shapes": len(d.boxes), "edges": len(d.edges)}
            for d in diagrams
        ],
        "shapes": sum(len(d.boxes) for d in diagrams),
        "edges": sum(len(d.edges) for d in diagrams),
    }


def validate_xml(xml: str) -> Dict[str, Any]:
    """Check the file is well-formed and structurally what draw.io expects."""
    try:
        root = ET.fromstring(xml)
    except ET.ParseError as exc:
        raise SpecError(f"not well-formed XML: {exc}") from exc

    if root.tag != "mxfile":
        raise SpecError(f"root element is <{root.tag}>, expected <mxfile>")
    diagrams = root.findall("diagram")
    if not diagrams:
        raise SpecError("no <diagram> element")

    summary = []
    for dia in diagrams:
        model = dia.find("mxGraphModel")
        if model is None:
            raise SpecError(f"diagram {dia.get('name')!r} has no <mxGraphModel>")
        cells = model.findall("root/mxCell")
        ids = {c.get("id") for c in cells}
        if "0" not in ids or "1" not in ids:
            raise SpecError("missing the mandatory root cells with id 0 and 1")
        # Dangling edge endpoints open as an empty page with no error, which is
        # the single most confusing failure mode, so catch it here.
        for cell in cells:
            if cell.get("edge") == "1":
                for attr in ("source", "target"):
                    ref = cell.get(attr)
                    if ref and ref not in ids:
                        raise SpecError(
                            f"edge {cell.get('id')!r} points at missing "
                            f"{attr}={ref!r}"
                        )
        summary.append({
            "name": dia.get("name"),
            "shapes": sum(1 for c in cells if c.get("vertex") == "1"),
            "edges": sum(1 for c in cells if c.get("edge") == "1"),
        })
    return {"ok": True, "diagrams": summary}


def validate_file(path: str) -> Dict[str, Any]:
    with open(path, encoding="utf-8") as fh:
        return validate_xml(fh.read())


def describe_roles() -> Dict[str, List[str]]:
    """The vocabulary a caller may use in a spec."""
    return {
        "node_roles": sorted(NODE_ROLES),
        "group_roles": sorted(GROUP_ROLES),
        "edge_roles": sorted(EDGE_ROLES),
        "label_roles": sorted(LABEL_ROLES),
        "anchors": sorted(ANCHORS),
    }
