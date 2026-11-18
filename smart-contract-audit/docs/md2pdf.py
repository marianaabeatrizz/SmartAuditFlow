"""Render the report Markdown to PDF with ReportLab.

Deliberately handles only the subset of Markdown the report uses: headings,
paragraphs, bullets, pipe tables, fenced code, **bold**, *italic*, `code`,
and --- as a section rule. Anything fancier would be a second project.
"""
from __future__ import annotations

import html
import os
import re
import sys

from reportlab.lib import colors
from reportlab.lib.enums import TA_JUSTIFY, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import cm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    BaseDocTemplate, Frame, HRFlowable, KeepTogether, PageBreak, Paragraph,
    PageTemplate, Preformatted, Spacer, Table, TableStyle,
)

FONTS = r"C:\Windows\Fonts"
for name, file in (("Body", "calibri.ttf"), ("Body-Bold", "calibrib.ttf"),
                   ("Body-Italic", "calibrii.ttf"), ("Body-BoldItalic", "calibriz.ttf"),
                   ("Mono", "consola.ttf"), ("Mono-Bold", "consolab.ttf")):
    pdfmetrics.registerFont(TTFont(name, os.path.join(FONTS, file)))
pdfmetrics.registerFontFamily("Body", normal="Body", bold="Body-Bold",
                              italic="Body-Italic", boldItalic="Body-BoldItalic")

INK = colors.HexColor("#1a1a1a")
MUTED = colors.HexColor("#5b6670")
RULE = colors.HexColor("#d4d9dd")
BAND = colors.HexColor("#eef1f3")
ACCENT = colors.HexColor("#0b5d7a")

S = {
    "title": ParagraphStyle("title", fontName="Body-Bold", fontSize=26, leading=30,
                            textColor=INK, spaceAfter=4),
    "subtitle": ParagraphStyle("subtitle", fontName="Body", fontSize=14, leading=19,
                               textColor=MUTED, spaceAfter=18),
    "h1": ParagraphStyle("h1", fontName="Body-Bold", fontSize=17, leading=21,
                         textColor=ACCENT, spaceBefore=20, spaceAfter=7),
    "h2": ParagraphStyle("h2", fontName="Body-Bold", fontSize=13, leading=17,
                         textColor=INK, spaceBefore=14, spaceAfter=5),
    "body": ParagraphStyle("body", fontName="Body", fontSize=10.3, leading=15.2,
                           textColor=INK, alignment=TA_JUSTIFY, spaceAfter=7),
    "bullet": ParagraphStyle("bullet", fontName="Body", fontSize=10.3, leading=15.2,
                             textColor=INK, alignment=TA_LEFT, leftIndent=14,
                             bulletIndent=3, spaceAfter=3),
    "cell": ParagraphStyle("cell", fontName="Body", fontSize=9.2, leading=12.4,
                           textColor=INK),
    "cellh": ParagraphStyle("cellh", fontName="Body-Bold", fontSize=9.2, leading=12.4,
                            textColor=INK),
    "code": ParagraphStyle("code", fontName="Mono", fontSize=8.0, leading=10.6,
                           textColor=INK),
}

INLINE = (
    (re.compile(r"\*\*(.+?)\*\*"), r"<b>\1</b>"),
    (re.compile(r"(?<![\w*])\*([^*\n]+?)\*(?![\w*])"), r"<i>\1</i>"),
    (re.compile(r"`([^`]+?)`"), r'<font face="Mono" size="9">\1</font>'),
)


def inline(text: str) -> str:
    out = html.escape(text, quote=False)
    for pattern, repl in INLINE:
        out = pattern.sub(repl, out)
    return out


def split_row(line: str):
    return [c.strip() for c in line.strip().strip("|").split("|")]


def build_table(rows, width):
    head, body = rows[0], rows[1:]
    ncols = max(len(r) for r in rows)
    # Numeric columns are narrow; the first text column takes what is left.
    def numericish(idx):
        vals = [r[idx] for r in body if idx < len(r)]
        if not vals:
            return False
        return all(re.fullmatch(r"[\s\d.,%+\-*]*", v or "") for v in vals)

    weights = []
    for i in range(ncols):
        if i == 0:
            weights.append(3.2)
        elif numericish(i):
            weights.append(1.0)
        else:
            weights.append(2.4)
    total = sum(weights)
    widths = [width * w / total for w in weights]

    data = []
    for ri, row in enumerate(rows):
        cells = list(row) + [""] * (ncols - len(row))
        style = S["cellh"] if ri == 0 else S["cell"]
        data.append([Paragraph(inline(c), style) for c in cells])

    t = Table(data, colWidths=widths, repeatRows=1, hAlign="LEFT")
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), BAND),
        ("LINEBELOW", (0, 0), (-1, 0), 0.9, ACCENT),
        ("LINEBELOW", (0, 1), (-1, -2), 0.35, RULE),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("LEFTPADDING", (0, 0), (-1, -1), 7),
        ("RIGHTPADDING", (0, 0), (-1, -1), 7),
    ]))
    return t


def code_block(lines, width):
    body = Preformatted("\n".join(lines), S["code"])
    t = Table([[body]], colWidths=[width], hAlign="LEFT")
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#f6f8f9")),
        ("BOX", (0, 0), (-1, -1), 0.5, RULE),
        ("LEFTPADDING", (0, 0), (-1, -1), 9),
        ("RIGHTPADDING", (0, 0), (-1, -1), 9),
        ("TOPPADDING", (0, 0), (-1, -1), 8),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
    ]))
    return t


def parse(md: str, width: float):
    flow = []
    lines = md.split("\n")
    i = 0
    para: list[str] = []
    seen_title = False

    def flush():
        nonlocal para
        if para:
            flow.append(Paragraph(inline(" ".join(para)), S["body"]))
            para = []

    while i < len(lines):
        raw = lines[i]
        line = raw.rstrip()

        if line.startswith("```"):
            flush()
            i += 1
            buf = []
            while i < len(lines) and not lines[i].startswith("```"):
                buf.append(lines[i].rstrip())
                i += 1
            i += 1
            flow.append(Spacer(1, 4))
            flow.append(code_block(buf, width))
            flow.append(Spacer(1, 8))
            continue

        if line.startswith("|") and i + 1 < len(lines) and re.fullmatch(
                r"\|[\s:|-]+\|", lines[i + 1].strip()):
            flush()
            rows = [split_row(line)]
            i += 2
            while i < len(lines) and lines[i].strip().startswith("|"):
                rows.append(split_row(lines[i]))
                i += 1
            flow.append(Spacer(1, 3))
            flow.append(build_table(rows, width))
            flow.append(Spacer(1, 10))
            continue

        if line.strip() == "---":
            flush()
            flow.append(Spacer(1, 7))
            flow.append(HRFlowable(width="100%", thickness=0.6, color=RULE,
                                   spaceBefore=0, spaceAfter=0))
            flow.append(Spacer(1, 3))
            i += 1
            continue

        m = re.match(r"^(#{1,4})\s+(.*)$", line)
        if m:
            flush()
            level, text = len(m.group(1)), m.group(2)
            if level == 1 and not seen_title:
                seen_title = True
                flow.append(Spacer(1, 2.2 * cm))
                flow.append(Paragraph(inline(text), S["title"]))
            elif level == 2 and not flow_has_h1(flow):
                flow.append(Paragraph(inline(text), S["subtitle"]))
            else:
                style = S["h1"] if level == 2 else S["h2"]
                flow.append(Paragraph(inline(text), style))
            i += 1
            continue

        m = re.match(r"^\s*[-*]\s+(.*)$", line)
        if m:
            flush()
            text = m.group(1)
            j = i + 1
            while j < len(lines) and re.match(r"^\s{2,}\S", lines[j]) and not re.match(
                    r"^\s*[-*]\s", lines[j]):
                text += " " + lines[j].strip()
                j += 1
            flow.append(Paragraph(inline(text), S["bullet"], bulletText="\u2022"))
            i = j
            continue

        if not line.strip():
            flush()
            i += 1
            continue

        para.append(line.strip())
        i += 1

    flush()
    return flow


_h1_emitted = {"done": False}


def flow_has_h1(flow):
    """The first '##' is the document subtitle; every later one is a section."""
    if _h1_emitted["done"]:
        return True
    _h1_emitted["done"] = True
    return False


def footer(canvas, doc):
    canvas.saveState()
    canvas.setFont("Body", 8.2)
    canvas.setFillColor(MUTED)
    if doc.page > 1:
        canvas.drawString(2.2 * cm, 1.35 * cm,
                          "SmartAuditFlow Multiagente \u2014 relat\u00f3rio t\u00e9cnico")
        canvas.drawRightString(A4[0] - 2.2 * cm, 1.35 * cm, str(doc.page))
    canvas.restoreState()


def main(src, dst):
    md = open(src, encoding="utf-8").read()
    doc = BaseDocTemplate(dst, pagesize=A4,
                          leftMargin=2.2 * cm, rightMargin=2.2 * cm,
                          topMargin=1.9 * cm, bottomMargin=2.1 * cm,
                          title="SmartAuditFlow Multiagente - relatorio tecnico",
                          author="Equipe SmartAuditFlow")
    frame = Frame(doc.leftMargin, doc.bottomMargin, doc.width, doc.height, id="main")
    doc.addPageTemplates([PageTemplate(id="p", frames=[frame], onPage=footer)])
    doc.build(parse(md, doc.width))
    print(f"ok: {dst} ({os.path.getsize(dst):,} bytes)")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
