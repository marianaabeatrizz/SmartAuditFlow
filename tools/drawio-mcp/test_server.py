"""End-to-end test: drive the server through a real MCP client over stdio.

Registering a tool is not the same as the server working. This spawns
``server.py`` as a subprocess, completes the protocol handshake, calls each
tool and checks the results -- which is what a client such as Claude Code
actually does.

Run:  .venv/Scripts/python.exe tools/drawio-mcp/test_server.py
"""
from __future__ import annotations

import asyncio
import json
import os
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from mcp import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client

import drawio_builder as db

SPEC = {
    "title": "teste",
    "nodes": [
        {"id": "in", "label": "Contrato", "sub": "+ estático", "col": 0, "row": 1},
        {"id": "plan", "label": "Planejador", "col": 1, "row": 1},
        {"id": "s1", "label": "reentrância", "col": 2, "row": 0, "height": 44},
        {"id": "s2", "label": "aritmética", "col": 2, "row": 1, "height": 44},
        {"id": "judge", "label": "Painel", "col": 3, "row": 1, "role": "accent"},
        {"id": "gap", "label": "não existe", "col": 4, "row": 1, "role": "absent"},
    ],
    "groups": [
        {"id": "swarm", "label": "ESPECIALISTAS", "members": ["s1", "s2"]},
    ],
    "edges": [
        {"from": "in", "to": "plan"},
        {"from": "plan", "to": "s1", "role": "accent"},
        {"from": "plan", "to": "s2", "role": "accent"},
        {"from": "s1", "to": "judge"},
        {"from": "s2", "to": "judge"},
    ],
    "labels": [{"text": "ESTÁGIOS", "col": 0, "row": 0, "role": "caption", "dy": -30}],
}


def payload(result) -> dict:
    """Unwrap the JSON a tool returned as its text content."""
    assert result.content, "tool returned no content"
    return json.loads(result.content[0].text)


async def main() -> int:
    python = sys.executable
    params = StdioServerParameters(
        command=python,
        args=[os.path.join(HERE, "server.py")],
        env={**os.environ, "PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8"},
    )

    failures = 0

    def check(label: str, condition: bool, detail: str = "") -> None:
        nonlocal failures
        if condition:
            print(f"PASS  {label}" + (f" — {detail}" if detail else ""))
        else:
            failures += 1
            print(f"FAIL  {label}" + (f" — {detail}" if detail else ""))

    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            init = await session.initialize()
            # The SDK exposes this snake_cased; the wire field is serverInfo.
            info = getattr(init, "server_info", None) or getattr(init, "serverInfo")
            check("handshake", info.name == "drawio", f"server_info.name={info.name}")

            tools = await session.list_tools()
            names = {t.name for t in tools.tools}
            expected = {
                "drawio_vocabulary", "drawio_write", "drawio_render",
                "drawio_validate", "drawio_read",
            }
            check("tools/list", expected <= names, f"{len(names)} ferramentas")

            vocab = payload(await session.call_tool("drawio_vocabulary", {}))
            check("vocabulary", vocab["ok"] and "accent" in vocab["roles"]["node_roles"])

            with tempfile.TemporaryDirectory() as tmp:
                good = os.path.join(tmp, "diagrama.drawio")
                res = payload(await session.call_tool(
                    "drawio_write", {"path": good, "spec": SPEC}))
                check("write", res["ok"] and os.path.exists(good),
                      f"{res.get('shapes')} formas, {res.get('edges')} arestas")

                val = payload(await session.call_tool("drawio_validate", {"path": good}))
                check("validate", val["ok"])

                red = payload(await session.call_tool(
                    "drawio_read", {"path": good, "include_xml": True}))
                check("read", red["ok"] and red["xml"].startswith("<?xml"))

                # The path that matters: a bad spec must fail loudly and leave
                # no file, because draw.io opens a broken file as a blank page.
                ghost = os.path.join(tmp, "nao-deve-existir.drawio")
                bad = payload(await session.call_tool("drawio_write", {
                    "path": ghost,
                    "spec": {"nodes": [{"id": "a"}],
                             "edges": [{"from": "a", "to": "FANTASMA"}]},
                }))
                check("spec inválida rejeitada", not bad["ok"], bad.get("error", "")[:60])
                check("nenhum arquivo quebrado escrito", not os.path.exists(ghost))

                unknown = payload(await session.call_tool("drawio_write", {
                    "path": os.path.join(tmp, "x.drawio"),
                    "spec": {"nodes": [{"id": "a", "role": "inventada"}]},
                }))
                check("role desconhecida rejeitada", not unknown["ok"])

                missing = payload(await session.call_tool(
                    "drawio_validate", {"path": os.path.join(tmp, "ausente.drawio")}))
                check("arquivo ausente tratado", not missing["ok"])

            render = payload(await session.call_tool("drawio_render", {"spec": SPEC}))
            check("render sem escrever", render["ok"] and "<mxfile" in render["xml"])

    print(f"\n{'' if failures else 'todos os testes passaram'}"
          f"{f'{failures} falha(s)' if failures else ''}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
