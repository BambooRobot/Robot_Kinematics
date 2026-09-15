"""@file render_json.py
@brief 把同一份报告渲染成 JSON —— 用来证明"用例不认识终端"这件事是真的。

重构前，用例直接拼中文字符串，想给别的程序调用就得把逻辑再写一遍；
现在用例产出的是 Report（结构化），换渲染器即可换格式。
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from ..contracts import HeadingBlock, MatrixBlock, Report, TableBlock, TextBlock, VectorBlock
from .render_text import write_output


def _jsonable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, dict):
        return {k: _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


class JsonRenderer:
    """JSON 渲染器。结构：{title, blocks:[...], fields:{...}}。"""

    def __init__(self, outputs_dir: str | Path | None = None, indent: int = 2) -> None:
        self.outputs_dir = Path(outputs_dir) if outputs_dir is not None else None
        self.indent = indent

    def render(self, report: Report) -> str:
        payload = {
            "title": report.title,
            "blocks": [self._render_block(b) for b in report.blocks],
            "fields": _jsonable(report.fields),
        }
        text = json.dumps(payload, ensure_ascii=False, indent=self.indent)
        if report.output_name and self.outputs_dir is not None:
            path = write_output(report.output_name, text, self.outputs_dir)
            return text + f"\n（已保存：{path}）"
        return text

    def _render_block(self, block) -> dict:
        if isinstance(block, HeadingBlock):
            return {"type": "heading", "title": block.title}
        if isinstance(block, TextBlock):
            return {"type": "text", "lines": list(block.lines)}
        if isinstance(block, VectorBlock):
            return {
                "type": "vector",
                "label": block.label,
                "values": np.asarray(block.values, dtype=float).ravel().tolist(),
                "suffix": block.suffix,
            }
        if isinstance(block, MatrixBlock):
            return {
                "type": "matrix",
                "label": block.label,
                "matrix": np.asarray(block.matrix, dtype=float).tolist(),
            }
        if isinstance(block, TableBlock):
            return {
                "type": "table",
                "headers": list(block.headers),
                "rows": [list(row) for row in block.rows],
            }
        return {"type": type(block).__name__, "unsupported": True}
