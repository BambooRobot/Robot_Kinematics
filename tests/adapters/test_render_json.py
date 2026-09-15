"""@file test_render_json.py
@brief JSON 渲染：证明"用例不认识终端"——同一份 Report 换个渲染器就换了格式。
"""

from __future__ import annotations

import json

import numpy as np

from robotkinematics.adapters.render_json import JsonRenderer
from robotkinematics.contracts import (
    HeadingBlock,
    MatrixBlock,
    Report,
    TextBlock,
    VectorBlock,
)
from robotkinematics.core.planar2r import Planar2R
from robotkinematics.usecases import planar


def test_json_render_has_title_blocks_and_fields():
    report = Report(
        title="标题",
        blocks=(HeadingBlock("小节"), TextBlock.of("文字"), VectorBlock("v", np.array([1.0, 2.0]))),
        fields={"answer": np.array([1, 2, 3]), "ok": True},
    )
    payload = json.loads(JsonRenderer().render(report))
    assert payload["title"] == "标题"
    assert [b["type"] for b in payload["blocks"]] == ["heading", "text", "vector"]
    assert payload["blocks"][2]["values"] == [1.0, 2.0]
    # numpy 必须被转成原生类型，否则 json.dumps 会直接报错
    assert payload["fields"]["answer"] == [1, 2, 3]
    assert payload["fields"]["ok"] is True


def test_json_render_of_a_real_report_is_parseable():
    """真实用例的产出必须能直接喂给别的程序 —— 这是重构的收益之一。"""
    report = planar.ik_report(Planar2R(), 1.0, 1.0)
    payload = json.loads(JsonRenderer().render(report))
    assert payload["fields"]["solutions_deg"][0] == [0.0, 90.0]
    assert payload["blocks"][1]["type"] == "table"


def test_json_render_of_matrix_block_keeps_numbers():
    report = Report(title="t", blocks=(MatrixBlock("J", np.eye(2)),))
    payload = json.loads(JsonRenderer().render(report))
    assert payload["blocks"][0]["matrix"] == [[1.0, 0.0], [0.0, 1.0]]


def test_json_render_writes_file_when_asked(tmp_path):
    report = Report(title="t", blocks=(TextBlock.of("x"),), output_name="r.json", fields={"a": 1})
    text = JsonRenderer(outputs_dir=tmp_path).render(report)
    assert "r.json" in text
    written = json.loads((tmp_path / "r.json").read_text(encoding="utf-8"))
    assert written["fields"] == {"a": 1}
