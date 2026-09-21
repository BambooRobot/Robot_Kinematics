"""@file test_render_text.py

@brief 终端渲染：中文宽度对齐、表格、块渲染、落盘。
"""

from __future__ import annotations

import numpy as np
import pytest

from robotkinematics.adapters.render_text import (
    TextRenderer,
    display_width,
    format_matrix,
    pad,
    table,
    write_output,
)
from robotkinematics.contracts import (
    HeadingBlock,
    MatrixBlock,
    Report,
    TableBlock,
    TextBlock,
    VectorBlock,
)
from robotkinematics.core.exceptions import KinematicsError


def test_display_width_counts_cjk_as_two_columns():
    """中文字符在等宽终端里按 2 列算。"""
    assert display_width("abc") == 3
    assert display_width("普通姿态") == 8
    assert display_width("q1=90°") == 6


def test_pad_aligns_by_display_width_not_character_count():
    """补齐按显示宽度算。"""
    assert display_width(pad("普通", 8)) == 8
    assert display_width(pad("ab", 8)) == 8
    assert pad("ab", 5, ">") == "   ab"


def test_table_columns_line_up():
    """中文与角度符号混排时每行显示宽度仍相等。"""
    text = table(["姿态", "q2"], [["普通姿态", "90°"], ["接近奇异", "10°"]])
    widths = {display_width(line) for line in text.splitlines()}
    assert len(widths) == 1


def test_table_handles_empty_rows():
    """没有数据行时返回空串。"""
    assert table(["a", "b"], []) == ""


def test_table_rejects_mismatched_alignments():
    """对齐符号个数与列数不符要报错。"""
    with pytest.raises(KinematicsError):
        table(["a", "b"], [["1", "2"]], aligns=["<"])


def test_format_matrix_aligns_columns():
    """矩阵各行的显示宽度要一致。"""
    text = format_matrix(np.array([[1.0, -20.0], [333.0, 4.0]]), digits=2)
    lines = text.splitlines()
    assert len(lines) == 2
    assert display_width(lines[0]) == display_width(lines[1])


def test_format_matrix_rejects_non_matrix():
    """传一维数组要报错。"""
    with pytest.raises(KinematicsError):
        format_matrix(np.zeros(3))


def test_renderer_renders_every_block_kind():
    """五种块类型都要渲染出内容。"""
    report = Report(
        title="标题",
        blocks=(
            HeadingBlock("小节"),
            TextBlock.of("一段文字"),
            VectorBlock("向量", np.array([1.0, 2.0])),
            MatrixBlock("矩阵", np.eye(2)),
            TableBlock(("列",), (("值",),)),
        ),
    )
    text = TextRenderer().render(report)
    for expected in ("标题", "小节", "一段文字", "向量 = [ 1.0000,  2.0000]", "矩阵", "列"):
        assert expected in text


def test_renderer_writes_only_the_body_to_file(tmp_path):
    """落盘的是正文（不带分节线、不带“已保存”那行）。"""
    report = Report(
        title="标题", blocks=(TextBlock.of("正文第一行", "正文第二行"),), output_name="out.txt"
    )
    text = TextRenderer(outputs_dir=tmp_path).render(report)
    written = (tmp_path / "out.txt").read_text(encoding="utf-8")
    assert written == "正文第一行\n正文第二行\n"
    assert "标题" not in written
    assert "已保存" in text


def test_renderer_without_outputs_dir_does_not_write(tmp_path):
    """没给输出目录时只渲染不落盘。"""
    report = Report(title="标题", blocks=(TextBlock.of("x"),), output_name="out.txt")
    TextRenderer(outputs_dir=None).render(report)
    assert not (tmp_path / "out.txt").exists()


def test_renderer_rejects_unknown_block():
    """未知块类型要显式报错。"""

    class Weird:
        pass

    with pytest.raises(KinematicsError):
        TextRenderer().render(Report(title="t", blocks=(Weird(),)))  # type: ignore[arg-type]


def test_write_output_creates_directory(tmp_path):
    """输出目录不存在时自动创建。"""
    path = write_output("report.txt", "内容", tmp_path / "nested" / "outputs")
    assert path.exists()
    assert path.read_text(encoding="utf-8") == "内容\n"


def test_renderer_reports_saved_path(tmp_path):
    """渲染结果里要回显落盘路径。"""
    report = Report(title="t", blocks=(TextBlock.of("x"),), output_name="r.txt")
    assert "r.txt" in TextRenderer(outputs_dir=tmp_path).render(report)
