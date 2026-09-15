"""@file test_render_text.py
@brief 终端渲染：中文宽度对齐、表格、块渲染、落盘 —— 布局的测试都集中在这里。
"""

from __future__ import annotations

import numpy as np
import pytest

from robotkinematics.adapters.render_text import (
    TextRenderer,
    ascii_two_link_chart,
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
    TwoLinkChartBlock,
    VectorBlock,
)
from robotkinematics.core.exceptions import KinematicsError


def test_display_width_counts_cjk_as_two_columns():
    assert display_width("abc") == 3
    assert display_width("普通姿态") == 8
    assert display_width("q1=90°") == 6  # ° 是全角


def test_pad_aligns_by_display_width_not_character_count():
    assert display_width(pad("普通", 8)) == 8
    assert display_width(pad("ab", 8)) == 8
    assert pad("ab", 5, ">") == "   ab"


def test_table_columns_line_up():
    text = table(["姿态", "q2"], [["普通姿态", "90°"], ["接近奇异", "10°"]])
    widths = {display_width(line) for line in text.splitlines()}
    assert len(widths) == 1, "每一行的显示宽度必须一致，否则终端里竖不齐"


def test_table_handles_empty_rows():
    assert table(["a", "b"], []) == ""


def test_table_rejects_mismatched_alignments():
    with pytest.raises(KinematicsError):
        table(["a", "b"], [["1", "2"]], aligns=["<"])


def test_format_matrix_aligns_columns():
    text = format_matrix(np.array([[1.0, -20.0], [333.0, 4.0]]), digits=2)
    lines = text.splitlines()
    assert len(lines) == 2
    assert display_width(lines[0]) == display_width(lines[1])


def test_format_matrix_rejects_non_matrix():
    with pytest.raises(KinematicsError):
        format_matrix(np.zeros(3))


def test_ascii_chart_marks_base_elbow_tool():
    from robotkinematics.core.planar2r import Planar2R

    drawing = ascii_two_link_chart(Planar2R().joint_points(0.0, np.pi / 2))
    for marker in ("B", "E", "T"):
        assert marker in drawing
    assert drawing.count("*") >= 8


def test_renderer_renders_every_block_kind():
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


def test_renderer_draws_the_chart_block():
    from robotkinematics.core.planar2r import Planar2R

    report = Report(
        title="带图", blocks=(TwoLinkChartBlock(points=Planar2R().joint_points(0.0, 1.0)),)
    )
    assert "T" in TextRenderer().render(report)


def test_renderer_writes_only_the_body_to_file(tmp_path):
    """落盘的是正文（不带分节线、不带“已保存”那行），这样才能与课件交付物对照。"""
    report = Report(
        title="标题", blocks=(TextBlock.of("正文第一行", "正文第二行"),), output_name="out.txt"
    )
    text = TextRenderer(outputs_dir=tmp_path).render(report)
    written = (tmp_path / "out.txt").read_text(encoding="utf-8")
    assert written == "正文第一行\n正文第二行\n"
    assert "标题" not in written
    assert "已保存" in text


def test_renderer_without_outputs_dir_does_not_write(tmp_path):
    report = Report(title="标题", blocks=(TextBlock.of("x"),), output_name="out.txt")
    TextRenderer(outputs_dir=None).render(report)
    assert not (tmp_path / "out.txt").exists()


def test_renderer_rejects_unknown_block():
    class Weird:
        pass

    with pytest.raises(KinematicsError):
        TextRenderer().render(Report(title="t", blocks=(Weird(),)))  # type: ignore[arg-type]


def test_write_output_creates_directory(tmp_path):
    path = write_output("report.txt", "内容", tmp_path / "nested" / "outputs")
    assert path.exists()
    assert path.read_text(encoding="utf-8") == "内容\n"


def test_renderer_reports_saved_path(tmp_path):
    report = Report(title="t", blocks=(TextBlock.of("x"),), output_name="r.txt")
    assert "r.txt" in TextRenderer(outputs_dir=tmp_path).render(report)
