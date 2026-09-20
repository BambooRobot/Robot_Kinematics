"""@file test_contracts.py

@brief 契约层：报告块、值格式化、以及"用例依赖抽象"这件事真的成立了。
"""

from __future__ import annotations

from dataclasses import FrozenInstanceError

import numpy as np
import pytest

from robotkinematics.contracts import (
    CaseSource,
    HeadingBlock,
    MatrixBlock,
    Renderer,
    Report,
    TableBlock,
    TextBlock,
    TwoLinkChartBlock,
    VectorBlock,
    fmt_condition,
    fmt_vector,
)


def test_fmt_vector_is_fixed_width_and_signed():
    """向量格式化定宽带符号 —— 报告里数字列必须对齐，负号不能把位宽挤歪。"""
    assert fmt_vector([1.0, -0.5]) == "[ 1.0000, -0.5000]"
    assert fmt_vector(np.array([0.0, 0.0, 0.0]), digits=2) == "[ 0.00,  0.00,  0.00]"


def test_fmt_condition_hides_meaningless_huge_numbers():
    """奇异时条件数会溢出到 inf / 1e18 这种天文数字，报告里统一显示为「inf/极大」。"""
    assert fmt_condition(2.618) == "2.62"
    assert fmt_condition(float("inf")) == "inf/极大"
    assert fmt_condition(1e18) == "inf/极大"


def test_text_block_of_helper():
    """TextBlock.of 是拼报告的常用入口：多行文字一次收进块，顺序不许被重排。"""
    block = TextBlock.of("第一行", "第二行")
    assert block.lines == ("第一行", "第二行")


def test_report_defaults():
    """报告默认是「只有标题的空壳」：无块、无输出名、无字段 —— 用例按需再填。"""
    report = Report(title="标题")
    assert report.blocks == ()
    assert report.output_name is None
    assert report.fields == {}


def test_report_blocks_are_immutable_and_typed():
    """报告块是 frozen dataclass：用例产出之后谁也不该改它。"""
    block = VectorBlock("标签", np.array([1.0, 2.0]))
    with pytest.raises(FrozenInstanceError):
        block.label = "改掉"  # type: ignore[misc]


def test_blocks_cover_what_the_reports_need():
    """这几种块足够表达现有全部报告 —— 加新块类型应该是罕见事件。"""
    block_types = (HeadingBlock, TextBlock, VectorBlock, MatrixBlock, TableBlock, TwoLinkChartBlock)
    assert len({t.__name__ for t in block_types}) == len(block_types)


# ── 协议：用例只依赖抽象 ────────────────────────────────────────────────────


def test_csv_case_source_satisfies_the_protocol():
    """用例层只依赖 CaseSource 抽象 —— CSV 数据源换掉不影响用例代码。"""
    from robotkinematics.adapters.cases_csv import CsvCaseSource

    assert isinstance(CsvCaseSource("data/cases"), CaseSource)


def test_renderers_satisfy_the_protocol():
    """文本与 JSON 两种渲染器都满足 Renderer 协议，输出格式可替换而报告不变。"""
    from robotkinematics.adapters.render_json import JsonRenderer
    from robotkinematics.adapters.render_text import TextRenderer

    assert isinstance(TextRenderer(), Renderer)
    assert isinstance(JsonRenderer(), Renderer)


def test_plotter_satisfies_the_protocol():
    """Matplotlib 绘图器满足 Plotter 协议 —— 本机 mplot3d 坏掉也不该拖垮契约层的类型检查。"""
    from robotkinematics.adapters.plot_mpl import MatplotlibPlotter

    assert isinstance(
        MatplotlibPlotter(), __import__("robotkinematics.contracts", fromlist=["Plotter"]).Plotter
    )
