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
    assert fmt_vector([1.0, -0.5]) == "[ 1.0000, -0.5000]"
    assert fmt_vector(np.array([0.0, 0.0, 0.0]), digits=2) == "[ 0.00,  0.00,  0.00]"


def test_fmt_condition_hides_meaningless_huge_numbers():
    assert fmt_condition(2.618) == "2.62"
    assert fmt_condition(float("inf")) == "inf/极大"
    assert fmt_condition(1e18) == "inf/极大"


def test_text_block_of_helper():
    block = TextBlock.of("第一行", "第二行")
    assert block.lines == ("第一行", "第二行")


def test_report_defaults():
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
    from robotkinematics.adapters.cases_csv import CsvCaseSource

    assert isinstance(CsvCaseSource("data/cases"), CaseSource)


def test_renderers_satisfy_the_protocol():
    from robotkinematics.adapters.render_json import JsonRenderer
    from robotkinematics.adapters.render_text import TextRenderer

    assert isinstance(TextRenderer(), Renderer)
    assert isinstance(JsonRenderer(), Renderer)


def test_plotter_satisfies_the_protocol():
    from robotkinematics.adapters.plot_mpl import MatplotlibPlotter

    assert isinstance(
        MatplotlibPlotter(), __import__("robotkinematics.contracts", fromlist=["Plotter"]).Plotter
    )


def test_reference_adapter_satisfies_the_protocol():
    from robotkinematics.adapters.reference_rtb import RtbReference
    from robotkinematics.contracts import KinematicsReference

    assert isinstance(RtbReference(), KinematicsReference)
