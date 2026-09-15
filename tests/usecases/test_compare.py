"""@file test_compare.py
@brief 交叉验证：判定逻辑（usecases/compare.py）与取数适配器（adapters/reference_rtb.py）。
"""

from __future__ import annotations

import numpy as np
import pytest

from robotkinematics.adapters.reference_rtb import RtbReference, _import_optional
from robotkinematics.contracts import ReferenceNumbers
from robotkinematics.usecases import compare


def _has_optional_deps() -> bool:
    spatialmath, rtb, _ = _import_optional()
    return spatialmath is not None and rtb is not None


class _FakeReference:
    """假的参照库：不装真库也能测判定逻辑。"""

    def __init__(self, numbers: ReferenceNumbers) -> None:
        self._numbers = numbers

    def probe(self) -> ReferenceNumbers:
        return self._numbers


def test_skip_report_is_explicit():
    """没装库时必须写清楚"跳过了"和怎么装，绝不假装验证过。"""
    report = compare.compare(_FakeReference(ReferenceNumbers(available=False, reason="未安装 X")))
    text = " ".join(line for block in report.blocks for line in getattr(block, "lines", ()))
    assert "跳过" in text
    assert "crosscheck" in text  # 提示里要写清楚怎么装可选依赖
    assert report.fields["available"] is False


def test_agreement_and_disagreement_are_judged_here_not_in_the_adapter():
    """判定逻辑属于用例：同样的数，符号一致就是"一致"，差一点就是"不一致"。"""
    from robotkinematics.contracts import fmt_vector

    mine = np.array([1.0, 2.0, 3.0])
    row_ok = compare._compare("项", mine, mine.copy())
    row_bad = compare._compare("项", mine, mine + 1e-3)
    assert row_ok[-1] == "一致"
    assert row_bad[-1] == "不一致"
    assert fmt_vector(mine, 4) in row_ok[1]


def test_missing_reference_numbers_are_reported_not_ignored():
    row = compare._compare("项", np.zeros(3), None)
    assert row[-1] == "无法对照"
    assert row[2] == "未取到"


def test_dimension_mismatch_is_reported():
    assert compare._compare("项", np.zeros(3), np.zeros(6))[-1] == "维度不同"


def test_real_reference_runs_without_crashing():
    """装了真库就走对照分支，没装就走跳过分支 —— 两条路都不能崩。"""
    report = compare.compare(RtbReference())
    assert isinstance(report.fields, dict)
    if _has_optional_deps():
        assert report.fields["available"] is True
        verdicts = {row["verdict"] for row in report.fields["rows"]}
        assert "一致" in verdicts
        # 逐项都应该对上（这正是重构前那次交叉验证的结论）
        assert verdicts == {"一致"}
    else:
        pytest.skip("本机没装可选依赖，只验了跳过分支")


def test_verdict_explains_the_hand_tcp_frame_when_available():
    if not _has_optional_deps():
        pytest.skip("本机没装可选依赖")
    report = compare.compare(RtbReference())
    text = " ".join(line for block in report.blocks for line in getattr(block, "lines", ()))
    assert "夹爪 TCP" in text
    assert "逐位一致" in text
