"""@file test_singularity.py

@brief 奇异点体检：numpy SVD + 库的可操作度，两边必须对得上（Panda）。
"""

from __future__ import annotations

import numpy as np
import pytest

from robotkinematics.core import robots
from robotkinematics.core.exceptions import KinematicsError
from robotkinematics.core.singularity import (
    amplification,
    analysis_of,
    analyze,
    describe_worst_direction,
    manipulability,
)


def test_identity_jacobian_is_perfectly_conditioned():
    """单位阵是条件数的下界 1。"""
    report = analyze(np.eye(3))
    assert np.isclose(report.condition, 1.0)
    assert np.isclose(report.manipulability, 1.0)
    assert not report.is_singular and not report.is_near_singular


def test_rank_deficient_jacobian_is_singular():
    """奇异判据用 σ_min 而不是 det。"""
    report = analyze(np.array([[1.0, 1.0], [0.0, 0.0]]))
    assert report.is_singular
    assert np.isinf(report.condition)


def test_worst_direction_is_the_smallest_singular_direction():
    """最坏方向 = 最小奇异值对应的右奇异向量。"""
    report = analyze(np.diag([5.0, 0.2, 1.0]))
    assert np.isclose(report.sigma_min, 0.2)
    assert np.allclose(np.abs(report.worst_direction), [0.0, 1.0, 0.0])


def test_describe_worst_direction_translates_to_plain_words():
    """最坏方向要翻译成人话。"""
    assert describe_worst_direction(np.array([0.0, 1.0])) == "平面方向 90.0°"
    assert "z 轴" in describe_worst_direction(np.array([0.0, 0.0, 1.0, 0.0, 0.0, 0.0]))
    assert "y 轴" in describe_worst_direction(np.array([0.0, 0.0, 0.0, 0.0, 1.0, 0.0]))


def test_manipulability_matches_the_library():
    """我们算的（numpy SVD）与库的 `robot.manipulability` 是同一个量。"""
    robot = robots.panda(robots.TCP)
    q = np.array([0.0, -0.3, 0.0, -2.2, 0.0, 2.0, 0.8])
    assert np.isclose(analysis_of(robot, q).manipulability, manipulability(robot, q), rtol=1e-9)
    for method in ("minsingular", "invcondition"):
        assert np.isfinite(manipulability(robot, q, method=method))


def test_panda_near_singular_amplification_is_large():
    """接近奇异时放大倍数应明显大于良好位形。"""
    robot = robots.panda(robots.TCP)
    q_ok = np.array([0.0, -0.4, 0.0, -2.2, 0.0, 2.0, 0.785])
    q_stretched = np.zeros(7)
    gain_ok = amplification(analysis_of(robot, q_ok))
    gain_stretched = amplification(analysis_of(robot, q_stretched))
    assert gain_stretched > gain_ok


def test_analyze_rejects_non_matrix_input():
    """一维数组当雅可比传进来要报 KinematicsError。"""
    with pytest.raises(KinematicsError):
        analyze(np.zeros(6))
