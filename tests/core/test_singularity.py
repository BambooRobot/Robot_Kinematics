"""@file test_singularity.py
@brief 奇异点体检：numpy SVD + 库的可操作度，两边必须对得上。
"""

from __future__ import annotations

import itertools

import numpy as np
import pytest

from robotkinematics.core import robots
from robotkinematics.core.exceptions import KinematicsError
from robotkinematics.core.planar2r import Planar2R
from robotkinematics.core.singularity import (
    amplification,
    analysis_of,
    analyze,
    describe_worst_direction,
    manipulability,
)

DEG = np.deg2rad


def test_identity_jacobian_is_perfectly_conditioned():
    report = analyze(np.eye(3))
    assert np.isclose(report.condition, 1.0)
    assert np.isclose(report.manipulability, 1.0)
    assert not report.is_singular and not report.is_near_singular


def test_rank_deficient_jacobian_is_singular():
    report = analyze(np.array([[1.0, 1.0], [0.0, 0.0]]))
    assert report.is_singular
    assert np.isinf(report.condition)


def test_worst_direction_is_the_smallest_singular_direction():
    report = analyze(np.diag([5.0, 0.2, 1.0]))
    assert np.isclose(report.sigma_min, 0.2)
    assert np.allclose(np.abs(report.worst_direction), [0.0, 1.0, 0.0])


def test_describe_worst_direction_translates_to_plain_words():
    assert describe_worst_direction(np.array([0.0, 1.0])) == "平面方向 90.0°"
    assert "z 轴" in describe_worst_direction(np.array([0.0, 0.0, 1.0, 0.0, 0.0, 0.0]))
    assert "y 轴" in describe_worst_direction(np.array([0.0, 0.0, 0.0, 0.0, 1.0, 0.0]))


def test_planar_condition_numbers_match_the_course_table():
    """课件 07 的表格：q2 = 90/30/10° 时 cond(J) = 2.62 / 9.36 / 28.58。"""
    robot = robots.planar(1.0, 1.0)
    for q2_deg, expected in [(90.0, 2.62), (30.0, 9.36), (10.0, 28.58)]:
        J = np.asarray(robot.jacob0([0.0, DEG(q2_deg)]))[:2, :]
        assert np.isclose(analyze(J).condition, expected, atol=5e-3)


def test_amplification_grows_as_the_planar_arm_straightens():
    robot = robots.planar(1.0, 1.0)
    gains = [
        amplification(analyze(np.asarray(robot.jacob0([0.0, DEG(q)]))[:2, :]))
        for q in (90.0, 30.0, 10.0, 1.0)
    ]
    assert all(a < b for a, b in itertools.pairwise(gains))


def test_manipulability_matches_the_library():
    """我们算的（numpy SVD）与库的 `robot.manipulability` 是同一个量。"""
    robot = robots.panda(robots.TCP)
    q = np.array([0.0, -0.3, 0.0, -2.2, 0.0, 2.0, 0.8])
    assert np.isclose(analysis_of(robot, q).manipulability, manipulability(robot, q), rtol=1e-9)
    # 库还提供另外三种定义；asada 对这一位形给出 nan（库的实现如此），所以只查"能调用"
    for method in ("minsingular", "invcondition"):
        assert np.isfinite(manipulability(robot, q, method=method))


def test_analyze_rejects_non_matrix_input():
    with pytest.raises(KinematicsError):
        analyze(np.zeros(6))


def test_planar_det_matches_amplification_at_singularity():
    arm = Planar2R(1.0, 1.0)
    robot = robots.planar(1.0, 1.0)
    J = np.asarray(robot.jacob0([0.0, 0.0]))[:2, :]
    assert np.isclose(arm.det_jacobian(0.0), np.linalg.det(J), atol=1e-12)
    assert np.isinf(amplification(analyze(J)))
