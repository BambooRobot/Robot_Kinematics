"""@file test_planar.py
@brief 二连杆：唯一保留的手推公式（解析 FK/IK），以及与库的互相验证。
"""

from __future__ import annotations

import numpy as np
import pytest

from robotkinematics.core import robots
from robotkinematics.core.planar2r import Planar2R

DEG = np.deg2rad


@pytest.fixture
def arm():
    return Planar2R(l1=1.0, l2=1.0)


@pytest.mark.parametrize(
    "q1_deg, q2_deg, expected",
    [
        (0.0, 90.0, [1.0, 1.0]),  # 课件算例
        (30.0, 60.0, [0.866, 1.5]),
        (60.0, -30.0, [1.366, 1.366]),
        (0.0, 0.0, [2.0, 0.0]),
    ],
)
def test_fk_matches_course_cases(arm, q1_deg, q2_deg, expected):
    assert np.allclose(arm.fk(DEG(q1_deg), DEG(q2_deg)), expected, atol=5e-4)


def test_fk_agrees_with_the_library(arm):
    """闭式解与库的 DHRobot 必须一致 —— 这是"我们手写的公式没错"的直接证据。"""
    robot = robots.planar(arm.l1, arm.l2)
    rng = np.random.default_rng(0)
    for q in rng.uniform(-np.pi, np.pi, size=(50, 2)):
        assert np.allclose(arm.fk(*q), np.asarray(robot.fkine(q).t)[:2], atol=1e-12)


def test_ik_returns_both_elbow_configurations(arm):
    """课件算例：目标 (1,1) 有两组解 —— 库只给一组，这正是保留闭式解的原因。"""
    solutions = arm.ik(1.0, 1.0)
    assert len(solutions) == 2
    assert np.allclose(np.rad2deg(solutions[0]), [0.0, 90.0], atol=1e-9)
    assert np.allclose(np.rad2deg(solutions[1]), [90.0, -90.0], atol=1e-9)


def test_every_ik_solution_passes_fk_verification(arm):
    for target in [(1.0, 1.0), (0.5, 0.8), (-1.2, 0.3), (0.0, 1.5), (1.9, 0.0)]:
        solutions = arm.ik(*target)
        assert solutions
        for q1, q2 in solutions:
            assert np.allclose(arm.fk(q1, q2), target, atol=1e-9)


def test_ik_reports_no_solution_when_target_is_too_far(arm):
    assert arm.ik(3.0, 0.0) == []  # 课件里的无解例子


def test_on_the_outer_boundary_the_two_solutions_degenerate(arm):
    solutions = arm.ik(2.0, 0.0)
    assert len(solutions) == 2
    assert np.allclose(solutions[0], solutions[1], atol=1e-9)


def test_det_jacobian_closed_form_matches_the_library(arm):
    """det(J) = L1·L2·sin(q2)（手推）必须等于库算出来的雅可比行列式。"""
    robot = robots.planar(arm.l1, arm.l2)
    for q2_deg in (90.0, 30.0, 10.0, 0.0, -30.0, 180.0):
        q2 = DEG(q2_deg)
        J = np.asarray(robot.jacob0([DEG(37), q2]))[:2, :]
        assert np.isclose(arm.det_jacobian(q2), float(np.linalg.det(J)), atol=1e-9)


def test_invalid_link_length_is_rejected():
    with pytest.raises(ValueError):
        Planar2R(l1=0.0, l2=1.0)
