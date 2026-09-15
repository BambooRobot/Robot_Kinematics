"""@file test_planar_fk.py
@brief 二连杆正运动学：课件 PPT 的算例 + 几何自洽性。
"""

from __future__ import annotations

import numpy as np
import pytest

from robotkinematics.core.exceptions import KinematicsError
from robotkinematics.core.planar2r import Planar2R

DEG = np.deg2rad


@pytest.fixture
def arm() -> Planar2R:
    return Planar2R(l1=1.0, l2=1.0)


@pytest.mark.parametrize(
    "q1_deg, q2_deg, expected",
    [
        (0.0, 90.0, [1.0, 1.0]),  # 课件算例：第一杆向右，第二杆向上
        (30.0, 60.0, [0.866, 1.5]),  # 练习 2
        (60.0, -30.0, [1.366, 1.366]),  # 练习 2
        (0.0, 0.0, [2.0, 0.0]),  # 完全伸直
        (90.0, -90.0, [1.0, 1.0]),  # 与第一组姿态不同但末端同点
    ],
)
def test_fk_matches_course_cases(arm, q1_deg, q2_deg, expected):
    assert np.allclose(arm.fk(DEG(q1_deg), DEG(q2_deg)), expected, atol=5e-4)


def test_joint_points_match_course_ascii_demo(arm):
    """课件 04 的默认参数 q1=q2=45°：elbow=[0.7071,0.7071]，tool=[0.7071,1.7071]。"""
    base, elbow, tool = arm.joint_points(DEG(45), DEG(45))
    assert np.allclose(base, [0.0, 0.0])
    assert np.allclose(elbow, [0.7071, 0.7071], atol=5e-5)
    assert np.allclose(tool, [0.7071, 1.7071], atol=5e-5)
    assert np.allclose(tool, arm.fk(DEG(45), DEG(45)), atol=1e-12)


def test_fk_never_exceeds_the_sum_of_link_lengths(arm):
    """工作空间是以 base 为心、半径 L1+L2 的圆（内半径 |L1-L2|）。"""
    for q1 in np.linspace(-np.pi, np.pi, 13):
        for q2 in np.linspace(-np.pi, np.pi, 13):
            assert np.linalg.norm(arm.fk(q1, q2)) <= arm.l1 + arm.l2 + 1e-12


def test_fk_is_continuous_in_joint_angles(arm):
    """关节角微小变化只能引起末端微小移动 —— 这是雅可比存在的前提。"""
    p0 = arm.fk(0.3, 0.4)
    p1 = arm.fk(0.3 + 1e-6, 0.4 + 1e-6)
    assert np.linalg.norm(p1 - p0) < 1e-5


def test_invalid_link_length_is_rejected():
    with pytest.raises(KinematicsError):
        Planar2R(l1=0.0, l2=1.0)
    with pytest.raises(KinematicsError):
        Planar2R(l1=1.0, l2=-0.5)
