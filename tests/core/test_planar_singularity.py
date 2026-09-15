"""@file test_planar_singularity.py
@brief 奇异点：det / cond 的数值、闭式公式，以及奇异时为什么解不出关节微调量。
"""

from __future__ import annotations

from itertools import pairwise

import numpy as np
import pytest

from robotkinematics.core.exceptions import SingularPoseError
from robotkinematics.core.planar2r import Planar2R

DEG = np.deg2rad


@pytest.fixture
def arm() -> Planar2R:
    return Planar2R(l1=1.0, l2=1.0)


@pytest.mark.parametrize(
    "q2_deg, expected_det",
    [(90.0, 1.0), (60.0, 0.866025), (30.0, 0.5), (-30.0, -0.5), (0.0, 0.0)],
)
def test_det_jacobian_equals_l1_l2_sin_q2(arm, q2_deg, expected_det):
    """课件结论：det(J) = L1·L2·sin(q2)，与 q1 无关。"""
    q2 = DEG(q2_deg)
    assert np.isclose(arm.det_jacobian(q2), expected_det, atol=1e-6)
    assert np.isclose(np.linalg.det(arm.jacobian(DEG(37), q2)), expected_det, atol=1e-6)


@pytest.mark.parametrize(
    "q2_deg, expected_cond",
    [(90.0, 2.62), (30.0, 9.36), (10.0, 28.58)],
)
def test_condition_number_matches_course_table(arm, q2_deg, expected_cond):
    """课件 07 的终端表格：q2=90/30/10° 时 cond(J) = 2.62 / 9.36 / 28.58。"""
    assert np.isclose(arm.condition_number(DEG(0), DEG(q2_deg)), expected_cond, atol=5e-3)


def test_condition_number_explodes_at_singularity(arm):
    assert arm.condition_number(DEG(0), DEG(0)) > 1e12
    assert not np.isfinite(arm.condition_number(DEG(0), DEG(0))) or (
        arm.condition_number(DEG(0), DEG(0)) > 1e12
    )


def test_det_approaches_zero_as_the_arm_straightens(arm):
    """q2 从 90° 往 0° 走，两杆逐渐共线，|det(J)| 单调降到 0。"""
    dets = [abs(arm.det_jacobian(DEG(q))) for q in (90.0, 30.0, 10.0, 0.0)]
    assert all(a > b for a, b in pairwise(dets))
    assert dets[-1] == 0.0


def test_is_singular_flags_collinear_configurations(arm):
    assert arm.is_singular(DEG(0))
    assert arm.is_singular(DEG(180))
    assert not arm.is_singular(DEG(1))
    assert not arm.is_singular(DEG(90))


def test_solve_step_refuses_to_return_a_meaningless_joint_step(arm):
    """奇异位姿下 J 不可逆：与其返回一个巨大的 dq，不如明确报错。"""
    with pytest.raises(SingularPoseError):
        arm.solve_step(DEG(0), DEG(0), np.array([0.0, 0.1]))


def test_tangential_motion_still_works_near_singularity(arm):
    """奇异只锁死一个方向（沿连杆向外），切向运动仍然能解出来。"""
    dq = arm.solve_step(DEG(0), DEG(0.0 + 1e-6), np.array([0.0, 1e-3]))
    assert np.all(np.isfinite(dq))


def test_singularity_is_not_an_error_but_a_property_of_the_pose(arm):
    """同一组关节角在奇异点上，FK 依然正常 —— 奇异影响的是“微动”，不是“位置”。"""
    assert np.allclose(arm.fk(DEG(0), DEG(0)), [2.0, 0.0])
    assert arm.ik(2.0, 0.0)
