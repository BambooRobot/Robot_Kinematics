"""@file test_planar_jacobian.py
@brief 二连杆雅可比：PPT 数值、与数值微分的一致性、还有它成立的边界。
"""

from __future__ import annotations

from itertools import pairwise

import numpy as np
import pytest

from robotkinematics.core.numerics import numeric_jacobian
from robotkinematics.core.planar2r import Planar2R

DEG = np.deg2rad


@pytest.fixture
def arm() -> Planar2R:
    return Planar2R(l1=1.0, l2=1.0)


def test_jacobian_matches_ppt_value(arm):
    """课件算例：q1=0°, q2=90° 时 J = [[-1,-1],[1,0]]。"""
    J = arm.jacobian(DEG(0), DEG(90))
    assert np.allclose(J, [[-1.0, -1.0], [1.0, 0.0]], atol=1e-12)


def test_jacobian_matches_numeric_differentiation(arm):
    """解析雅可比必须等于数值微分 —— 这是判定“偏导写没写错”的尺子。"""
    for q1 in (-1.0, -0.3, 0.0, 0.7, 2.0):
        for q2 in (-2.5, -0.8, 0.0, 0.9, 2.9):
            analytic = arm.jacobian(q1, q2)
            numeric = numeric_jacobian(lambda q: arm.fk(q[0], q[1]), np.array([q1, q2]))
            assert np.allclose(analytic, numeric, atol=1e-7)


def test_solve_step_reproduces_ppt_example(arm):
    """课件算例：末端从 (1,1) 微调到 (1,1.1)，解出 dq = [0.1, -0.1] rad = [5.73°,-5.73°]。"""
    dq = arm.solve_step(DEG(0), DEG(90), np.array([0.0, 0.1]))
    assert np.allclose(dq, [0.1, -0.1], atol=1e-9)
    assert np.allclose(np.rad2deg(dq), [5.73, -5.73], atol=1e-2)


def test_linearization_predicts_a_small_move_accurately(arm):
    """J 是局部线性近似：小位移预测得准，位移一大就不准了。"""
    q1, q2 = DEG(0), DEG(90)
    dx = np.array([0.0, 0.001])
    dq = arm.solve_step(q1, q2, dx)
    predicted = arm.jacobian(q1, q2) @ dq
    actual = arm.fk(q1 + dq[0], q2 + dq[1]) - arm.fk(q1, q2)
    assert np.allclose(predicted, dx, atol=1e-12)
    assert np.allclose(actual, dx, atol=1e-6)

    # 同样的 J，用在一个大位移上就偏了 —— 所以 Jacobian 只适合“小步微调”
    big_dq = arm.solve_step(q1, q2, np.array([0.0, 0.5]))
    actual_big = arm.fk(q1 + big_dq[0], q2 + big_dq[1]) - arm.fk(q1, q2)
    assert not np.allclose(actual_big, [0.0, 0.5], atol=1e-3)


def test_worst_case_joint_step_grows_near_singularity(arm):
    """越接近 q2=0°，最坏方向上的放大倍数越大 —— 这就是奇异点前的速度爆炸。

    注意不能拿某个固定方向 dx 的步长来断言单调：放大倍数取决于方向与奇异方向的
    夹角，固定方向的步长完全可能不随 q2 单调。真正单调的是 1/σ_min(J)，
    也就是“所有方向里最坏的那个放大倍数”。
    """
    q2s = [DEG(90), DEG(30), DEG(10), DEG(1)]
    gains = [1.0 / np.linalg.svd(arm.jacobian(DEG(0), q2), compute_uv=False)[-1] for q2 in q2s]
    assert all(g < h for g, h in pairwise(gains))
    assert gains[-1] > 30.0 * gains[0]


def test_jacobian_columns_are_the_two_link_velocity_contributions(arm):
    """J 的第 i 列 = 关节 i 单独以单位角速度转动时，末端的线速度方向。"""
    q1, q2 = DEG(30), DEG(40)
    _, elbow, tool = arm.joint_points(q1, q2)
    base = np.zeros(2)
    # 第 1 列：整条手臂绕 base 转，末端速度 = ω × r，平面内即 r 逆时针转 90°
    r1 = tool - base
    expected_col1 = np.array([-r1[1], r1[0]])
    # 第 2 列：只有第二段绕 elbow 转
    r2 = tool - elbow
    expected_col2 = np.array([-r2[1], r2[0]])
    J = arm.jacobian(q1, q2)
    assert np.allclose(J[:, 0], expected_col1, atol=1e-12)
    assert np.allclose(J[:, 1], expected_col2, atol=1e-12)
