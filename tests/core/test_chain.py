"""@file test_chain.py
@brief 通用串联链：FK / 雅可比 / 关节帧，以及“平面链与二连杆解析式等价”。
"""

from __future__ import annotations

import numpy as np
import pytest

from robotkinematics.core.chain import DHLink, DHRobot, ETSLink, SerialChain
from robotkinematics.core.exceptions import KinematicsError
from robotkinematics.core.ik_solvers import planar_chain
from robotkinematics.core.planar2r import Planar2R
from robotkinematics.core.se3 import SE3

DEG = np.deg2rad


def test_planar_chain_agrees_with_analytic_two_link_fk():
    """同一条二连杆，用“通用链”写和用“解析式”写必须给出同一个末端位置。"""
    arm = Planar2R(l1=0.7, l2=1.3)
    chain = planar_chain(l1=0.7, l2=1.3)
    rng = np.random.default_rng(0)
    for _ in range(50):
        q = rng.uniform(-np.pi, np.pi, size=2)
        assert np.allclose(chain.fk(q).t[:2], arm.fk(*q), atol=1e-12)


def test_planar_chain_jacobian_agrees_with_analytic_two_link_jacobian():
    arm = Planar2R(l1=1.0, l2=1.0)
    chain = planar_chain()
    rng = np.random.default_rng(1)
    for _ in range(20):
        q = rng.uniform(-np.pi, np.pi, size=2)
        # 通用链给的是 6x2（三维空间），二连杆只看平面内的前两行
        assert np.allclose(chain.jacobian(q)[:2, :], arm.jacobian(*q), atol=1e-12)


def test_joint_frames_put_the_second_joint_at_the_elbow():
    arm = Planar2R(l1=1.0, l2=1.0)
    chain = planar_chain()
    q = np.array([DEG(30), DEG(40)])
    frames = chain.joint_frames(q)
    _, elbow, _ = arm.joint_points(*q)
    assert np.allclose(frames[1][:3, 3][:2], elbow, atol=1e-12)
    assert np.allclose(frames[0][:3, 3], np.zeros(3))  # 第一个关节在 base 原点


def test_prismatic_joint_column_is_pure_translation():
    """移动关节的雅可比列只有线速度、没有角速度 —— 它不会让末端转起来。"""
    chain = SerialChain(links=(ETSLink(motion="Tz", name="slide"),))
    q = np.array([0.4])
    assert np.allclose(chain.fk(q).t, [0.0, 0.0, 0.4])
    J = chain.jacobian(q)
    assert np.allclose(J[:3, 0], [0.0, 0.0, 1.0])
    assert np.allclose(J[3:, 0], np.zeros(3))


def test_wrong_number_of_joint_angles_is_rejected():
    chain = planar_chain()
    with pytest.raises(KinematicsError):
        chain.fk(np.zeros(3))
    with pytest.raises(KinematicsError):
        chain.jacobian(np.zeros(1))


def test_unknown_joint_motion_type_is_rejected():
    with pytest.raises(KinematicsError):
        ETSLink(motion="Ry")  # ETS 里关节只用绕 z / 沿 z 两种


def test_empty_chain_is_rejected():
    with pytest.raises(KinematicsError):
        SerialChain(links=())


def test_dh_and_ets_agree_on_a_toy_two_link():
    """先在小机器人上验证两套参数化等价，再去验证 Panda。"""
    dh = DHRobot(
        links=(DHLink(alpha=0.0, a=0.0, d=0.0), DHLink(alpha=0.0, a=1.0, d=0.0)),
        tool=SE3.Trans(1.0, 0.0, 0.0),
    )
    chain = dh.to_serial_chain()
    rng = np.random.default_rng(2)
    for _ in range(10):
        q = rng.uniform(-np.pi, np.pi, size=2)
        assert dh.fk(q).is_close(chain.fk(q), tol=1e-12)


def test_fk_positions_returns_one_point_per_frame_plus_tool():
    chain = planar_chain()
    points = chain.fk_positions(np.array([0.0, 0.0]))
    # base + 两个关节坐标系 + 末端 = 4 个点
    assert points.shape == (4, 3)
    assert np.allclose(points[0], np.zeros(3))
    assert np.allclose(points[-1][:2], [2.0, 0.0])
