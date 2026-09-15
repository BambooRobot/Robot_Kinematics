"""@file test_panda_fk.py
@brief Franka Panda 正运动学：几何事实、两套参数化等价、关节限位提示。
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from robotkinematics.core import panda
from robotkinematics.core.se3 import SE3


@pytest.fixture
def chain():
    return panda.panda_urdf_chain()


def test_panda_has_seven_joints(chain):
    assert chain.n_joints == 7
    assert len(panda.PANDA_JOINT_NAMES) == 7


def test_zero_pose_flange_height_is_the_sum_of_link_offsets(chain):
    """全零姿态下手臂是竖直的：法兰高度 = 0.333 + 0.316 + 0.384 = 1.033 m。

    这三个数是 URDF 里三段轴间距离，不是随口写的常数。
    """
    flange = replace(chain, tool=SE3.identity()).fk(panda.PANDA_Q_ZERO)
    assert np.isclose(flange.t[2], 0.333 + 0.316 + 0.384, atol=1e-12)
    assert np.isclose(flange.t[0], 0.088, atol=1e-12)
    assert np.allclose(flange.t[1], 0.0, atol=1e-12)


def test_zero_pose_tool_is_107mm_below_the_flange(chain):
    """工具坐标系在法兰前方 107mm；全零姿态下它朝向 -z，所以在法兰下面。"""
    tool = chain.fk(panda.PANDA_Q_ZERO)
    assert np.isclose(tool.t[2], 1.033 - 0.107, atol=1e-9)
    assert np.allclose(tool.t, [0.088, 0.0, 0.926], atol=1e-3)


def test_ppt_goal_pose_position(chain):
    """课件 panda_cases.csv 的 ppt_goal 姿态。"""
    tool = chain.fk(panda.PANDA_Q_PPT_GOAL)
    assert np.allclose(tool.t, [0.4531, 0.0, 0.5619], atol=5e-5)


def test_end_effector_never_exceeds_the_triangle_inequality_bound(chain):
    """末端离原点的距离不可能超过所有固定平移的长度之和（三角不等式）。"""
    bound = sum(float(np.linalg.norm(link.offset.t)) for link in chain.links) + float(
        np.linalg.norm(chain.tool.t)
    )
    rng = np.random.default_rng(3)
    for q in rng.uniform(-np.pi, np.pi, size=(200, 7)):
        assert np.linalg.norm(chain.fk(q).t) <= bound + 1e-9


def test_ets_and_mdh_give_identical_fk():
    """两套独立实现的参数化必须逐位一致 —— 这是本次“自研”最硬的一条交叉验证。"""
    ets = panda.panda_urdf_chain()
    dh = panda.panda_dh_robot()
    rng = np.random.default_rng(4)
    for q in rng.uniform(-np.pi, np.pi, size=(200, 7)):
        assert ets.fk(q).is_close(dh.fk(q), tol=1e-12)
    # 课件里的两个姿态也验一遍
    for q in (panda.PANDA_Q_ZERO, panda.PANDA_Q_PPT_GOAL, panda.PANDA_Q_IK_SEED):
        assert ets.fk(q).is_close(dh.fk(q), tol=1e-12)


def test_dh_converted_chain_also_agrees():
    dh = panda.panda_dh_robot()
    converted = dh.to_serial_chain()
    rng = np.random.default_rng(5)
    for q in rng.uniform(-np.pi, np.pi, size=(50, 7)):
        assert converted.fk(q).is_close(dh.fk(q), tol=1e-12)


def test_joint_limits_flag_the_mathematically_impossible_zero_pose():
    """课件的“全零姿态”其实越过了 q4 的关节限位（[-3.0718, -0.0698]）。

    这不影响它作为数学参考位形，但真机摆不出来 —— CLI 会据此给出提示。
    """
    assert panda.joints_out_of_limits(panda.PANDA_Q_ZERO) == ["panda_joint4"]
    assert panda.joints_out_of_limits(panda.PANDA_Q_PPT_GOAL) == []
    assert panda.joints_out_of_limits(panda.PANDA_Q_IK_SEED) == []


def test_default_model_is_the_urdf_chain():
    assert panda.panda().name.startswith("Franka Panda (URDF")


# ── 夹爪 TCP 帧：课件日志那 0.103 m 差异的真实来源 ─────────────────────────


def test_hand_tcp_offset_is_a_fixed_transform():
    """法兰 → 夹爪 TCP 是**固定**变换：换任何姿态，两者的相对位姿都不变。"""
    flange = panda.panda_urdf_chain()
    hand = panda.panda_hand_tcp_chain()
    rng = np.random.default_rng(9)
    for q in rng.uniform(-1.5, 1.5, size=(20, 7)):
        relative = flange.fk(q).inverse() * hand.fk(q)
        assert relative.is_close(panda.PANDA_HAND_TCP_OFFSET, tol=1e-12)


def test_hand_tcp_frame_reproduces_the_course_log():
    """课件 run_logs 记的是夹爪 TCP 帧；本项目加上夹爪后必须复现它。

    这是那 0.103 m 之谜的结论：课件 `robot.fkine(q)` 默认返回夹爪 TCP，
    本项目默认给法兰帧 —— 两个都没错，只是“末端”的定义不同。
    """
    T = panda.panda_hand_tcp_chain().fk(panda.PANDA_Q_ZERO)
    assert np.allclose(T.t, panda.COURSE_LOG_ZERO_POSE_TOOL_T, atol=5e-5)
    # 课件日志同时记录了这个旋转矩阵（含绕 z 的 45° 耦合），也必须对上
    expected_R = np.array([[0.7071, 0.7071, 0.0], [0.7071, -0.7071, 0.0], [0.0, 0.0, -1.0]])
    assert np.allclose(T.R, expected_R, atol=1e-4)


def test_flange_and_tcp_differ_by_1034_millimetres_at_the_zero_pose():
    flange = panda.panda_urdf_chain().fk(panda.PANDA_Q_ZERO)
    tcp = panda.panda_hand_tcp_chain().fk(panda.PANDA_Q_ZERO)
    assert np.isclose(flange.t[2] - tcp.t[2], 0.1034, atol=1e-12)


def test_hand_tcp_chain_keeps_the_same_joints():
    """加夹爪只改末端帧，关节表必须一模一样（ETSLink 是 frozen dataclass，可直接比较）。"""
    flange = panda.panda_urdf_chain()
    hand = panda.panda_hand_tcp_chain()
    assert hand.links == flange.links
    assert hand.n_joints == flange.n_joints == 7
