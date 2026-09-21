"""@file test_robots.py

@brief 模型构造与查询：库的 API 调对了没有（课件数值是硬标准）。
"""

from __future__ import annotations

import numpy as np
import pytest

from robotkinematics.core import robots


@pytest.fixture
def panda():
    """被测对象：Panda 配 FLANGE 末端帧 —— 本文件要直接对课件日志的数值，必须用法兰帧。"""
    return robots.panda(robots.FLANGE)


def test_panda_has_seven_joints(panda):
    """7 关节这个前提不成立，后面关于冗余、零空间、次要任务的结论全都没意义。"""
    assert panda.n == 7
    assert len(robots.joint_names(panda)) == 7


def test_flange_and_tcp_differ_by_1034_millimetres(panda):
    """两种末端定义差 103.4mm —— 这就是课件日志那 0.103 m 之谜。"""
    flange = np.asarray(robots.end_pose(panda, np.zeros(7), robots.FLANGE).t, dtype=float)
    tcp = np.asarray(robots.end_pose(panda, np.zeros(7), robots.TCP).t, dtype=float)
    assert np.allclose(flange, [0.088, 0.0, 0.926], atol=1e-6)
    assert np.allclose(tcp, [0.088, 0.0, 0.8226], atol=1e-6)


@pytest.mark.parametrize(
    "q, expected",
    [
        (np.zeros(7), [0.088, 0.0, 0.926]),  # 课件日志零位姿（法兰帧）
        (np.array([0.0, -0.4, 0.0, -2.2, 0.0, 2.0, 0.7853981634]), [0.4531, 0.0, 0.5619]),
    ],
)
def test_fk_matches_course_numbers(panda, q, expected):
    """课件日志里的两组位姿（零位姿 + 目标位姿）都要在位 —— 数值是硬标准。"""
    assert np.allclose(np.asarray(robots.end_pose(panda, q, robots.FLANGE).t), expected, atol=5e-5)


def test_tcp_frame_matches_the_course_log(panda):
    """课件 run_logs 里的零位姿（0.8226）= 库的默认末端，即夹爪 TCP。"""
    tcp = np.asarray(robots.end_pose(panda, np.zeros(7), robots.TCP).t, dtype=float)
    assert np.allclose(tcp, [0.088, 0.0, 0.8226], atol=5e-5)


def test_joint_limits_come_from_the_library(panda):
    """限位取自库里的 Franka 数据表；q4 区间不含 0，所以「全零姿态」真机根本摆不出来。"""
    limits = robots.joint_limits(panda)
    assert limits.shape == (7, 2)
    # Franka 数据表：q4 的区间不含 0，所以"全零姿态"真机摆不出来
    assert limits[3][1] < 0 < limits[3][0] * -1 + limits[3][1]  # 上界为负
    assert robots.out_of_limits(np.zeros(7), limits) == [3]
    assert robots.out_of_limits(np.array([0, -0.4, 0, -2.2, 0, 2.0, 0.785]), limits) == []


def test_limit_margin_sign(panda):
    """限位余量的符号约定：负数=越界、正数=安全 —— 报告里的体检结论就靠它。"""
    limits = robots.joint_limits(panda)
    assert robots.limit_margin(np.zeros(7), limits) < 0  # 越界
    assert robots.limit_margin(np.array([0, -0.4, 0, -2.2, 0, 2.0, 0.785]), limits) > 0


def test_link_points_shape_for_skeleton(panda):
    """骨架点是 (N,3) 且首点为基座原点 —— 形状错了画出来的机械臂就是歪的。"""
    points = robots.link_points(panda, np.zeros(7))
    assert points.shape[1] == 3
    assert points.shape[0] >= 3  # 至少 base + 若干关节 + 末端
    assert np.allclose(points[0], np.zeros(3))  # 第一个点是 base 原点


def test_unknown_frame_is_rejected():
    """未知末端帧名必须报错，不能悄悄退回默认帧 —— 那正是 0.103m 对不上的成因。"""
    with pytest.raises(ValueError):
        robots.panda("nope")
