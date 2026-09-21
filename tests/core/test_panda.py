"""@file test_panda.py

@brief Panda：库的 FK / IK / 雅可比用法对不对（数学是库的事，用法是我们的责任）。
"""

from __future__ import annotations

import numpy as np
import pytest

from robotkinematics.core import robots
from robotkinematics.core.singularity import analysis_of, analyze, manipulability

Q_GOAL = np.array([0.0, -0.4, 0.0, -2.2, 0.0, 2.0, 0.7853981634])
Q_SEED = np.array([0.0, -0.3, 0.0, -2.2, 0.0, 2.0, 0.8])


@pytest.fixture
def robot():
    """被测对象：Panda 配 TCP 末端帧（夹爪尖）。"""
    return robots.panda(robots.TCP)


def test_fk_matches_course_numbers(robot):
    """FK 数值与课件对得上。"""
    assert np.allclose(
        np.asarray(robots.end_pose(robot, Q_GOAL, robots.TCP).t), [0.4737, 0.0, 0.4606], atol=5e-5
    )


def test_jacobian_shape_is_6_by_7(robot):
    """6x7 是冗余机械臂的标志。"""
    assert np.asarray(robot.jacob0(Q_SEED)).shape == (6, 7)


def test_jacobian_last_column_is_pure_rotation(robot):
    """第 7 列线速度恒为 0 —— 工具坐标系挂在第 7 关节的轴上。"""
    J = np.asarray(robot.jacob0(Q_SEED))
    assert np.allclose(J[:3, 6], 0.0, atol=1e-9)
    assert np.isclose(np.linalg.norm(J[3:, 6]), 1.0)


def test_ik_round_trip_recovers_the_target(robot):
    """先用 FK 造一个一定可达的目标，再让 IK 找回来。"""
    target = robots.end_pose(robot, Q_GOAL, robots.TCP)
    solution = robot.ikine_LM(target, q0=Q_SEED, tol=1e-9)
    assert solution.success
    reached = np.asarray(robots.end_pose(robot, np.asarray(solution.q), robots.TCP).t)
    assert np.linalg.norm(reached - np.asarray(target.t)) < 1e-5


def test_redundant_robot_reaches_the_same_pose_from_different_seeds(robot):
    """7 自由度冗余：不同初值 → 不同关节角，但到位姿相同。"""
    target = robots.end_pose(robot, Q_GOAL, robots.TCP)
    rng = np.random.default_rng(0)
    solved = []
    for seed in (np.zeros(7), rng.uniform(-1.5, 1.5, 7), rng.uniform(-1.5, 1.5, 7)):
        solution = robot.ikine_LM(target, q0=seed, tol=1e-9)
        if solution.success:
            solved.append(np.asarray(solution.q))
    assert len(solved) >= 2
    for q in solved:
        assert np.allclose(
            np.asarray(robots.end_pose(robot, q, robots.TCP).t), np.asarray(target.t), atol=1e-4
        )
    assert np.linalg.norm(solved[0] - solved[1]) > 0.01


def test_analysis_and_library_manipulability_agree(robot):
    """我们算的可操作度与库的 manipulability 必须一致。"""
    assert np.isclose(analysis_of(robot, Q_SEED).manipulability, manipulability(robot, Q_SEED))


def test_zero_pose_is_singular(robot):
    """全零是奇异位形：手臂竖直、腕部对齐。"""
    report = analyze(np.asarray(robot.jacob0(np.zeros(7))))
    assert report.is_singular
    assert report.manipulability < 1e-12


def test_flange_and_tcp_differ_at_zero():
    """零位姿下法兰与夹爪 TCP 差约 103.4mm。"""
    robot = robots.panda(robots.FLANGE)
    flange = np.asarray(robots.end_pose(robot, np.zeros(7), robots.FLANGE).t)
    tcp = np.asarray(robots.end_pose(robot, np.zeros(7), robots.TCP).t)
    assert np.isclose(flange[2], 0.9260, atol=5e-4)
    assert np.isclose(tcp[2], 0.8226, atol=5e-4)
