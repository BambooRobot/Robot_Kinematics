"""@file test_panda.py
@brief Panda：库的 FK / IK / 雅可比**用法**对不对（数学是库的事，用法是我们的责任）。
"""

from __future__ import annotations

import numpy as np
import pytest

from robotkinematics.core import robots
from robotkinematics.core.singularity import analysis_of, analyze, manipulability
from robotkinematics.usecases import panda as panda_uc

Q_GOAL = np.array([0.0, -0.4, 0.0, -2.2, 0.0, 2.0, 0.7853981634])
Q_SEED = np.array([0.0, -0.3, 0.0, -2.2, 0.0, 2.0, 0.8])


@pytest.fixture
def robot():
    return robots.panda(robots.TCP)


def test_fk_matches_course_numbers(robot):
    assert np.allclose(
        np.asarray(robots.end_pose(robot, Q_GOAL, robots.TCP).t), [0.4737, 0.0, 0.4606], atol=5e-5
    )


def test_jacobian_shape_is_6_by_7(robot):
    assert np.asarray(robot.jacob0(Q_SEED)).shape == (6, 7)


def test_jacobian_last_column_is_pure_rotation(robot):
    """第 7 列线速度恒为 0 —— 工具坐标系挂在第 7 关节的轴上。"""
    J = np.asarray(robot.jacob0(Q_SEED))
    assert np.allclose(J[:3, 6], 0.0, atol=1e-9)
    assert np.isclose(np.linalg.norm(J[3:, 6]), 1.0)


def test_ik_round_trip_recovers_the_target(robot):
    """先用 FK 造一个一定可达的目标，再让 IK 找回来（课件 09 的思路）。"""
    target = robots.end_pose(robot, Q_GOAL, robots.TCP)
    solution = robot.ikine_LM(target, q0=Q_SEED, tol=1e-9)
    assert solution.success
    reached = np.asarray(robots.end_pose(robot, np.asarray(solution.q), robots.TCP).t)
    # 库的 tol 是关节步长判据，换算到位姿残差是 1e-7 量级（实测 5.2e-7）
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
        # ⚠️ 容差按实测取 1e-4：库的 tol 是**关节步长**判据，不直接约束位姿精度，
        #    不同初值收敛到的位姿残差在 1e-7 ~ 1e-5 之间浮动。
        assert np.allclose(
            np.asarray(robots.end_pose(robot, q, robots.TCP).t), np.asarray(target.t), atol=1e-4
        )
    # 差值不大（0.07 rad 量级）—— 7 自由度只有一个零空间方向，沿它的位移本来就有界，
    # 这与本项目"次要任务收益很小"的实测结论一致（见 docs/GUIDE.md 第 7 章）
    assert np.linalg.norm(solved[0] - solved[1]) > 0.01


def test_analysis_and_library_manipulability_agree(robot):
    """我们算的可操作度（numpy SVD）与库的 manipulability 必须一致。"""
    assert np.isclose(analysis_of(robot, Q_SEED).manipulability, manipulability(robot, Q_SEED))


def test_zero_pose_is_singular(robot):
    """课件姿态 1（全零）是奇异位形：手臂竖直、腕部对齐。"""
    report = analyze(np.asarray(robot.jacob0(np.zeros(7))))
    assert report.is_singular
    assert report.manipulability < 1e-12


def test_ik_report_mentions_the_frame_and_verifies_with_fk(robot):
    target = robots.end_pose(robot, Q_GOAL, robots.TCP)
    report = panda_uc.ik_report(robot, target, Q_SEED, show_redundancy=False)
    assert report.fields["success"] is True
    assert report.fields["position_error"] < 1e-5


def test_fk_report_shows_both_frames(robot):
    """零位姿报告要把两种末端帧并排打出来 —— 这就是那个 0.103 m 之谜的说明。"""
    report = panda_uc.fk_report(robot, np.zeros(7), frame=robots.FLANGE)
    text = " ".join(line for block in report.blocks for line in getattr(block, "lines", ()))
    assert "0.9260" in text and "0.8226" in text
    assert report.fields["out_of_limits"] == [3]  # q4 越限位
    # 齐次矩阵的合法性：最后一行
    A = np.asarray(report.fields["A"])
    assert A.shape == (4, 4) and np.allclose(A[3], [0, 0, 0, 1])
