"""@file test_panda_ik.py
@brief 数值 IK：收敛、冗余、不可达、零空间次要任务，以及和平面对照。
"""

from __future__ import annotations

import numpy as np
import pytest

from robotkinematics.core import panda
from robotkinematics.core.exceptions import KinematicsError, NotConvergedError
from robotkinematics.core.ik_solvers import (
    planar_chain,
    pose_error,
    random_seeds,
    solve_dls,
    solve_dls_or_raise,
    solve_multi_seed,
)
from robotkinematics.core.planar2r import Planar2R
from robotkinematics.core.se3 import SE3


@pytest.fixture
def chain():
    return panda.panda_urdf_chain()


def test_solve_generates_a_target_by_fk_then_recovers_it(chain):
    """课件 09 的思路：先用 FK 造一个一定可达的目标，再让 IK 自己找回来。"""
    q_true = panda.PANDA_Q_PPT_GOAL
    target = chain.fk(q_true)
    result = solve_dls(chain, target, np.zeros(7))
    assert result.success
    assert result.position_error < 1e-9
    assert result.orientation_error < 1e-9
    assert np.allclose(chain.fk(result.q).t, target.t, atol=1e-8)


def test_redundant_robot_reaches_the_same_pose_with_different_joint_angles(chain):
    """7 自由度对 6 维任务有冗余：课件里 ikine_LM 解出的 q 和生成目标的 q 不同。"""
    q_true = panda.PANDA_Q_PPT_GOAL
    target = chain.fk(q_true)
    results = solve_multi_seed(chain, target, random_seeds(chain, 4, seed=7, scale=1.5))
    solved = [r for r in results if r.success]
    assert len(solved) >= 2
    for r in solved:
        assert np.allclose(chain.fk(r.q).t, target.t, atol=1e-7)
    # 它们确实是不同的关节角配置，而不是同一个解
    assert np.linalg.norm(solved[0].q - solved[1].q) > 0.1


def test_unreachable_target_fails_cleanly(chain):
    """目标远在工作空间之外：不收敛是可预期的结论，不是崩溃。"""
    far = SE3(chain.fk(panda.PANDA_Q_ZERO).R, np.array([3.0, 0.0, 0.0]))
    result = solve_dls(chain, far, np.zeros(7), max_iter=50)
    assert not result.success
    assert result.position_error > 1.0
    with pytest.raises(NotConvergedError):
        solve_dls_or_raise(chain, far, np.zeros(7), max_iter=50)


def test_nullspace_task_pulls_the_solution_toward_the_reference_posture(chain):
    """同样的末端位姿，用多出来的自由度让关节角更靠近参考姿态。"""
    target = chain.fk(panda.PANDA_Q_PPT_GOAL)
    q_ref = panda.PANDA_Q_IK_SEED
    plain = solve_dls(chain, target, np.zeros(7))
    guided = solve_dls(chain, target, np.zeros(7), q_ref=q_ref, nullspace_gain=0.5)
    assert plain.success and guided.success
    assert np.linalg.norm(guided.q - q_ref) < np.linalg.norm(plain.q - q_ref)


def test_nullspace_improvement_bound_explains_why_the_payoff_can_be_tiny(chain):
    """次要任务的收益上限远小于“能动分量”的长度 —— 这是最容易误判的一点。

    能动分量 |N·(q_ref−q)| 只是“偏好里能动的那部分”，它与整条差值向量的夹角
    决定了能省下多少距离。本例偏好方向与零空间接近正交，于是：
        能动分量 0.023 rad，实际只缩短 2e-4 rad。
    """
    from robotkinematics.core.ik_solvers import nullspace_improvement_bound

    target = chain.fk(panda.PANDA_Q_PPT_GOAL)
    q_ref = panda.PANDA_Q_IK_SEED
    plain = solve_dls(chain, target, np.zeros(7))
    guided = solve_dls(chain, target, np.zeros(7), q_ref=q_ref, nullspace_gain=0.5)

    movable, bound = nullspace_improvement_bound(chain, plain.q, q_ref)
    improvement = float(np.linalg.norm(plain.q - q_ref) - np.linalg.norm(guided.q - q_ref))

    assert improvement > 0.0, "次要任务应当确实起了一点作用"
    assert bound < movable, "缩短量上限必须小于能动分量本身（两者只有在方向重合时才相等）"
    # 同一量级即可：零空间随姿态转动，实际收益可以略高于直线估算
    assert improvement < bound * 10.0
    # 两者都远小于到参考姿态的距离 —— 冗余自由度不是万能的
    assert movable < 0.1 * float(np.linalg.norm(plain.q - q_ref))


def test_pose_error_is_zero_for_identical_poses(chain):
    T = chain.fk(panda.PANDA_Q_PPT_GOAL)
    assert np.allclose(pose_error(T, T), 0.0, atol=1e-15)


def test_pose_error_separates_position_and_orientation(chain):
    T = chain.fk(panda.PANDA_Q_PPT_GOAL)
    moved = SE3(T.R, T.t + np.array([0.0, 0.0, 0.1]))
    e = pose_error(moved, T)
    assert np.isclose(np.linalg.norm(e[:3]), 0.1)
    assert np.allclose(e[3:], 0.0, atol=1e-12)


def test_numeric_ik_agrees_with_the_analytic_two_link_solution():
    """数值 IK 与二连杆的闭式解必须落在同一批点上 —— 两条完全不同的路径。"""
    arm = Planar2R(l1=1.0, l2=1.0)
    chain = planar_chain()
    target_xy = np.array([1.0, 1.0])
    analytic = arm.ik(*target_xy)
    assert analytic

    target = SE3.Trans(target_xy[0], target_xy[1], 0.0)
    for q1, q2 in analytic:
        result = solve_dls(chain, target, np.array([q1, q2]) + 0.2, rows=slice(0, 2), max_iter=500)
        assert result.success
        assert np.allclose(chain.fk(result.q).t[:2], target_xy, atol=1e-7)


def test_wrong_seed_dimension_is_rejected(chain):
    with pytest.raises(KinematicsError):
        solve_dls(chain, chain.fk(panda.PANDA_Q_ZERO), np.zeros(5))


def test_iteration_count_is_reported(chain):
    target = chain.fk(panda.PANDA_Q_PPT_GOAL)
    result = solve_dls(chain, target, np.zeros(7))
    assert 0 < result.iterations < 200
    assert "迭代" in result.summary()
