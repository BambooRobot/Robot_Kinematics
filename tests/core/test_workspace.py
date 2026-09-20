"""@file test_workspace.py

@brief 可达性采样：库没有这个 API（`robot.reach` 实测返回 0），所以这里是自研的。
"""

from __future__ import annotations

import numpy as np
import pytest

from robotkinematics.core import robots, workspace


@pytest.fixture
def robot():
    """被测对象：Panda 配 FLANGE 末端帧 —— 可达性只关心腕部到哪，不含夹爪长度。"""
    return robots.panda(robots.FLANGE)


@pytest.fixture
def limits(robot):
    """关节限位表：采样必须落在它里面，否则「可达」会被明显夸大。"""
    return robots.joint_limits(robot)


def test_sampling_is_reproducible(robot):
    """固定 seed：同样的输入必须给同样的结论，否则失败分类就成了掷骰子。"""
    a = workspace.sample_joint_space(robot, 100, seed=42)
    b = workspace.sample_joint_space(robot, 100, seed=42)
    assert np.array_equal(a, b)
    assert not np.array_equal(a, workspace.sample_joint_space(robot, 100, seed=43))


def test_sampling_respects_limits(robot, limits):
    """采样点必须逐个落在关节限位内 —— 越界的样本会把工作空间算大一圈。"""
    qs = workspace.sample_joint_space(robot, 500, seed=0, limits=limits)
    assert (qs >= limits[:, 0]).all()
    assert (qs <= limits[:, 1]).all()


def test_limits_shape_is_validated(robot):
    """限位形状不对要指名道姓地报「关节限位」，不能靠 numpy 广播糊过去。"""
    with pytest.raises(ValueError, match="关节限位"):
        workspace.sample_joint_space(robot, 10, limits=np.zeros((3, 2)))


def test_directional_reach_is_reproducible(robot, limits):
    """固定 seed 才可复现：同样的方向与样本数必须给出同样的半径与摘要，否则结论是掷骰子。"""
    first = workspace.directional_reach(robot, [0, 0, 1], samples=500, seed=7, limits=limits)
    second = workspace.directional_reach(robot, [0, 0, 1], samples=500, seed=7, limits=limits)
    assert first.radius == second.radius
    assert first.summary() == second.summary()


def test_workspace_is_not_a_sphere(robot, limits):
    """沿不同方向的可达距离不同 —— "工作空间不是球"的定量说明。"""
    up = workspace.directional_reach(robot, [0, 0, 1], samples=4000, seed=0, limits=limits)
    sideways = workspace.directional_reach(robot, [1, 0, 0], samples=4000, seed=0, limits=limits)
    assert up.radius > sideways.radius
    assert sideways.radius < workspace.max_reach(robot, samples=4000, seed=0, limits=limits)


def test_direction_is_normalized(robot):
    """方向向量先归一化再用 —— 半径不能随输入向量的模长变化。"""
    estimate = workspace.directional_reach(robot, [0, 0, 123], samples=200, seed=0)
    assert np.isclose(np.linalg.norm(estimate.direction), 1.0)


def test_zero_direction_is_rejected(robot):
    """零方向没有意义，必须报错 —— 归一化时会除零，静默返回的半径是垃圾数。"""
    with pytest.raises(ValueError, match="方向"):
        workspace.directional_reach(robot, [0, 0, 0])


def test_margin_sign_tells_inside_outside(robot, limits):
    """margin 的符号就是「点在可达边界内还是外」的判据，供抓取规划做准入检查。"""
    estimate = workspace.directional_reach(robot, [0, 0, 1], samples=2000, seed=0, limits=limits)
    assert estimate.margin(estimate.radius - 0.01) > 0
    assert estimate.margin(estimate.radius + 0.01) < 0


def test_batch_positions_match_single_fk(robot):
    """批量 fkine（采样用的）与逐个 fkine 必须一致。"""
    rng = np.random.default_rng(0)
    qs = rng.uniform(-1, 1, size=(30, 7))
    batch = robots.end_positions(robot, qs, robots.FLANGE)
    single = np.array([np.asarray(robots.end_pose(robot, q, robots.FLANGE).t) for q in qs])
    assert np.allclose(batch, single, atol=1e-12)


def test_single_pose_also_returns_a_matrix(robot):
    """单个姿态也要返回 (1,3) —— 库的返回值类型会骗人，这里钉住形状。"""
    positions = robots.end_positions(robot, np.zeros(7), robots.FLANGE)
    assert positions.shape == (1, 3)
