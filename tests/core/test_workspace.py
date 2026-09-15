"""@file test_workspace.py
@brief 工作空间采样：批量 FK 必须与逐次 FK 一致，估计必须可复现。
"""

from __future__ import annotations

import numpy as np
import pytest

from robotkinematics.core import panda
from robotkinematics.core.ik_solvers import planar_chain
from robotkinematics.core.workspace import (
    directional_reach,
    fk_positions_batch,
    max_reach,
    sample_joint_space,
)


@pytest.fixture
def chain():
    return panda.panda_urdf_chain()


@pytest.fixture
def limits():
    return np.array(panda.PANDA_JOINT_LIMITS)


def test_batch_fk_equals_single_fk(chain):
    """批量 FK 是 `chain.fk()` 的重写版（为了快），两者必须逐位相等。

    这是"同一条公式只该有一份实现"的例外 —— 为了性能不得不写两份，
    所以用这条断言把它们钉在一起（与 ETS vs MDH 那套交叉验证同一思路）。
    """
    rng = np.random.default_rng(0)
    qs = rng.uniform(-np.pi, np.pi, size=(500, 7))
    batch = fk_positions_batch(chain, qs)
    single = np.array([chain.fk(q).t for q in qs])
    assert np.array_equal(batch, single)


def test_batch_fk_on_two_link_chain():
    chain = planar_chain(l1=1.0, l2=1.0)
    rng = np.random.default_rng(1)
    qs = rng.uniform(-np.pi, np.pi, size=(200, 2))
    batch = fk_positions_batch(chain, qs)
    single = np.array([chain.fk(q).t for q in qs])
    assert np.array_equal(batch, single)


def test_batch_fk_rejects_wrong_shape(chain):
    with pytest.raises(ValueError, match="形状"):
        fk_positions_batch(chain, np.zeros((10, 5)))


def test_sampling_is_reproducible(chain):
    """固定 seed：同样的输入必须给同样的结论，否则失败分类就成了掷骰子。"""
    a = sample_joint_space(chain, 100, seed=42)
    b = sample_joint_space(chain, 100, seed=42)
    assert np.array_equal(a, b)
    c = sample_joint_space(chain, 100, seed=43)
    assert not np.array_equal(a, c)


def test_directional_reach_is_reproducible(chain, limits):
    first = directional_reach(chain, [0.0, 0.0, 1.0], samples=500, seed=7, limits=limits)
    second = directional_reach(chain, [0.0, 0.0, 1.0], samples=500, seed=7, limits=limits)
    assert first.radius == second.radius
    assert first.summary() == second.summary()


def test_reach_is_larger_toward_the_max_reach_direction(chain, limits):
    """沿最大可达方向的投影最大 —— 这就是"工作空间不是球"的定量说明。"""
    up = directional_reach(chain, [0.0, 0.0, 1.0], samples=4000, seed=0, limits=limits)
    sideways = directional_reach(chain, [1.0, 0.0, 0.0], samples=4000, seed=0, limits=limits)
    assert up.radius > sideways.radius
    assert sideways.radius < max_reach(chain, samples=4000, seed=0, limits=limits)


def test_direction_is_normalized(chain):
    long_vector = np.array([0.0, 0.0, 123.0])
    estimate = directional_reach(chain, long_vector, samples=200, seed=0)
    assert np.isclose(np.linalg.norm(estimate.direction), 1.0)


def test_zero_direction_is_rejected(chain):
    with pytest.raises(ValueError, match="方向"):
        directional_reach(chain, [0.0, 0.0, 0.0])


def test_margin_sign_tells_inside_outside(chain, limits):
    estimate = directional_reach(chain, [0.0, 0.0, 1.0], samples=2000, seed=0, limits=limits)
    assert estimate.margin(estimate.radius - 0.01) > 0
    assert estimate.margin(estimate.radius + 0.01) < 0


def test_limits_shape_is_validated(chain):
    with pytest.raises(ValueError, match="关节限位"):
        sample_joint_space(chain, 10, limits=np.zeros((3, 2)))


def test_sampling_respects_limits(chain, limits):
    qs = sample_joint_space(chain, 500, seed=0, limits=limits)
    assert (qs >= limits[:, 0]).all()
    assert (qs <= limits[:, 1]).all()
