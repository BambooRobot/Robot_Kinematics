"""@file test_panda_jacobian.py
@brief Panda 的 6×7 几何雅可比：与数值微分对照，以及它的物理含义。
"""

from __future__ import annotations

import numpy as np
import pytest

from robotkinematics.core import panda
from robotkinematics.core.numerics import numeric_pose_jacobian
from robotkinematics.core.singularity import analyze


@pytest.fixture
def chain():
    return panda.panda_urdf_chain()


def test_jacobian_shape_is_6_by_7(chain):
    """课件结论：真实机械臂的雅可比是 6×n；Panda 的 n=7，所以是 6×7。

    前 3 行对应末端线速度，后 3 行对应角速度。
    """
    J = chain.jacobian(panda.PANDA_Q_IK_SEED)
    assert J.shape == (6, 7)


@pytest.mark.parametrize(
    "q",
    [
        np.array([0.5, -0.9, 1.1, -1.6, 0.4, 1.3, -0.7]),
        np.array([-1.2, 0.3, 0.8, -2.5, 1.1, 2.4, -0.4]),
        np.array([2.1, 1.4, -0.6, -1.0, -1.7, 0.5, 2.6]),
    ],
)
def test_jacobian_matches_numeric_differentiation(chain, q):
    """解析雅可比必须等于数值微分 —— 这一条曾经抓出过 rot_log 的数值 bug。"""
    J = chain.jacobian(q)
    numeric = numeric_pose_jacobian(chain, q)
    assert np.allclose(J, numeric, atol=1e-6)


def test_last_column_is_a_pure_rotation_about_the_last_joint_axis(chain):
    """第 7 个关节只管转：工具坐标系就挂在它的轴上，所以线速度部分恒为 0。

    课件日志里 `jacob0` 最后一列全是 0，是因为那个模型的工具端与法兰重合；
    本项目的模型含 107mm 工具偏移，于是最后一列只有角速度、没有线速度。
    """
    q = panda.PANDA_Q_IK_SEED
    J = chain.jacobian(q)
    assert np.allclose(J[:3, 6], 0.0, atol=1e-12)
    assert np.isclose(np.linalg.norm(J[3:, 6]), 1.0)  # 单位轴向量
    # 这一列就是第 7 关节坐标系（= 法兰）的 z 轴在 base 下的方向
    flange_frame = chain.joint_frames(q)[6]
    z7 = flange_frame[:3, :3] @ np.array([0.0, 0.0, 1.0])
    assert np.allclose(J[3:, 6], z7, atol=1e-12)


def test_zero_pose_is_singular(chain):
    """全零姿态下手臂竖直、腕部对齐，雅可比降秩 —— 这正是课件里那个“姿态 1”。"""
    report = analyze(chain.jacobian(panda.PANDA_Q_ZERO))
    assert report.is_singular
    assert report.manipulability < 1e-12


def test_a_normal_working_pose_is_not_singular(chain):
    report = analyze(chain.jacobian(panda.PANDA_Q_PPT_GOAL))
    assert not report.is_singular
    assert report.condition < 100.0


def test_redundancy_shows_up_as_a_nonzero_null_space(chain):
    """7 个关节、6 维任务 → 零空间至少 1 维：存在“不动末端也能动关节”的方向。"""
    q = panda.PANDA_Q_IK_SEED
    J = chain.jacobian(q)
    _, _, Vt = np.linalg.svd(J, full_matrices=True)
    null_space = Vt[6:, :]  # 最小那一端剩下的方向
    assert null_space.shape[0] == 1
    for direction in null_space:
        # 沿零空间方向动 1e-6，末端位姿几乎不变
        q2 = q + 1e-6 * direction
        assert np.allclose(chain.fk(q2).t, chain.fk(q).t, atol=1e-9)


def test_jacobian_linear_rows_scale_with_the_distance_to_the_joint_axis(chain):
    """越远的关节，同样的角速度产生的末端线速度越大 —— 这就是 z×(p_ee - p_i) 的含义。"""
    q = panda.PANDA_Q_PPT_GOAL
    J = chain.jacobian(q)
    p_ee = chain.fk(q).t
    p0 = np.zeros(3)
    # 第 1 关节在 base 原点，其线速度贡献的模长必须等于末端到 z 轴的距离
    radius = np.linalg.norm(np.cross(np.array([0.0, 0.0, 1.0]), p_ee - p0))
    assert np.isclose(np.linalg.norm(J[:3, 0]), radius, atol=1e-12)
