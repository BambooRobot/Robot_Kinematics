"""@file test_planar_ik.py
@brief 二连杆逆运动学：多解、无解、边界与 FK 回验。
"""

from __future__ import annotations

import numpy as np
import pytest

from robotkinematics.core.planar2r import Planar2R

DEG = np.deg2rad


@pytest.fixture
def arm() -> Planar2R:
    return Planar2R(l1=1.0, l2=1.0)


def test_ik_returns_both_elbow_configurations(arm):
    """课件算例：目标 (1,1) 有两组解 —— 肘上 (0°,90°) 与 肘下 (90°,-90°)。"""
    solutions = arm.ik(1.0, 1.0)
    assert len(solutions) == 2
    # 顺序约定：第一组 q2 = +arccos（肘上），第二组 q2 = -arccos（肘下）
    assert np.allclose(np.rad2deg(solutions[0]), [0.0, 90.0], atol=1e-9)
    assert np.allclose(np.rad2deg(solutions[1]), [90.0, -90.0], atol=1e-9)


def test_every_ik_solution_passes_fk_verification(arm):
    """IK 求出 q 后必须送回 FK 检查末端是否回到目标点 —— 不要直接相信结果。"""
    for target in [(1.0, 1.0), (0.5, 0.8), (-1.2, 0.3), (0.0, 1.5), (1.9, 0.0)]:
        solutions = arm.ik(*target)
        assert solutions, f"目标 {target} 应当可达"
        for q1, q2 in solutions:
            assert np.allclose(arm.fk(q1, q2), target, atol=1e-9)


def test_ik_reports_no_solution_when_target_is_too_far(arm):
    """课件里的无解例子：目标 (3,0) 超出工作空间。"""
    assert arm.ik(3.0, 0.0) == []


def test_ik_reports_no_solution_inside_the_inner_hole():
    """L1=L2 时原点是内部的“洞”：两杆折叠才能到，而原点处方向不定。"""
    arm = Planar2R(l1=1.0, l2=0.3)
    assert arm.ik(0.0, 0.0) == []  # 内半径 |L1-L2| = 0.7 > 0
    assert arm.ik(0.69, 0.0) == []


def test_on_the_outer_boundary_the_two_solutions_degenerate(arm):
    """目标正好在可达边界（L1+L2）上：两组解退化成一组完全伸直的姿态。"""
    solutions = arm.ik(2.0, 0.0)
    assert len(solutions) == 2
    assert np.allclose(solutions[0], solutions[1], atol=1e-9)
    assert np.allclose(np.rad2deg(solutions[0]), [0.0, 0.0], atol=1e-9)


@pytest.mark.parametrize("seed", range(5))
def test_ik_fk_round_trip_on_random_reachable_points(arm, seed):
    rng = np.random.default_rng(seed)
    for _ in range(50):
        q1, q2 = rng.uniform(-np.pi, np.pi, size=2)
        target = arm.fk(q1, q2)
        if np.linalg.norm(target) < 1e-6:
            continue  # 原点方向不定，跳过
        solutions = arm.ik(*target)
        assert any(np.allclose(arm.fk(*s), target, atol=1e-9) for s in solutions)


def test_reachable_flag_agrees_with_ik(arm):
    assert arm.reachable(1.0, 1.0)
    assert not arm.reachable(3.0, 0.0)
