"""@file test_singularity.py
@brief 奇异点分析：奇异值、条件数、可操作度、最坏方向。
"""

from __future__ import annotations

from itertools import pairwise

import numpy as np
import pytest

from robotkinematics.core.exceptions import KinematicsError
from robotkinematics.core.planar2r import Planar2R
from robotkinematics.core.singularity import (
    amplification,
    analyze,
    describe_worst_direction,
)

DEG = np.deg2rad


def test_identity_jacobian_is_perfectly_conditioned():
    report = analyze(np.eye(3))
    assert np.isclose(report.condition, 1.0)
    assert np.isclose(report.manipulability, 1.0)
    assert not report.is_singular
    assert not report.is_near_singular


def test_rank_deficient_jacobian_is_singular():
    """两列相同 = 两个关节做的是同一件事 → 有一个方向彻底动不了。"""
    J = np.array([[1.0, 1.0], [0.0, 0.0]])
    report = analyze(J)
    assert report.is_singular
    assert np.isclose(report.manipulability, 0.0, atol=1e-12)
    assert np.isinf(report.condition)


def test_manipulability_equals_product_of_singular_values():
    """可操作度 = σ₁σ₂…σᵣ = √det(J·Jᵀ)，两种算法必须一致。"""
    rng = np.random.default_rng(0)
    J = rng.normal(size=(6, 7))
    report = analyze(J)
    assert np.isclose(report.manipulability, float(np.sqrt(np.linalg.det(J @ J.T))))
    assert np.isclose(report.manipulability, float(np.prod(np.linalg.svd(J, compute_uv=False))))


def test_near_singular_flag_uses_the_condition_number():
    J = np.diag([1.0, 1e-3])  # cond = 1000
    report = analyze(J, cond_warn=100.0)
    assert report.is_near_singular
    assert not report.is_singular  # 还没到降秩，但已经很病态


def test_worst_direction_is_the_smallest_singular_direction():
    J = np.diag([5.0, 0.2, 1.0])
    report = analyze(J)
    # 最小奇异值是 0.2，对应 y 轴
    assert np.isclose(report.sigma_min, 0.2)
    assert np.allclose(np.abs(report.worst_direction), [0.0, 1.0, 0.0])


def test_describe_worst_direction_translates_to_plain_words():
    assert describe_worst_direction(np.array([0.0, 1.0])) == "平面方向 90.0°"
    assert "z 轴" in describe_worst_direction(np.array([0.0, 0.0, 1.0, 0.0, 0.0, 0.0]))
    assert "y 轴" in describe_worst_direction(np.array([0.0, 0.0, 0.0, 0.0, 1.0, 0.0]))


def test_amplification_grows_as_the_planar_arm_straightens():
    arm = Planar2R(l1=1.0, l2=1.0)
    gains = [amplification(arm.jacobian(DEG(0), DEG(q))) for q in (90.0, 30.0, 10.0, 1.0)]
    assert all(a < b for a, b in pairwise(gains))
    assert np.isinf(amplification(arm.jacobian(DEG(0), DEG(0))))


def test_planar_analysis_matches_the_closed_form_condition_number():
    """二连杆的 cond(J) 有闭式值（课件表格里的 2.62 / 9.36 / 28.58），与 SVD 结果一致。"""
    arm = Planar2R(l1=1.0, l2=1.0)
    for q2_deg, expected in [(90.0, 2.62), (30.0, 9.36), (10.0, 28.58)]:
        report = analyze(arm.jacobian(DEG(0), DEG(q2_deg)))
        assert np.isclose(report.condition, expected, atol=5e-3)
        assert np.isclose(report.condition, arm.condition_number(DEG(0), DEG(q2_deg)))


def test_analyze_rejects_non_matrix_input():
    with pytest.raises(KinematicsError):
        analyze(np.zeros(6))
