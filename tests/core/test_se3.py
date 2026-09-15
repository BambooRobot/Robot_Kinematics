"""@file test_se3.py
@brief SE3 与齐次变换矩阵，包含课件 01/02 两个算例的精确数值。
"""

from __future__ import annotations

import numpy as np
import pytest

from robotkinematics.core.exceptions import InvalidRotationError
from robotkinematics.core.se3 import SE3

DEG = np.deg2rad
# 课程 outputs/ppt_cases_batch_result.txt 与 run_logs 记录的数值
BASE_CAMERA = (0.30, 0.00, 0.60)
CUP_CAMERA = (0.50, -0.10, 0.20)


def test_identity_and_translation():
    T = SE3.Trans(0.4, 0.2, 0.3)
    assert np.allclose(T.t, [0.4, 0.2, 0.3])
    assert np.allclose(T.R, np.eye(3))
    # 纯平移只回答“在哪里”，点变换就是平移相加
    assert np.allclose(T.apply([1.0, 1.0, 1.0]), [1.4, 1.2, 1.3])


def test_rotation_only_does_not_move_the_origin():
    T = SE3.Rz(DEG(90))
    assert np.allclose(T.t, np.zeros(3))
    assert np.allclose(T.apply([1.0, 0.0, 0.0]), [0.0, 1.0, 0.0], atol=1e-12)


def test_point_transform_matches_ppt_case():
    """课件 01_se3_basic.py：0.4/0.2/0.3 + Rz(90°)，点 (0.1,0,0) → (0.4,0.3,0.3)。"""
    T = SE3.from_pose6(0.4, 0.2, 0.3, 0.0, 0.0, DEG(90))
    assert np.allclose(T.apply([0.1, 0.0, 0.0]), [0.4, 0.3, 0.3], atol=1e-12)


def test_camera_to_base_chain_matches_course_log():
    """课件 02 的变换链：^base T_cup = ^base T_camera · ^camera T_cup。"""
    T_base_camera = SE3.Trans(*BASE_CAMERA) * SE3.Rz(DEG(30))
    T_camera_cup = SE3.Trans(*CUP_CAMERA)
    T_base_cup = T_base_camera * T_camera_cup
    assert np.allclose(T_base_cup.t, [0.7830, 0.1634, 0.8], atol=5e-5)


def test_multiplication_order_matters():
    T_pos = SE3.Trans(0.4, 0.2, 0.3)
    T_rot = SE3.Rz(DEG(90))
    assert not (T_pos * T_rot).is_close(T_rot * T_pos, tol=1e-6)


def test_inverse_undoes_the_transform():
    T = SE3.Trans(0.30, 0.0, 0.60) * SE3.Rz(DEG(30)) * SE3.Ry(DEG(-15))
    point = np.array([0.5, -0.1, 0.2])
    assert np.allclose(T.inverse().apply(T.apply(point)), point, atol=1e-12)
    assert (T * T.inverse()).is_close(SE3.identity(), tol=1e-12)


def test_inverse_is_not_just_negated_translation():
    """带旋转的变换，逆不等于“平移取负” —— 这是最常见的错法之一。"""
    T = SE3.Trans(0.30, 0.0, 0.60) * SE3.Rz(DEG(30))
    naive = SE3(R=T.R, t=-T.t)
    assert not T.is_close(naive, tol=1e-6)
    assert (T.inverse() * T).is_close(SE3.identity(), tol=1e-12)


def test_A_round_trip_and_layout():
    T = SE3.Trans(0.1, 0.2, 0.3) * SE3.Rz(DEG(45))
    A = T.A
    assert A.shape == (4, 4)
    assert np.allclose(A[3, :], [0, 0, 0, 1])
    assert SE3.from_matrix(A).is_close(T)
    assert np.allclose(SE3.from_matrix(np.eye(4)).A, np.eye(4))


def test_pose6_round_trip():
    pose = (0.4, 0.2, 0.3, DEG(10), DEG(20), DEG(30))
    back = SE3.from_pose6(*pose).pose6()
    assert np.allclose(back, pose, atol=1e-12)


def test_apply_direction_ignores_translation():
    T = SE3.Trans(0.4, 0.2, 0.3) * SE3.Rz(DEG(90))
    assert np.allclose(T.apply_direction([1.0, 0.0, 0.0]), [0.0, 1.0, 0.0], atol=1e-12)


def test_constructor_rejects_invalid_rotation():
    with pytest.raises(InvalidRotationError):
        SE3(R=np.zeros((3, 3)))
    with pytest.raises(InvalidRotationError):
        SE3(R=np.diag([1.0, 1.0, -1.0]))  # det = -1，是镜像不是旋转


def test_from_matrix_rejects_non_homogeneous():
    bad = np.eye(4)
    bad[3, :] = [0, 0, 0, 2]
    with pytest.raises(InvalidRotationError):
        SE3.from_matrix(bad)
    with pytest.raises(InvalidRotationError):
        SE3.from_matrix(np.eye(3))


def test_multiplying_with_non_se3_is_an_error_not_a_silent_bug():
    with pytest.raises(TypeError):
        SE3.Trans(0.1, 0, 0) * np.eye(4)  # type: ignore[operator]


def test_properties_return_copies():
    """外部改 .R/.t 不该影响对象内部状态。"""
    T = SE3.Trans(0.1, 0.2, 0.3)
    T.t[0] = 99.0
    T.R[0, 0] = 99.0
    assert np.allclose(T.t, [0.1, 0.2, 0.3])
    assert np.allclose(T.R, np.eye(3))
