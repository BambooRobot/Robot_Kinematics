"""@file test_rotations.py
@brief SO(3) 各种表示之间的往返一致性，以及 PPT 里的具体数值。

这些断言不是“随便写几个数”，每一个都能在课程材料里找到出处 —— 见 docs/NUMBERS.md。
"""

from __future__ import annotations

import numpy as np
import pytest

from robotkinematics.core import rotations
from robotkinematics.core.exceptions import InvalidRotationError

DEG = np.deg2rad


def test_basic_rotations_match_textbook_form():
    assert np.allclose(rotations.rotx(0.0), np.eye(3))
    # 绕 z 转 90°：x 轴转到 y 轴方向上
    assert np.allclose(rotations.rotz(DEG(90)), [[0, -1, 0], [1, 0, 0], [0, 0, 1]], atol=1e-12)
    assert np.allclose(rotations.roty(DEG(90)), [[0, 0, 1], [0, 1, 0], [-1, 0, 0]], atol=1e-12)
    assert np.allclose(rotations.rotx(DEG(90)), [[1, 0, 0], [0, 0, -1], [0, 1, 0]], atol=1e-12)


def test_rpy_convention_is_rz_ry_rx():
    """课件约定 R = Rz(yaw)·Ry(pitch)·Rx(roll)，顺序写反结果就完全不同。"""
    roll, pitch, yaw = DEG(10), DEG(20), DEG(30)
    expected = rotations.rotz(yaw) @ rotations.roty(pitch) @ rotations.rotx(roll)
    assert np.allclose(rotations.rpy_to_matrix(roll, pitch, yaw), expected)
    # 反例：换个顺序就不相等（否则这个测试没在保护任何东西）
    assert not np.allclose(
        expected, rotations.rotx(roll) @ rotations.roty(pitch) @ rotations.rotz(yaw)
    )


@pytest.mark.parametrize(
    "rpy",
    [
        (0.0, 0.0, 0.0),
        (0.0, 0.0, DEG(90)),  # PPT 里的坐标变换算例
        (DEG(10), DEG(20), DEG(30)),
        (DEG(-45), DEG(80), DEG(170)),
        (0.3, -0.7, 2.1),
    ],
)
def test_rpy_round_trip(rpy):
    R = rotations.rpy_to_matrix(*rpy)
    back = np.array(rotations.matrix_to_rpy(R))
    assert np.allclose(back, rpy, atol=1e-12)


def test_rpy_gimbal_lock_puts_all_angle_on_yaw():
    """pitch = ±90° 时 roll 与 yaw 只能确定组合，约定 roll = 0。"""
    for pitch in (DEG(90), DEG(-90)):
        for roll, yaw in [(DEG(30), DEG(50)), (0.0, 0.0), (DEG(-20), DEG(140))]:
            R = rotations.rpy_to_matrix(roll, pitch, yaw)
            r_back, p_back, y_back = rotations.matrix_to_rpy(R)
            assert r_back == 0.0
            assert np.isclose(p_back, pitch, atol=1e-9)
            # 关键不是角度相等，而是重建出来的 R 一致
            assert np.allclose(rotations.rpy_to_matrix(r_back, p_back, y_back), R, atol=1e-9)


@pytest.mark.parametrize("deg", [0.0, 1.0, 45.0, 90.0, 179.0, 180.0, -90.0])
def test_axang_round_trip(deg):
    axis = np.array([1.0, 2.0, -0.5])
    R = rotations.axang_to_matrix(axis, DEG(deg))
    axis_back, angle_back = rotations.matrix_to_axang(R)
    assert np.isclose(angle_back, abs(DEG(deg)), atol=1e-9)
    assert np.allclose(rotations.axang_to_matrix(axis_back, angle_back), R, atol=1e-9)


def test_axang_rejects_zero_axis_with_nonzero_angle():
    with pytest.raises(InvalidRotationError):
        rotations.axang_to_matrix(np.zeros(3), DEG(90))


@pytest.mark.parametrize("deg", [0.0, 30.0, 120.0, 180.0, -120.0])
def test_quaternion_round_trip(deg):
    R = rotations.axang_to_matrix(np.array([0.3, -0.9, 0.2]), DEG(deg))
    q = rotations.matrix_to_quat(R)
    assert np.isclose(np.linalg.norm(q), 1.0)
    assert q[3] >= 0.0  # 约定 w ≥ 0，否则 q 与 -q 会让测试随机失败
    assert np.allclose(rotations.quat_to_matrix(q), R, atol=1e-9)


def test_quaternion_composition_matches_matrix_composition():
    """四元数乘法与矩阵乘法的顺序约定必须是同一套，否则复合姿态会静默错位。"""
    q1 = rotations.matrix_to_quat(rotations.rotx(DEG(30)))
    q2 = rotations.matrix_to_quat(rotations.rotz(DEG(40)))
    assert np.allclose(
        rotations.quat_to_matrix(q1) @ rotations.quat_to_matrix(q2),
        rotations.rotx(DEG(30)) @ rotations.rotz(DEG(40)),
    )


@pytest.mark.parametrize("deg", [0.0, 0.5, 90.0, 179.0, -179.0])
def test_log_exp_round_trip(deg):
    R = rotations.axang_to_matrix(np.array([1.0, 0.0, 0.0]), DEG(deg))
    assert np.allclose(rotations.rot_exp(rotations.rot_log(R)), R, atol=1e-9)


def test_rot_log_length_is_the_angle():
    w = rotations.rot_log(rotations.rotz(DEG(75)))
    assert np.isclose(np.linalg.norm(w), DEG(75), atol=1e-12)
    assert np.allclose(w / np.linalg.norm(w), [0.0, 0.0, 1.0], atol=1e-12)


def test_is_rotation_accepts_valid_and_rejects_invalid():
    assert rotations.is_rotation(rotations.rpy_to_matrix(0.1, 0.2, 0.3))
    # det = -1 的正交矩阵（含镜像）必须判非法：关节给不出镜像变换
    mirror = np.diag([1.0, 1.0, -1.0])
    assert not rotations.is_rotation(mirror)
    assert not rotations.is_rotation(np.zeros((3, 3)))
    assert not rotations.is_rotation(np.eye(4))


def test_matrix_to_rpy_rejects_wrong_shape():
    with pytest.raises(InvalidRotationError):
        rotations.matrix_to_rpy(np.eye(4))
