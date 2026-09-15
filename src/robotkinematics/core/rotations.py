"""@file rotations.py
@brief SO(3)：旋转矩阵、RPY、轴角、四元数、指数/对数映射。

【约定】全部沿用课件，不再自创第二套：
  * 六元组转旋转矩阵：R = Rz(yaw) @ Ry(pitch) @ Rx(roll)
    读作“先绕 x 转 roll，再绕 y 转 pitch，最后绕 z 转 yaw”（固定轴外旋，x→y→z）。
    课件 01_se3_basic.py 里 rpy_to_R 就是这么拼的，ROS 的 RPY 也是这个顺序。
  * 四元数一律 (x, y, z, w)（scalar-last），与 Eigen / ROS 一致，不是 (w, x, y, z)。
  * 所有角度都是弧度。本模块不替调用方做 deg/rad 转换 —— 忘记 np.deg2rad 是这套课
    最高频的错误，与其在这里猜，不如让调用方显式写出来。
"""

from __future__ import annotations

import numpy as np

from .exceptions import InvalidRotationError

# 判定“等于 0 / 等于 π”的阈值。浮点三角函数给不出干净的 0，sin(π) 是 1.2e-16。
_ANGLE_EPS = 1e-9
# 判定万向锁（pitch = ±90°，cos(pitch)=0）的阈值。
_GIMBAL_EPS = 1e-9
# 判定“sin(转角) 小到反对称部分不可用”（即转角接近 π）的阈值。
_SIN_EPS = 1e-8


def rotx(theta: float) -> np.ndarray:
    """绕 x 轴转 theta（弧度）的旋转矩阵。"""
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])


def roty(theta: float) -> np.ndarray:
    """绕 y 轴转 theta（弧度）的旋转矩阵。"""
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


def rotz(theta: float) -> np.ndarray:
    """绕 z 轴转 theta（弧度）的旋转矩阵。"""
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def rpy_to_matrix(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """六元组里的姿态部分（弧度）转 3x3 旋转矩阵：Rz(yaw)·Ry(pitch)·Rx(roll)。"""
    return rotz(yaw) @ roty(pitch) @ rotx(roll)


def matrix_to_rpy(R: np.ndarray) -> tuple[float, float, float]:
    """rpy_to_matrix 的逆。返回 (roll, pitch, yaw)，均为弧度。

    万向锁（pitch = ±90°）时 roll 与 yaw 只能确定它们的组合，这里约定 roll = 0，
    把全部角度记在 yaw 上 —— 这是 ROS / Eigen 的通用做法，目的是让结果可复现，
    而不是假装此时还能解出唯一解。
    """
    _check_shape(R)
    sin_pitch = float(np.clip(-R[2, 0], -1.0, 1.0))
    pitch = float(np.arcsin(sin_pitch))

    if abs(sin_pitch) < 1.0 - _GIMBAL_EPS:
        roll = float(np.arctan2(R[2, 1], R[2, 2]))
        yaw = float(np.arctan2(R[1, 0], R[0, 0]))
    else:
        # cos(pitch)=0：此时 R[0,1] = sin(roll-yaw)、R[1,1] = cos(roll-yaw)
        roll = 0.0
        yaw = float(np.arctan2(-R[0, 1], R[1, 1]))
    return roll, pitch, yaw


def is_rotation(R: np.ndarray, tol: float = 1e-9) -> bool:
    """判断 R 是否为合法旋转矩阵：正交（RᵀR = I）且 det = +1。

    det = -1 的情况（正交但含镜像）必须判为非法 —— 机器人关节给不出镜像变换，
    放过去会让后面的 FK/IK 静默出错。
    """
    R = np.asarray(R, dtype=float)
    if R.shape != (3, 3) or not np.all(np.isfinite(R)):
        return False
    if not np.allclose(R.T @ R, np.eye(3), atol=tol):
        return False
    return bool(abs(np.linalg.det(R) - 1.0) < tol)


def axang_to_matrix(axis: np.ndarray, angle: float) -> np.ndarray:
    """轴角转旋转矩阵（罗德里格斯公式）。"""
    axis = np.asarray(axis, dtype=float).reshape(3)
    norm = float(np.linalg.norm(axis))
    if norm < _ANGLE_EPS:
        if abs(angle) > _ANGLE_EPS:
            raise InvalidRotationError("轴向量为零但转角非零，无法确定旋转轴")
        return np.eye(3)  # 转角为 0，任意轴都表示不旋转
    k = axis / norm
    K = np.array([[0.0, -k[2], k[1]], [k[2], 0.0, -k[0]], [-k[1], k[0], 0.0]])
    return np.eye(3) + np.sin(angle) * K + (1.0 - np.cos(angle)) * (K @ K)


def matrix_to_axang(R: np.ndarray) -> tuple[np.ndarray, float]:
    """axang_to_matrix 的逆。返回 (单位轴, 转角)，转角取值 [0, π]。

    ⚠️ 这里刻意不用 angle = arccos((tr-1)/2)。arccos 在自变量接近 1（小转角）时
    条件数无限大：转角 1e-7 rad 的话，迹上 1e-15 的舍入误差就能让算出的角度偏掉
    百分之几。改成同时用反对称部（sin）和迹（cos）做 atan2，全角度范围都稳定 ——
    IK 迭代里每一步都要求“当前位姿与目标的偏差”，这个偏差恰恰是小角度，
    用 arccos 版本会让收敛精度在末端姿态接近 π 时悄悄劣化。
    """
    _check_shape(R)
    # 反对称部分的向量形式：它的模长是 2·sin(angle)
    v = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    sin_angle = float(np.linalg.norm(v)) / 2.0
    cos_angle = float(np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0))
    angle = float(np.arctan2(sin_angle, cos_angle))

    if angle < _ANGLE_EPS:
        return np.array([1.0, 0.0, 0.0]), 0.0

    if sin_angle > _SIN_EPS:
        axis = v / (2.0 * sin_angle)
        return axis / np.linalg.norm(axis), angle

    # sin≈0 且 angle≈π：反对称部分退化，改用 R = 2aaᵀ - I。
    # 于是 A = (R+I)/2 = aaᵀ，它的第 k 列就是 a·a_k；取 diag 最大的一列最稳。
    A = (R + np.eye(3)) / 2.0
    k = int(np.argmax(np.diag(A)))
    axis = A[:, k] / np.sqrt(max(A[k, k], _ANGLE_EPS))
    return axis / np.linalg.norm(axis), angle


def quat_to_matrix(quat: np.ndarray) -> np.ndarray:
    """四元数 (x, y, z, w) 转旋转矩阵。内部先归一化，避免累积误差放大。"""
    q = np.asarray(quat, dtype=float).reshape(4)
    norm = float(np.linalg.norm(q))
    if norm < _ANGLE_EPS:
        raise InvalidRotationError("四元数范数为零，无法确定旋转")
    x, y, z, w = q / norm
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


def matrix_to_quat(R: np.ndarray) -> np.ndarray:
    """matrix_to_quat 的逆。返回 (x, y, z, w)，并约定 w ≥ 0。

    q 与 -q 表示同一个旋转，不约定符号的话测试会随机失败。
    按迹的正负分四支，是为了避开 trace 接近 0 时的开方放大误差。
    """
    _check_shape(R)
    trace = float(np.trace(R))
    if trace > 0.0:
        s = np.sqrt(trace + 1.0) * 2.0
        w = 0.25 * s
        x = (R[2, 1] - R[1, 2]) / s
        y = (R[0, 2] - R[2, 0]) / s
        z = (R[1, 0] - R[0, 1]) / s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s
    q = np.array([x, y, z, w])
    q /= np.linalg.norm(q)
    return -q if q[3] < 0.0 else q


def rot_log(R: np.ndarray) -> np.ndarray:
    """SO(3) → so(3)：返回旋转向量（轴 × 转角），长度为转角。"""
    axis, angle = matrix_to_axang(R)
    return axis * angle


def rot_exp(w: np.ndarray) -> np.ndarray:
    """so(3) → SO(3)：旋转向量转旋转矩阵。"""
    w = np.asarray(w, dtype=float).reshape(3)
    angle = float(np.linalg.norm(w))
    if angle < _ANGLE_EPS:
        return np.eye(3)
    return axang_to_matrix(w / angle, angle)


def _check_shape(R: np.ndarray) -> None:
    if np.asarray(R).shape != (3, 3):
        raise InvalidRotationError(f"旋转矩阵必须是 3x3，收到 {np.asarray(R).shape}")
