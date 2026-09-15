"""@file numerics.py
@brief 数值工具：用差分验证解析结果。

【为什么需要它】雅可比是 FK 的偏导，写错一个符号在纸面上很难看出来，
但拿“数值微分”当尺子一量就露馅：解析雅可比与差分雅可比必须逐项相等。
本模块是自研数学的“尺子”，只服务于验证，不参与正经计算路径。
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from .rotations import rot_log


def numeric_jacobian(
    func: Callable[[np.ndarray], np.ndarray],
    q: np.ndarray,
    eps: float = 1e-7,
) -> np.ndarray:
    """中心差分求 func 在 q 处的雅可比：J[:, i] = ∂func/∂q_i。

    func: q -> 向量（末端位置或位姿展平）。q: (n,) 关节角。
    返回 (m, n)，m 为 func 输出维度。

    用中心差分而不是前向差分：误差是 O(eps²) 而非 O(eps)，用 1e-7 的步长就能跟
    解析解对齐到 1e-9 量级，足以判定实现有没有写错。
    """
    q = np.asarray(q, dtype=float).reshape(-1)
    f0 = np.asarray(func(q), dtype=float).reshape(-1)
    J = np.zeros((f0.size, q.size))
    for i in range(q.size):
        dq = np.zeros_like(q)
        dq[i] = eps
        f_plus = np.asarray(func(q + dq), dtype=float).reshape(-1)
        f_minus = np.asarray(func(q - dq), dtype=float).reshape(-1)
        J[:, i] = (f_plus - f_minus) / (2.0 * eps)
    return J


def numeric_pose_jacobian(robot, q: np.ndarray, eps: float = 1e-7) -> np.ndarray:
    """用差分求“末端位姿对关节角”的 6×n 雅可比，用来验证解析几何雅可比。

    ⚠️ 不能直接把 rot_log(R(q)) 拿去差分：rot_log 在转角接近 π 处有分支跳变
    （轴会突然反向），中心差分一旦跨过分支就会给出完全错误的值。正确做法是
    始终相对同一个参考姿态取误差 —— R(q±δ)·R(q)ᵀ 永远是一个小旋转，
    rot_log 落在安全区间里，而且它的导数恰好就是几何雅可比的角速度部分。
    """
    q = np.asarray(q, dtype=float).reshape(-1)
    T0 = robot.fk(q)
    t0 = T0.t
    R0 = T0.R

    def error(qq: np.ndarray) -> np.ndarray:
        T = robot.fk(qq)
        return np.concatenate([T.t - t0, rot_log(T.R @ R0.T)])

    return numeric_jacobian(error, q, eps)
