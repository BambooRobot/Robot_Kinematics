"""@file singularity.py
@brief 奇异位姿判定：奇异值、条件数、可操作度、最难运动的方向。

【为什么用 SVD 而不是 det】det 只对 2x2 这种方阵雅可比有意义。真实机械臂的雅可比是
6×n（Panda 是 6×7），根本不存在行列式。SVD 对任意形状都成立，而且给的信息更多：
奇异值 → 各方向上的运动放大倍数；最小奇异值 → 最坏方向；左奇异向量 → 那个方向是什么。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .exceptions import KinematicsError


@dataclass(frozen=True)
class SingularityReport:
    """一次位姿的“局部运动能力”体检报告。"""

    sigma_max: float
    sigma_min: float
    condition: float
    manipulability: float
    worst_direction: np.ndarray
    is_singular: bool
    is_near_singular: bool

    def summary(self) -> str:
        state = "奇异" if self.is_singular else ("接近奇异" if self.is_near_singular else "正常")
        return (
            f"σ_max={self.sigma_max:.4e} σ_min={self.sigma_min:.4e} "
            f"cond={self.condition:.2f} 可操作度={self.manipulability:.4e} → {state}"
        )


def analyze(J: np.ndarray, det_eps: float = 1e-9, cond_warn: float = 100.0) -> SingularityReport:
    """对雅可比做 SVD，判断当前位姿离奇异有多远。

    可操作度（Yoshikawa）w = σ₁σ₂…σ_r = √det(J·Jᵀ)：
    它是雅可比张成的“超椭球体积”，w = 0 表示至少有一个方向彻底动不了。
    """
    J = np.asarray(J, dtype=float)
    if J.ndim != 2:
        raise KinematicsError(f"雅可比必须是二维矩阵，收到 {J.shape}")

    # 一次 SVD 拿两样东西：奇异值（放大倍数）和左奇异向量（最坏方向）
    U, sigma, _ = np.linalg.svd(J, full_matrices=True)
    sigma_min = float(sigma[-1]) if sigma.size else 0.0
    sigma_max = float(sigma[0]) if sigma.size else 0.0
    condition = float(sigma_max / sigma_min) if sigma_min > 0.0 else float("inf")
    manipulability = float(np.prod(sigma))
    return SingularityReport(
        sigma_max=sigma_max,
        sigma_min=sigma_min,
        condition=condition,
        manipulability=manipulability,
        worst_direction=U[:, -1].copy(),
        is_singular=sigma_min < det_eps,
        is_near_singular=condition > cond_warn,
    )


def describe_worst_direction(vector: np.ndarray, eps: float = 1e-6) -> str:
    """把“最难运动的方向”翻译成一句人话。

    6 维向量前 3 位是线速度、后 3 位是角速度（与几何雅可比的排列一致）；
    2 维向量就是平面里的一个方向。
    """
    v = np.asarray(vector, dtype=float).reshape(-1)
    if v.size == 2:
        angle = float(np.rad2deg(np.arctan2(v[1], v[0])))
        return f"平面方向 {angle:.1f}°"
    if v.size == 6 and np.linalg.norm(v[3:]) < eps:
        axes = "xyz"
        dominant = axes[int(np.argmax(np.abs(v[:3])))]
        return f"纯平移，主要沿 {dominant} 轴"
    if v.size == 6 and np.linalg.norm(v[:3]) < eps:
        axes = "xyz"
        dominant = axes[int(np.argmax(np.abs(v[3:])))]
        return f"纯转动，主要绕 {dominant} 轴"
    return "平移与转动耦合方向"


def amplification(J: np.ndarray) -> float:
    """最坏方向上的关节速度放大倍数 = 1/σ_min：末端要求 1 单位位移时最费关节的方向。"""
    sigma = np.linalg.svd(np.asarray(J, dtype=float), compute_uv=False)
    if sigma[-1] <= 0.0:
        return float("inf")
    return float(1.0 / sigma[-1])
