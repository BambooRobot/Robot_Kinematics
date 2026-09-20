"""@file singularity.py

@brief 奇异位姿体检 —— 一层薄封装：SVD 用 numpy，可操作度用库的现成实现。

【为什么还留着这一层】因为"读出结论"这件事库不包：
  `robot.manipulability()` 给你一个数（4 种定义可选），但不会告诉你
  "这算不算接近奇异""最难动的方向是哪个""该不该拒绝这组解"。
  这里把那几个数收敛成一张体检报告，供抓取流水线的第 ④ 道工序使用。

❌ 已删除的自研部分：解析几何雅可比、数值微分、det/cond 的手写实现 ——
   雅可比交给 `robot.jacob0(q)`，SVD 交给 `numpy.linalg.svd`。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import roboticstoolbox as rtb

from .exceptions import KinematicsError


@dataclass(frozen=True)
class SingularityReport:
    """一次位姿的"局部运动能力"体检报告。"""

    sigma_max: float
    sigma_min: float
    condition: float
    manipulability: float
    worst_direction: np.ndarray
    is_singular: bool
    is_near_singular: bool

    def summary(self) -> str:
        """把体检结论压成一行：四个关键数值 + 正常 / 接近奇异 / 奇异。"""
        state = "奇异" if self.is_singular else ("接近奇异" if self.is_near_singular else "正常")
        return (
            f"σ_max={self.sigma_max:.4e} σ_min={self.sigma_min:.4e} "
            f"cond={self.condition:.2f} 可操作度={self.manipulability:.4e} → {state}"
        )


def analyze(J: np.ndarray, det_eps: float = 1e-9, cond_warn: float = 100.0) -> SingularityReport:
    """对雅可比做 SVD，判断当前位姿离奇异有多远。

    用 SVD 而不是 det：真实机械臂的雅可比是 6×n（Panda 是 6×7），根本不存在行列式。
    """
    J = np.asarray(J, dtype=float)
    if J.ndim != 2:
        raise KinematicsError(f"雅可比必须是二维矩阵，收到 {J.shape}")

    U, sigma, _ = np.linalg.svd(J, full_matrices=True)
    sigma_min = float(sigma[-1]) if sigma.size else 0.0
    sigma_max = float(sigma[0]) if sigma.size else 0.0
    condition = float(sigma_max / sigma_min) if sigma_min > 0.0 else float("inf")
    return SingularityReport(
        sigma_max=sigma_max,
        sigma_min=sigma_min,
        condition=condition,
        manipulability=float(np.prod(sigma)),  # 与库的 yoshikawa 定义一致
        worst_direction=U[:, -1].copy(),
        is_singular=sigma_min < det_eps,
        is_near_singular=condition > cond_warn,
    )


def manipulability(robot: rtb.Robot, q: np.ndarray, method: str = "yoshikawa") -> float:
    """可操作度：直接调库（`robot.manipulability` 支持 yoshikawa / asada / minsingular / invcondition）。

    与 `analyze(J).manipulability` 的 yoshikawa 定义是同一个量，两边可互相校验。
    """
    return float(robot.manipulability(np.asarray(q, dtype=float).reshape(-1), method=method))


def analysis_of(robot: rtb.Robot, q: np.ndarray, **kwargs) -> SingularityReport:
    """对某个位形做体检：雅可比交给库，分析交给 analyze。"""
    return analyze(robot.jacob0(np.asarray(q, dtype=float).reshape(-1)), **kwargs)


def describe_worst_direction(vector: np.ndarray, eps: float = 1e-6) -> str:
    """把"最难运动的方向"翻译成一句人话（6 维：前 3 线速度、后 3 角速度）。"""
    v = np.asarray(vector, dtype=float).reshape(-1)
    if v.size == 2:
        return f"平面方向 {float(np.rad2deg(np.arctan2(v[1], v[0]))):.1f}°"
    axes = "xyz"
    if v.size == 6 and np.linalg.norm(v[3:]) < eps:
        return f"纯平移，主要沿 {axes[int(np.argmax(np.abs(v[:3])))]} 轴"
    if v.size == 6 and np.linalg.norm(v[:3]) < eps:
        return f"纯转动，主要绕 {axes[int(np.argmax(np.abs(v[3:])))]} 轴"
    return "平移与转动耦合方向"


def amplification(report: SingularityReport) -> float:
    """最坏方向上的关节速度放大倍数 = 1/σ_min。"""
    return float("inf") if report.sigma_min <= 0.0 else float(1.0 / report.sigma_min)
