"""@file planar2r.py
@brief 平面二连杆：最简单的机械臂，用来看清 FK / IK / 雅可比的本质。

   base ──q1──> link1(L1) ──q2──> link2(L2) ──> tool
   全部在 x-y 平面内，q1、q2 为关节角（弧度，逆时针为正）。

【为什么先讲二连杆】它的公式能手推、能手算验证，且奇异点、多解、无解这些在
六轴机械臂上会把人绕晕的现象，在这里都能一眼看明白。Panda 只是同一个问题
换成了 7 个关节、3 维位置 + 3 维姿态。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .exceptions import KinematicsError, SingularPoseError

# cos(q2) 的可行域判定容差：目标点恰好在可达边界上时，浮点会让 cos 略微越界。
_BOUNDARY_EPS = 1e-12


@dataclass(frozen=True)
class Planar2R:
    """平面二连杆模型。l1、l2 是固定几何结构，q1、q2 才是关节变量。"""

    l1: float = 1.0
    l2: float = 1.0

    def __post_init__(self) -> None:
        if self.l1 <= 0.0 or self.l2 <= 0.0:
            raise KinematicsError(f"连杆长度必须为正，收到 l1={self.l1}, l2={self.l2}")

    # ── 正运动学：关节角 -> 末端位置 ────────────────────────────────────────

    def fk(self, q1: float, q2: float) -> np.ndarray:
        """末端位置。两段向量相加：第一段方向由 q1 定，第二段由 q1+q2 定。"""
        x = self.l1 * np.cos(q1) + self.l2 * np.cos(q1 + q2)
        y = self.l1 * np.sin(q1) + self.l2 * np.sin(q1 + q2)
        return np.array([x, y])

    def joint_points(self, q1: float, q2: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """返回 (base, elbow, tool) 三个点，供画图和 ASCII 示意使用。"""
        base = np.zeros(2)
        elbow = np.array([self.l1 * np.cos(q1), self.l1 * np.sin(q1)])
        tool = elbow + np.array([self.l2 * np.cos(q1 + q2), self.l2 * np.sin(q1 + q2)])
        return base, elbow, tool

    # ── 逆运动学：末端位置 -> 关节角 ────────────────────────────────────────

    def ik(self, x: float, y: float) -> list[tuple[float, float]]:
        """求所有可行解，按肘上、肘下的顺序返回。

        返回空列表表示目标超出工作空间 —— 那是物理限制，不是程序错误，
        所以这里不抛异常，由调用方决定怎么呈现（CLI 会把它翻译成清晰的报错）。
        """
        r2 = x * x + y * y
        cos_q2 = (r2 - self.l1**2 - self.l2**2) / (2.0 * self.l1 * self.l2)
        if cos_q2 < -1.0 - _BOUNDARY_EPS or cos_q2 > 1.0 + _BOUNDARY_EPS:
            return []
        cos_q2 = float(np.clip(cos_q2, -1.0, 1.0))

        solutions: list[tuple[float, float]] = []
        for q2 in (float(np.arccos(cos_q2)), -float(np.arccos(cos_q2))):
            # 目标方向角减去第二段相对第一段的夹角，就是第一段该指的方向
            q1 = np.arctan2(y, x) - np.arctan2(self.l2 * np.sin(q2), self.l1 + self.l2 * np.cos(q2))
            solutions.append((float(q1), q2))
        return solutions

    def reachable(self, x: float, y: float) -> bool:
        return len(self.ik(x, y)) > 0

    # ── 雅可比：末端速度与关节速度的局部线性关系 ────────────────────────────

    def jacobian(self, q1: float, q2: float) -> np.ndarray:
        """2x2 雅可比 J：Δx ≈ J·Δq。由 fk 对 q1、q2 求偏导而来。"""
        s1 = np.sin(q1)
        c1 = np.cos(q1)
        s12 = np.sin(q1 + q2)
        c12 = np.cos(q1 + q2)
        return np.array(
            [
                [-self.l1 * s1 - self.l2 * s12, -self.l2 * s12],
                [self.l1 * c1 + self.l2 * c12, self.l2 * c12],
            ]
        )

    def det_jacobian(self, q2: float) -> float:
        """det(J) = L1·L2·sin(q2)：与 q1 无关，只由两杆的相对夹角决定。

        q2 = 0° 或 180° 时两杆共线，det(J) = 0 —— 这就是奇异姿态。
        """
        return self.l1 * self.l2 * float(np.sin(q2))

    def condition_number(self, q1: float, q2: float) -> float:
        """cond(J) = 最大奇异值 / 最小奇异值。越大越接近奇异。"""
        return float(np.linalg.cond(self.jacobian(q1, q2)))

    def is_singular(self, q2: float, det_eps: float = 1e-9) -> bool:
        return abs(self.det_jacobian(q2)) < det_eps

    def solve_step(self, q1: float, q2: float, dx: np.ndarray, det_eps: float = 1e-9) -> np.ndarray:
        """已知末端想微调 dx，求关节微调量 dq：解 J·dq = dx。

        奇异位姿下 J 不可逆（或病态），解出来的 dq 会大到没有物理意义，
        所以这里主动抛 SingularPoseError，而不是把巨大的数交给上层。
        """
        if self.is_singular(q2, det_eps=det_eps):
            raise SingularPoseError(
                f"当前位姿接近奇异（q2={np.rad2deg(q2):.1f}°，det(J)={self.det_jacobian(q2):.3e}），"
                "末端在该方向上的瞬时运动能力丧失，无法解出可用的关节微调量"
            )
        J = self.jacobian(q1, q2)
        return np.linalg.solve(J, np.asarray(dx, dtype=float).reshape(2))
