"""@file planar2r.py
@brief 平面二连杆的解析运动学 —— 本项目**唯一保留的手推公式**。

   base ──q1──> link1(L1) ──q2──> link2(L2) ──> tool

【为什么这一处不用库】因为库给不了：
  * 库的数值 IK（`ikine_LM` / `ikine_GN`）只返回**一组**解，而"IK 多解（肘上/肘下）"
    是这一章的核心知识点之一 —— 必须用闭式解才能同时给出两组；
  * 二连杆还是本项目的"对照组"：它有解析解，可以反过来校验整条抓取流水线。

所以这里保留手推的两条公式（FK 的向量相加、IK 的余弦定理），其余一律交给库。
⚠️ 本文件刻意不 import 任何机器人库 —— 它就是"公式本身"的实体。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# cos(q2) 的可行域判定容差：目标恰在可达边界上时，浮点会让 cos 略微越界
_BOUNDARY_EPS = 1e-12


@dataclass(frozen=True)
class Planar2R:
    """平面二连杆：l1、l2 是固定几何结构，q1、q2 才是关节变量。"""

    l1: float = 1.0
    l2: float = 1.0

    def __post_init__(self) -> None:
        if self.l1 <= 0.0 or self.l2 <= 0.0:
            raise ValueError(f"连杆长度必须为正，收到 l1={self.l1}, l2={self.l2}")

    def fk(self, q1: float, q2: float) -> np.ndarray:
        """末端位置：两段向量相加。第一段方向由 q1 定，第二段由 q1+q2 定。"""
        x = self.l1 * np.cos(q1) + self.l2 * np.cos(q1 + q2)
        y = self.l1 * np.sin(q1) + self.l2 * np.sin(q1 + q2)
        return np.array([x, y])

    def ik(self, x: float, y: float) -> list[tuple[float, float]]:
        """求**所有**可行解，按肘上、肘下的顺序返回；空列表表示目标够不着。

        这就是库给不了的那两行：三角形的余弦定理先解出 q2，± 号对应两组解。
        """
        r2 = x * x + y * y
        cos_q2 = (r2 - self.l1**2 - self.l2**2) / (2.0 * self.l1 * self.l2)
        if cos_q2 < -1.0 - _BOUNDARY_EPS or cos_q2 > 1.0 + _BOUNDARY_EPS:
            return []
        cos_q2 = float(np.clip(cos_q2, -1.0, 1.0))

        solutions: list[tuple[float, float]] = []
        for q2 in (float(np.arccos(cos_q2)), -float(np.arccos(cos_q2))):
            q1 = np.arctan2(y, x) - np.arctan2(self.l2 * np.sin(q2), self.l1 + self.l2 * np.cos(q2))
            solutions.append((float(q1), q2))
        return solutions

    def joint_points(self, q1: float, q2: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """返回 (base, elbow, tool) 三个点，供画图与字符画使用。"""
        base = np.zeros(2)
        elbow = np.array([self.l1 * np.cos(q1), self.l1 * np.sin(q1)])
        tool = elbow + np.array([self.l2 * np.cos(q1 + q2), self.l2 * np.sin(q1 + q2)])
        return base, elbow, tool

    def det_jacobian(self, q2: float) -> float:
        """det(J) = L1·L2·sin(q2)：与 q1 无关，只由两杆的相对夹角决定。

        这是课件里那张奇异点表的依据（q2 = 0° / 180° 时两杆共线、det = 0）。
        """
        return self.l1 * self.l2 * float(np.sin(q2))
