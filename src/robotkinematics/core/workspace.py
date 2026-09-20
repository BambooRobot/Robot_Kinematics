"""@file workspace.py

@brief 可达性估计：沿某个方向"最远能到哪里" —— 库没有这个 API，靠采样。

【为什么库给不了】robotics toolbox 的 `robot.reach` 实测对本项目的模型返回 0，
它也不提供"沿某方向的可达边界"。而这道工序很有用：
IK 迭代失败时，你只知道"没解出来"，不知道是"根本够不着"还是"初值不好" ——
先做一次可达性预筛就能把这两种情况分开，而处置方式完全不同。

【为什么是采样，不是公式】7 自由度的可达集没有解析边界。采样是唯一现实的做法，
所以结论是一个**估计值**：把采样数、随机种子、是否计入关节限位一起返回，
可复现、也可被质疑。

⚠️ 采样本身用库的批量 `fkine`（内部向量化）—— 8000 个姿态是毫秒级。
   本项目只自己写"沿方向取最大投影"这一行判断。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import roboticstoolbox as rtb

from .robots import end_positions


@dataclass(frozen=True)
class ReachEstimate:
    """沿某个方向的可达边界估计。"""

    robot_name: str
    direction: np.ndarray  # 单位向量
    radius: float  # 沿该方向投影的最大值 [m]
    samples: int
    seed: int
    within_limits: bool  # 采样时是否计入了关节限位

    def margin(self, distance: float) -> float:
        """目标距离与边界的差：> 0 表示在界内。"""
        return float(self.radius - distance)

    def summary(self) -> str:
        """把估计值连同"怎么估出来的"一起写出来：采样数、种子、是否计限位。"""
        scope = "计关节限位" if self.within_limits else "未计关节限位"
        return (
            f"沿该方向可达 {self.radius:.4f} m"
            f"（{self.samples} 次采样估计，seed={self.seed}，{scope}）"
        )


def directional_reach(
    robot: rtb.Robot,
    direction: np.ndarray,
    *,
    samples: int = 8000,
    seed: int = 0,
    limits: np.ndarray | None = None,
) -> ReachEstimate:
    """沿 `direction` 方向采样估计可达边界。"""
    direction = np.asarray(direction, dtype=float).reshape(3)
    norm = float(np.linalg.norm(direction))
    if norm < 1e-12:
        raise ValueError("方向向量不能为零")
    direction = direction / norm

    qs = sample_joint_space(robot, samples, seed=seed, limits=limits)
    projections = end_positions(robot, qs) @ direction
    return ReachEstimate(
        robot_name=robot.name or "robot",
        direction=direction,
        radius=float(projections.max()),
        samples=samples,
        seed=seed,
        within_limits=limits is not None,
    )


def max_reach(
    robot: rtb.Robot, *, samples: int = 8000, seed: int = 0, limits: np.ndarray | None = None
) -> float:
    """采样估计的最大可达距离（所有方向里最大的那一个）。"""
    qs = sample_joint_space(robot, samples, seed=seed, limits=limits)
    return float(np.linalg.norm(end_positions(robot, qs), axis=1).max())


def sample_joint_space(
    robot: rtb.Robot, samples: int, *, seed: int = 0, limits: np.ndarray | None = None
) -> np.ndarray:
    """均匀采样关节空间，返回 (samples, n)。固定 seed，保证结论可复现。

    limits 形如 (n, 2)；None 表示按 ±180° 采样（"不检查限位"的理想情况）。
    """
    rng = np.random.default_rng(seed)
    n = robot.n
    if limits is None:
        return rng.uniform(-np.pi, np.pi, size=(samples, n))
    limits = np.asarray(limits, dtype=float)
    if limits.shape != (n, 2):
        raise ValueError(f"关节限位形状应为 ({n}, 2)，收到 {limits.shape}")
    return rng.uniform(limits[:, 0], limits[:, 1], size=(samples, n))
