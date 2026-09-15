"""@file workspace.py
@brief 工作空间的采样估计：沿某个方向"最远能到哪里"。

【为什么需要它】IK 迭代失败时，你只知道"没解出来"，不知道是"根本够不着"还是"初值不好"。
先做一次工作空间预筛就能把这两种情况分开 —— 而处置方式完全不同：
前者要挪目标，后者换个初值可能就成了。

【为什么是采样，而不是公式】7 自由度的可达集没有解析边界。采样是唯一现实的做法。
代价是"边界"只是一个估计值，所以把采样数、随机种子、是否计入关节限位一起返回：
结论可复现，也可被质疑。⚠️ 也正因如此，工作空间**不是球** ——
"最大可达半径 1.19 m" 只在某个方向成立，换个方向可能只有 1.09 m。

【为什么自己写一遍批量 FK】逐次调用 `chain.fk()` 采 8000 个姿态要 1.5 秒，
放在命令行的预筛里太慢。这里对 N 个姿态同时做一遍串联链乘法（numpy 批量矩阵乘），
快两个数量级。代价是多了一份 FK 实现 —— 所以 tests/test_workspace.py 里
有一条断言：批量 FK 与 `chain.fk()` 逐位相等（同"ETS vs MDH"那套交叉验证的思路）。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .chain import SerialChain

# 无关节限位信息时的采样范围：±180°
DEFAULT_LIMITS = (-np.pi, np.pi)


@dataclass(frozen=True)
class ReachEstimate:
    """沿某个方向的可达边界估计。"""

    chain_name: str
    direction: np.ndarray  # 单位向量
    radius: float  # 沿该方向投影的最大值 [m]
    samples: int
    seed: int
    within_limits: bool  # 采样时是否计入了关节限位

    def margin(self, distance: float) -> float:
        """目标距离与边界的差：> 0 表示在界内。"""
        return float(self.radius - distance)

    def summary(self) -> str:
        scope = "计关节限位" if self.within_limits else "未计关节限位"
        return (
            f"沿该方向可达 {self.radius:.4f} m"
            f"（{self.samples} 次采样估计，seed={self.seed}，{scope}）"
        )


def fk_positions_batch(chain: SerialChain, qs: np.ndarray) -> np.ndarray:
    """批量正运动学：一次算 N 个姿态的末端位置。返回 (N, 3)。

    ⚠️ 这是 `SerialChain.fk()` 的批量版本，数学必须完全一致
       （tests/test_workspace.py 会逐位核对）。
    """
    qs = np.asarray(qs, dtype=float)
    if qs.ndim != 2 or qs.shape[1] != chain.n_joints:
        raise ValueError(f"姿态矩阵形状应为 (N, {chain.n_joints})，收到 {qs.shape}")

    n = qs.shape[0]
    T = np.tile(np.eye(4), (n, 1, 1))
    for index, link in enumerate(chain.links):
        T = T @ link.offset.A
        if link.motion == "Rz":
            theta = qs[:, index]
            c, s = np.cos(theta), np.sin(theta)
            R = np.zeros((n, 4, 4))
            R[:, 0, 0] = c
            R[:, 0, 1] = -s
            R[:, 1, 0] = s
            R[:, 1, 1] = c
            R[:, 2, 2] = 1.0
            R[:, 3, 3] = 1.0
        else:  # Tz：沿自身 z 平移
            R = np.tile(np.eye(4), (n, 1, 1))
            R[:, 2, 3] = qs[:, index]
        T = T @ R
    T = T @ chain.tool.A
    return T[:, :3, 3]


def directional_reach(
    chain: SerialChain,
    direction: np.ndarray,
    *,
    samples: int = 8000,
    seed: int = 0,
    limits: np.ndarray | None = None,
) -> ReachEstimate:
    """沿 `direction` 方向采样估计可达边界。

    limits: (n, 2) 的关节限位数组；None 表示按 ±180° 采样。
    """
    direction = np.asarray(direction, dtype=float).reshape(3)
    norm = float(np.linalg.norm(direction))
    if norm < 1e-12:
        raise ValueError("方向向量不能为零")
    direction = direction / norm

    qs = sample_joint_space(chain, samples, seed=seed, limits=limits)
    projections = fk_positions_batch(chain, qs) @ direction
    return ReachEstimate(
        chain_name=chain.name,
        direction=direction,
        radius=float(projections.max()),
        samples=samples,
        seed=seed,
        within_limits=limits is not None,
    )


def max_reach(
    chain: SerialChain, *, samples: int = 8000, seed: int = 0, limits: np.ndarray | None = None
) -> float:
    """采样估计的最大可达距离（所有方向里最大的那一个）。"""
    qs = sample_joint_space(chain, samples, seed=seed, limits=limits)
    return float(np.linalg.norm(fk_positions_batch(chain, qs), axis=1).max())


def sample_joint_space(
    chain: SerialChain, samples: int, *, seed: int = 0, limits: np.ndarray | None = None
) -> np.ndarray:
    """均匀采样关节空间，返回 (samples, n)。固定 seed，保证结论可复现。"""
    rng = np.random.default_rng(seed)
    if limits is None:
        return rng.uniform(*DEFAULT_LIMITS, size=(samples, chain.n_joints))
    limits = np.asarray(limits, dtype=float)
    if limits.shape != (chain.n_joints, 2):
        raise ValueError(f"关节限位形状应为 ({chain.n_joints}, 2)，收到 {limits.shape}")
    return rng.uniform(limits[:, 0], limits[:, 1], size=(samples, chain.n_joints))
