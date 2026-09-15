"""@file chain.py
@brief 通用串联链：FK 与几何雅可比，以及 ETS / MDH 两种参数化。

【模型怎么描述】一条串联链就是“固定件 + 关节”的交替序列：
    base ──[安装变换]──(关节1)──[安装变换]──(关节2)── ... ──> tool

  * `SerialChain`（ETS 形式）：每一项 = 一个固定安装变换 + 绕自身 z 轴的转动/移动。
    这就是 Franka URDF 的写法，也是课件里 roboticstoolbox 打印出来的那张表。
  * `DHRobot`（改进 DH / Craig 形式）：每一项 = Rx(α)·Tx(a)·Rz(θ)·Tz(d)。
    机械臂教材的标准写法。

两者是同一台机器人的两种记账方式。本模块**故意把两套 FK 各写一遍**（不互相调用），
这样“两种参数化给出同一结果”就成了一次真正的交叉验证：谁写错了都会立刻暴露。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .exceptions import KinematicsError
from .se3 import SE3

# ETS 里每个关节只有两种自由度：绕自身 z 转、沿自身 z 移
REVOLUTE = "Rz"
PRISMATIC = "Tz"


@dataclass(frozen=True)
class ETSLink:
    """一个关节单元：固定安装变换 `offset` + 关节自由度 `motion`。"""

    offset: SE3 = field(default_factory=SE3.identity)
    motion: str = REVOLUTE
    name: str = ""

    def __post_init__(self) -> None:
        if self.motion not in (REVOLUTE, PRISMATIC):
            raise KinematicsError(f"未知的关节自由度类型: {self.motion}")

    def motion_matrix(self, q: float) -> np.ndarray:
        return SE3.Rz(q).A if self.motion == REVOLUTE else SE3.Trans(0.0, 0.0, q).A


@dataclass(frozen=True)
class SerialChain:
    """串联链。q 的长度必须等于关节数。"""

    links: tuple[ETSLink, ...]
    tool: SE3 = field(default_factory=SE3.identity)
    name: str = ""

    def __post_init__(self) -> None:
        if len(self.links) == 0:
            raise KinematicsError("串联链至少要有一个关节")

    @property
    def n_joints(self) -> int:
        return len(self.links)

    def _check_q(self, q: np.ndarray) -> np.ndarray:
        q = np.asarray(q, dtype=float).reshape(-1)
        if q.size != self.n_joints:
            raise KinematicsError(
                f"{self.name or '串联链'} 有 {self.n_joints} 个关节，收到 {q.size} 个关节角"
            )
        return q

    def joint_frames(self, q: np.ndarray) -> list[np.ndarray]:
        """返回每个关节“运动之前”的坐标系在 base 下的 4x4 位姿。

        几何雅可比只需要这些坐标系：关节 i 的转轴就是它的 z 轴。
        """
        q = self._check_q(q)
        frames: list[np.ndarray] = []
        T = np.eye(4)
        for link, qi in zip(self.links, q, strict=True):
            T = T @ link.offset.A
            frames.append(T.copy())
            T = T @ link.motion_matrix(qi)
        return frames

    def fk(self, q: np.ndarray) -> SE3:
        """正运动学：关节角 -> 末端位姿。"""
        q = self._check_q(q)
        T = np.eye(4)
        for link, qi in zip(self.links, q, strict=True):
            T = T @ link.offset.A @ link.motion_matrix(qi)
        return SE3.from_matrix(T @ self.tool.A)

    def jacobian(self, q: np.ndarray) -> np.ndarray:
        """几何雅可比 6×n，前 3 行是线速度、后 3 行是角速度（都在 base 下表达）。

        转动关节第 i 列：线速度 = z_i × (p_ee - p_i)，角速度 = z_i。
        直觉：关节只提供绕自身轴的角速度，末端因此产生一段切向线速度。
        移动关节第 i 列：线速度 = z_i，角速度 = 0。
        """
        q = self._check_q(q)
        frames = self.joint_frames(q)
        p_ee = self.fk(q).t

        J = np.zeros((6, self.n_joints))
        for i, (link, T_i) in enumerate(zip(self.links, frames, strict=True)):
            z_i = T_i[:3, :3] @ np.array([0.0, 0.0, 1.0])
            if link.motion == REVOLUTE:
                p_i = T_i[:3, 3]
                J[:3, i] = np.cross(z_i, p_ee - p_i)
                J[3:, i] = z_i
            else:
                J[:3, i] = z_i
        return J

    def fk_positions(self, q: np.ndarray) -> np.ndarray:
        """返回每个关节坐标系原点 + 末端原点的位置，供 3D 画骨架用。"""
        q = self._check_q(q)
        points = [np.zeros(3)]
        T = np.eye(4)
        for link, qi in zip(self.links, q, strict=True):
            T = T @ link.offset.A @ link.motion_matrix(qi)
            points.append(T[:3, 3].copy())
        points.append((T @ self.tool.A)[:3, 3].copy())
        return np.array(points)


@dataclass(frozen=True)
class DHLink:
    """改进 DH 参数（Craig）：α 绕 x 的扭转、a 沿 x 的连杆长度、d 沿 z 的偏置。"""

    alpha: float = 0.0
    a: float = 0.0
    d: float = 0.0
    theta_offset: float = 0.0


@dataclass(frozen=True)
class DHRobot:
    """改进 DH 形式：T_i = Rx(α_{i-1})·Tx(a_{i-1})·Rz(θ_i)·Tz(d_i)。"""

    links: tuple[DHLink, ...]
    tool: SE3 = field(default_factory=SE3.identity)
    name: str = ""

    @property
    def n_joints(self) -> int:
        return len(self.links)

    def fk(self, q: np.ndarray) -> SE3:
        """独立实现的 DH 正运动学（刻意不复用 SerialChain）。"""
        q = np.asarray(q, dtype=float).reshape(-1)
        if q.size != self.n_joints:
            raise KinematicsError(
                f"{self.name or 'DH 机器人'} 有 {self.n_joints} 个关节，收到 {q.size} 个关节角"
            )
        T = np.eye(4)
        for link, qi in zip(self.links, q, strict=True):
            T = (
                T
                @ SE3.Rx(link.alpha).A
                @ SE3.Trans(link.a, 0.0, 0.0).A
                @ SE3.Rz(qi + link.theta_offset).A
                @ SE3.Trans(0.0, 0.0, link.d).A
            )
        return SE3.from_matrix(T @ self.tool.A)

    def to_serial_chain(self) -> SerialChain:
        """转成等价的 ETS 形式，用来复用几何雅可比与 IK。

        依据 Rz(θ)·Tz(d) = Tz(d)·Rz(θ)（同轴转动与平移可交换），把 Tz(d) 并进固定部分。
        """
        links = tuple(
            ETSLink(
                offset=(
                    SE3.Rx(link.alpha) * SE3.Trans(link.a, 0.0, 0.0) * SE3.Trans(0.0, 0.0, link.d)
                ),
                motion=REVOLUTE,
            )
            for link in self.links
        )
        return SerialChain(links=links, tool=self.tool, name=self.name)
