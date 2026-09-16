"""@file robots.py
@brief 机器人模型的构造与查询 —— 底层是 robotics toolbox，这里只做"配成我们需要的形态"。

【本层剩下的领域逻辑并不多，但都是库不提供的】
  * 末端帧的选择：库的 `Panda()` 自带夹爪，`fkine(q)` 默认返回**夹爪 TCP**，
    要拿法兰必须显式 `end="panda_link8"` —— 这个约定太容易忘，所以在这里封一层；
  * 二连杆的杆长：`rtb.models.DH.Planar2()` 的 `a` 是只读属性，要改杆长只能自建 `DHRobot`；
  * 画骨架要的"每个 link 的原点"：库给的是 `fkine_all`，返回的是位姿对象，要摊平成位置数组。

⚠️ 除了这几处，本项目的运动学全部由库负责（FK / IK / 雅可比 / 限位 / 可操作度）。
"""

from __future__ import annotations

import numpy as np
import roboticstoolbox as rtb
from spatialmath import SE3

# Panda 的两种末端定义（详见 docs/KNOWLEDGE_MAP.md 第 8 节）
FLANGE = "flange"  # 法兰 panda_link8
TCP = "tcp"  # 夹爪 TCP（= 库的默认末端）

PANDA_FLANGE_LINK = "panda_link8"


def panda(frame: str = FLANGE) -> rtb.ERobot:
    """Franka Panda 模型。frame 决定末端取法兰还是夹爪 TCP。"""
    if frame not in (FLANGE, TCP):
        raise ValueError(f"未知的末端帧 {frame!r}（可选 {FLANGE} / {TCP}）")
    robot = rtb.models.Panda()
    robot.name = f"Franka Panda（{'法兰 panda_link8' if frame == FLANGE else '夹爪 TCP'}）"
    return robot


def panda_end(frame: str = FLANGE) -> str | None:
    """把我们的末端帧名字翻译成库的 `end=` 参数（None = 库默认的夹爪）。"""
    return PANDA_FLANGE_LINK if frame == FLANGE else None


def planar(l1: float = 1.0, l2: float = 1.0) -> rtb.DHRobot:
    """平面二连杆。

    ⚠️ 不用 `rtb.models.DH.Planar2()`：它的杆长 `a` 是只读属性，改不了。
       要自定义杆长只能自建 DHRobot（这也是库的推荐做法）。
    """
    if l1 <= 0 or l2 <= 0:
        raise ValueError(f"连杆长度必须为正，收到 l1={l1}, l2={l2}")
    return rtb.DHRobot(
        [rtb.RevoluteDH(a=l1), rtb.RevoluteDH(a=l2)],
        name=f"planar2r(L1={l1:g}, L2={l2:g})",
    )


def end_pose(robot: rtb.Robot, q: np.ndarray, frame: str = FLANGE) -> SE3:
    """末端位姿。Panda 需要按 frame 指定 end；其他模型忽略 frame。"""
    q = np.asarray(q, dtype=float).reshape(-1)
    end = panda_end(frame) if _is_panda(robot) else None
    return robot.fkine(q, end=end) if end is not None else robot.fkine(q)


def end_positions(robot: rtb.Robot, qs: np.ndarray, frame: str = FLANGE) -> np.ndarray:
    """批量末端位置 (N, 3)。

    库的 `fkine` 原生支持 (N, n) 的批量输入，内部向量化 —— 比逐个调用快两个数量级，
    所以工作空间采样（几千个姿态）是瞬间完成的。

    ⚠️ 判别单/批量**不能**用 isinstance(SE3)：批量返回的也是 SE3（值数组），
       而且它的 `.shape` 还报 (4,4)（骗人的）。可靠的办法是看 `.t` 的维数：
       单个 → (3,)，批量 → (N, 3)。
    """
    qs = np.asarray(qs, dtype=float)
    end = panda_end(frame) if _is_panda(robot) else None
    poses = robot.fkine(qs, end=end) if end is not None else robot.fkine(qs)
    t = np.asarray(poses.t, dtype=float)
    if t.ndim == 1:  # 单个姿态
        return t.reshape(1, 3)
    return t if t.shape[-1] == 3 else t.T


def joint_limits(robot: rtb.Robot) -> np.ndarray | None:
    """关节限位 (n, 2)；模型没定义限位时返回 None（如实告知，不编一组）。"""
    qlim = getattr(robot, "qlim", None)
    if qlim is None:
        return None
    qlim = np.asarray(qlim, dtype=float)
    return None if not np.isfinite(qlim).all() else qlim.T  # 库是 (2, n)，我们统一成 (n, 2)


def link_points(robot: rtb.Robot, q: np.ndarray) -> np.ndarray:
    """每个 link 坐标系原点 + 末端，共 (n+2, 3) 个点 —— 画骨架用。"""
    q = np.asarray(q, dtype=float).reshape(-1)
    poses = robot.fkine_all(q)[:-1]  # 最后一个重复了末端，去掉
    points = [np.zeros(3)] + [np.asarray(T.t, dtype=float) for T in poses]
    points.append(np.asarray(robot.fkine(q).t, dtype=float))
    return np.array(points)


def joint_names(robot: rtb.Robot) -> list[str]:
    return [link.name for link in robot.links if link.isjoint]


def limit_margin(q: np.ndarray, limits: np.ndarray | None) -> float:
    """离最近的关节限位还有多少 [rad]；越界为负；无限位信息时返回 inf。"""
    if limits is None:
        return float("inf")
    q = np.asarray(q, dtype=float).reshape(-1)
    limits = np.asarray(limits, dtype=float)
    return float(min((q - limits[:, 0]).min(), (limits[:, 1] - q).min()))


def out_of_limits(q: np.ndarray, limits: np.ndarray | None, tol: float = 1e-9) -> list[int]:
    """越界的关节下标（用于提示；数学上库对任何 q 都算得出 FK）。"""
    if limits is None:
        return []
    q = np.asarray(q, dtype=float).reshape(-1)
    limits = np.asarray(limits, dtype=float)
    return [
        i
        for i, (qi, (lo, hi)) in enumerate(zip(q, limits, strict=True))
        if qi < lo - tol or qi > hi + tol
    ]


def _is_panda(robot: rtb.Robot) -> bool:
    return "panda" in (robot.name or "").lower() or any(
        link.name.startswith("panda_link") for link in robot.links
    )
