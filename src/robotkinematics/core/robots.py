"""@file robots.py

@brief Panda 模型的构造与查询 —— 底层是 robotics toolbox，这里只做"配成我们需要的形态"。

【本层剩下的领域逻辑并不多，但都是库不提供的】
  * 末端帧的选择：库的 `Panda()` 自带夹爪，`fkine(q)` 默认返回**夹爪 TCP**，
    要拿法兰必须显式 `end="panda_link8"` —— 这个约定太容易忘，所以在这里封一层；
  * 画骨架要的"每个 link 的原点"：库给的是 `fkine_all`，返回的是位姿对象，要摊平成位置数组。

⚠️ 除了这几处，本项目的运动学全部由库负责（FK / IK / 雅可比 / 限位 / 可操作度）。
机器人固定为 Franka Panda。
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
    """@brief 建一个 Franka Panda 模型，末端按 frame 取法兰或夹爪 TCP。

    ⚠️ 库的 `Panda()` 自带夹爪，所以"末端到底是哪个坐标系"在建模型时就要定下来。
       两种末端差 103.4mm —— 详见 docs/GUIDE.md 第 7.3 节。

    @param frame 末端帧：FLANGE（法兰 panda_link8）或 TCP（夹爪）
    @return robotics toolbox 的 ERobot 实例（名字里会标出用的是哪个末端）
    @throws ValueError frame 既不是 FLANGE 也不是 TCP
    """
    if frame not in (FLANGE, TCP):
        raise ValueError(f"未知的末端帧 {frame!r}（可选 {FLANGE} / {TCP}）")
    robot = rtb.models.Panda()
    robot.name = f"Franka Panda（{'法兰 panda_link8' if frame == FLANGE else '夹爪 TCP'}）"
    return robot


def panda_end(frame: str = FLANGE) -> str | None:
    """@brief 把本项目的末端帧名字翻译成库 `fkine(end=…)` 要的参数。

    @param frame 末端帧：FLANGE 或 TCP
    @return 法兰返回 "panda_link8"；夹爪返回 None（库的默认末端就是夹爪）
    """
    return PANDA_FLANGE_LINK if frame == FLANGE else None


def end_pose(robot: rtb.Robot, q: np.ndarray, frame: str = FLANGE) -> SE3:
    """@brief 算末端位姿（位置 + 姿态）。

    @param robot Panda 模型实例
    @param q 关节角 [rad]，一维数组，长度等于关节数
    @param frame 末端帧：FLANGE 或 TCP
    @return 4×4 位姿对象（spatialmath.SE3）
    """
    q = np.asarray(q, dtype=float).reshape(-1)
    end = panda_end(frame)
    return robot.fkine(q, end=end) if end is not None else robot.fkine(q)


def end_positions(robot: rtb.Robot, qs: np.ndarray, frame: str = FLANGE) -> np.ndarray:
    """@brief 批量算末端位置，一次算 N 个姿态。

    库的 `fkine` 原生支持 (N, n) 的批量输入、内部向量化 —— 比逐个调用快两个数量级，
    所以工作空间采样（几千个姿态）是瞬间完成的。

    ⚠️ 判别单/批量**不能**用 `isinstance(poses, SE3)`：批量返回的也是 SE3（值数组），
       而且它的 `.shape` 还报 (4,4)（骗人的）。只能看 `.t` 的维数：单个 (3,)，批量 (N,3)。

    @param robot Panda 模型实例
    @param qs 关节角，形状 (N, n)；传单个 (n,) 也可以
    @param frame 末端帧：FLANGE 或 TCP
    @return 位置矩阵，形状 (N, 3)；传单个姿态时返回 (1, 3)
    """
    qs = np.asarray(qs, dtype=float)
    end = panda_end(frame)
    poses = robot.fkine(qs, end=end) if end is not None else robot.fkine(qs)
    t = np.asarray(poses.t, dtype=float)
    if t.ndim == 1:  # 单个姿态
        return t.reshape(1, 3)
    return t if t.shape[-1] == 3 else t.T


def joint_limits(robot: rtb.Robot) -> np.ndarray | None:
    """@brief 取关节限位。

    模型没定义限位、或限位里含 inf 时返回 None —— 如实告知"不知道"，
    而不是编一组数出来（调用方据此跳过限位校验）。

    @param robot robots toolbox 的模型实例
    @return 形状 (n, 2) 的数组（每行是 [下限, 上限]），或 None
    """
    qlim = getattr(robot, "qlim", None)
    if qlim is None:
        return None
    qlim = np.asarray(qlim, dtype=float)
    return None if not np.isfinite(qlim).all() else qlim.T  # 库是 (2, n)，我们统一成 (n, 2)


def link_points(robot: rtb.Robot, q: np.ndarray) -> np.ndarray:
    """@brief 取每个 link 坐标系的原点，供画机械臂骨架用。

    @param robot robots toolbox 的模型实例
    @param q 关节角 [rad]，长度等于关节数
    @return 位置数组，形状 (n+2, 3)：第一行是 base 原点，中间是各关节，最后一行是末端
    """
    q = np.asarray(q, dtype=float).reshape(-1)
    poses = robot.fkine_all(q)[:-1]  # 最后一个重复了末端，去掉
    points = [np.zeros(3)] + [np.asarray(T.t, dtype=float) for T in poses]
    points.append(np.asarray(robot.fkine(q).t, dtype=float))
    return np.array(points)


def joint_names(robot: rtb.Robot) -> list[str]:
    """@brief 取各关节的名字（库里的名字，如 panda_joint1）。

    @param robot robots toolbox 的模型实例
    @return 关节名列表，长度等于关节数
    """
    return [link.name for link in robot.links if link.isjoint]


def limit_margin(q: np.ndarray, limits: np.ndarray | None) -> float:
    """@brief 算"离最近的关节限位还有多少"。

    择优时用它挑"离限位远"的解 —— 贴限位的解在真机上很危险。

    @param q 关节角 [rad]
    @param limits 形状 (n, 2) 的限位数组；None 表示没有限位信息
    @return 最小余量 [rad]：越界为负；limits 为 None 时返回 inf（当作不受限）
    """
    if limits is None:
        return float("inf")
    q = np.asarray(q, dtype=float).reshape(-1)
    limits = np.asarray(limits, dtype=float)
    return float(min((q - limits[:, 0]).min(), (limits[:, 1] - q).min()))


def out_of_limits(q: np.ndarray, limits: np.ndarray | None, tol: float = 1e-9) -> list[int]:
    """@brief 找出越界的关节下标。

    只用于提示 —— 数学上库对任何 q 都算得出 FK，越界只是"真机摆不出来"。

    @param q 关节角 [rad]
    @param limits 形状 (n, 2) 的限位数组；None 表示没有限位信息（返回空列表）
    @param tol 判定容差 [rad]，避免浮点噪声把贴边的解判成越界
    @return 越界关节的下标列表，按从小到大排列
    """
    if limits is None:
        return []
    q = np.asarray(q, dtype=float).reshape(-1)
    limits = np.asarray(limits, dtype=float)
    return [
        i
        for i, (qi, (lo, hi)) in enumerate(zip(q, limits, strict=True))
        if qi < lo - tol or qi > hi + tol
    ]
