"""@file panda.py
@brief Franka Emika Panda（7 自由度协作机械臂）的运动学模型。

【参数从哪来】下面每个固定变换都对应 Franka 官方 URDF 里一个关节的 <origin>：
    <origin xyz="..." rpy="..."/> —— 转动部分在这个机器人上只有绕 x 的 ±90°。
课件里 roboticstoolbox 打印出的那张 ETS 表与它逐项一致，可以作为对照。

【为什么还要写一份 DH】机械臂教材一律用 DH 参数，而 URDF/ETS 是工程主流。
两套参数化描述同一台机器人，本项目把两套 FK 各写一遍，互为交叉验证：
tests/test_panda_fk.py 会断言两者在随机姿态下逐位相等。
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from .chain import DHLink, DHRobot, ETSLink, SerialChain
from .se3 import SE3

DEG = np.deg2rad

# ── 关节顺序与课件一致：q = [q1..q7]，单位 rad ───────────────────────────────
PANDA_JOINT_NAMES = tuple(f"panda_joint{i}" for i in range(1, 8))

# Franka 官方数据表的关节限位 [rad]（下限, 上限）。
# ⚠️ 注意 q4 的区间不含 0：课件里的“全零姿态”在关节限位上其实是不可达的，
#    它是一个数学上的参考位形，不是一个真机能摆出来的姿态。
PANDA_JOINT_LIMITS = (
    (-2.8973, 2.8973),
    (-1.7628, 1.7628),
    (-2.8973, 2.8973),
    (-3.0718, -0.0698),
    (-2.8973, 2.8973),
    (-0.0175, 3.7525),
    (-2.8973, 2.8973),
)

# 课件 data/panda_cases.csv 里的三个姿态
PANDA_Q_ZERO = np.zeros(7)
PANDA_Q_PPT_GOAL = np.array([0.0, -0.4, 0.0, -2.2, 0.0, 2.0, 0.7853981634])
PANDA_Q_IK_SEED = np.array([0.0, -0.3, 0.0, -2.2, 0.0, 2.0, 0.8])

# ── 夹爪 TCP：法兰再往前 103.4mm、并绕 z 转 -45° ───────────────────────────
# 这就是 Franka 官方夹爪 TCP 的约定（franka_ros 里 panda_hand_tcp 的静态变换），
# 也是 roboticstoolbox 的 Panda 模型自带的 panda_hand 夹爪。
# 已用 roboticstoolbox 1.4.3 实测核对：法兰 -> TCP 的固定变换在任意姿态下恒定，
# 最大差异 2e-16。加上它之后，本项目的零位姿 FK 与课件 run_logs 记录的数值完全一致。
PANDA_HAND_TCP_OFFSET = SE3.Trans(0.0, 0.0, 0.1034) * SE3.Rz(DEG(-45))

# ── 课件日志里的参照值：它记录的是**夹爪 TCP** 帧，不是法兰 ─────────────────
# 来源：课程 run_logs/08_franka_panda_fk.log，零位姿的 robot.fkine(zeros) 输出。
# ⚠️ 这个坑值得记住：roboticstoolbox 的 Panda 模型带一个夹爪，`fkine(q)` 默认返回的是
#    **夹爪 TCP**（法兰 + 103.4mm + 绕 z 转 45°），不是法兰 panda_link8。
#    本项目默认给的是法兰帧，所以两者差 0.1034 m —— 这不是谁算错了，而是“末端”的定义不同。
#    用 panda_hand_tcp_chain() 即可复现课件日志的数值（见 tests/test_panda_fk.py）。
COURSE_LOG_ZERO_POSE_TOOL_T = (0.088, 0.0, 0.8226)
COURSE_LOG_SOURCE = "课程 run_logs/08_franka_panda_fk.log（夹爪 TCP 帧）"


def panda_urdf_chain() -> SerialChain:
    """ETS 形式：与 Franka URDF 的关节 origin 一一对应。"""
    links = (
        ETSLink(offset=SE3.Trans(0.0, 0.0, 0.333), name="joint1"),
        ETSLink(offset=SE3.Rx(DEG(-90)), name="joint2"),
        ETSLink(offset=SE3.Trans(0.0, -0.316, 0.0) * SE3.Rx(DEG(90)), name="joint3"),
        ETSLink(offset=SE3.Trans(0.0825, 0.0, 0.0) * SE3.Rx(DEG(90)), name="joint4"),
        ETSLink(offset=SE3.Trans(-0.0825, 0.384, 0.0) * SE3.Rx(DEG(-90)), name="joint5"),
        ETSLink(offset=SE3.Rx(DEG(90)), name="joint6"),
        ETSLink(offset=SE3.Trans(0.088, 0.0, 0.0) * SE3.Rx(DEG(90)), name="joint7"),
    )
    # 法兰 -> link8（工具安装面）的固定偏移
    return SerialChain(links=links, tool=SE3.Trans(0.0, 0.0, 0.107), name="Franka Panda (URDF/ETS)")


def panda_dh_robot() -> DHRobot:
    """改进 DH 形式（Craig）：教材上的那张标准 Panda 参数表。"""
    table = (
        DHLink(alpha=0.0, a=0.0, d=0.333),
        DHLink(alpha=DEG(-90), a=0.0, d=0.0),
        DHLink(alpha=DEG(90), a=0.0, d=0.316),
        DHLink(alpha=DEG(90), a=0.0825, d=0.0),
        DHLink(alpha=DEG(-90), a=-0.0825, d=0.384),
        DHLink(alpha=DEG(90), a=0.0, d=0.0),
        DHLink(alpha=DEG(90), a=0.088, d=0.0),
    )
    return DHRobot(links=table, tool=SE3.Trans(0.0, 0.0, 0.107), name="Franka Panda (MDH)")


def panda_hand_tcp_chain() -> SerialChain:
    """带 Franka 夹爪的模型：末端是夹爪 TCP，而不是法兰。

    课件日志里那组零位姿数值（t = [0.088, 0, 0.8226]）就是这个帧的 —— 用它能逐位复现。
    """
    flange = panda_urdf_chain()
    return replace(flange, tool=flange.tool * PANDA_HAND_TCP_OFFSET, name="Franka Panda + 夹爪 TCP")


def panda() -> SerialChain:
    """默认模型：与 URDF 一致的 ETS 链，末端是法兰（panda_link8）。"""
    return panda_urdf_chain()


def joints_out_of_limits(q: np.ndarray, tol: float = 1e-9) -> list[str]:
    """返回越界的关节名。只用于提示 —— 数学上 FK 对任何 q 都算得出来。"""
    q = np.asarray(q, dtype=float).reshape(-1)
    out = [
        name
        for name, qi, (lo, hi) in zip(PANDA_JOINT_NAMES, q, PANDA_JOINT_LIMITS, strict=True)
        if qi < lo - tol or qi > hi + tol
    ]
    return out
