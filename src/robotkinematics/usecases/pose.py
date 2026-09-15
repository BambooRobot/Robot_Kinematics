"""@file pose.py
@brief 位姿在几种表示之间互转 —— 对应课件 01_se3_basic.py 与第一节的旋转表示。

课件演示的是"六元组 → 齐次变换矩阵 → 点变换"；这里把同一个位姿的四种表示
（旋转矩阵、RPY、轴角、四元数）放在一起互验，因为工程里踩的坑多半出在"我以为它们等价"。
"""

from __future__ import annotations

import numpy as np

from ..contracts import MatrixBlock, Report, TextBlock, VectorBlock, fmt_vector
from ..core import rotations
from ..core.se3 import SE3


def run(
    x: float,
    y: float,
    z: float,
    roll_deg: float,
    pitch_deg: float,
    yaw_deg: float,
    point: tuple[float, float, float] = (0.1, 0.0, 0.0),
) -> Report:
    roll, pitch, yaw = np.deg2rad([roll_deg, pitch_deg, yaw_deg])
    T = SE3.from_pose6(x, y, z, roll, pitch, yaw)
    target = np.asarray(point, dtype=float)

    axis, angle = rotations.matrix_to_axang(T.R)
    quat = rotations.matrix_to_quat(T.R)
    rpy_back = rotations.matrix_to_rpy(T.R)
    p_base = T.apply(target)
    # 只平移不旋转 —— 用来对照"少乘了旋转会错成什么样"
    p_translate_only = target + T.t
    representation_gap = max(
        float(np.abs(rotations.axang_to_matrix(axis, angle) - T.R).max()),
        float(np.abs(rotations.quat_to_matrix(quat) - T.R).max()),
    )

    return Report(
        title="位姿的几种表示",
        blocks=(
            TextBlock.of(
                f"六元组 (x, y, z, roll, pitch, yaw) = "
                f"({x}, {y}, {z}, {roll_deg}°, {pitch_deg}°, {yaw_deg}°)",
            ),
            MatrixBlock("1) 旋转部分 R = Rz(yaw) · Ry(pitch) · Rx(roll)", T.R),
            TextBlock.of(
                "2) R 是不是一个合法旋转？两条都要满足，缺一不可",
                f"   正交性  ‖RᵀR - I‖ = {np.abs(T.R.T @ T.R - np.eye(3)).max():.3e}",
                f"   det(R) = {np.linalg.det(T.R):.12f}   "
                "（必须是 +1；-1 表示含镜像，关节给不出来）",
                f"   结论：{'合法' if rotations.is_rotation(T.R) else '非法'}",
            ),
            MatrixBlock("3) 齐次变换矩阵 ^A T_B（左上 3x3 是姿态，右上 3x1 是位置）", T.A),
            TextBlock.of(
                f"4) 点变换：p_tool = {fmt_vector(target, 3)} → p_base = R · p_tool + t",
            ),
            VectorBlock("   p_base", p_base),
            VectorBlock("   只平移不旋转会得到", p_translate_only, suffix="　← 少乘了旋转就是错的"),
            TextBlock.of("5) 同一个姿态的另外三种写法（互相验证）"),
            TextBlock.of(
                f"   轴角  ：绕 {fmt_vector(axis)} 转 {np.rad2deg(angle):.4f}°",
                f"   四元数：(x, y, z, w) = {fmt_vector(quat, 6)}  范数 {np.linalg.norm(quat):.12f}",
                f"   RPY 回代：({np.rad2deg(rpy_back[0]):.4f}°, {np.rad2deg(rpy_back[1]):.4f}°, "
                f"{np.rad2deg(rpy_back[2]):.4f}°)",
                f"   三种写法重建出的 R 最大偏差 = {representation_gap:.3e}",
            ),
        ),
        fields={
            "xyz": [x, y, z],
            "rpy_deg": [roll_deg, pitch_deg, yaw_deg],
            "R": T.R,
            "A": T.A,
            "point": target.tolist(),
            "p_base": p_base,
            "axis": axis,
            "angle_rad": angle,
            "quaternion_xyzw": quat,
            "is_rotation": rotations.is_rotation(T.R),
            "representation_gap": representation_gap,
        },
    )
