"""@file pose.py

@brief 位姿在几种表示之间互转 —— 对应课件 01_se3_basic.py 与第一节的旋转表示。

换成 spatialmath 之后，本节的重点从"怎么算"变成"**约定对不对**"：
同一个旋转，RPY 有 6 种取法、四元数有两种排法、轴角有 (轴,角) 与 (角,轴) 两种顺序。
库能算，但**约定要自己盯住** —— 下面每处都标了本项目采用的约定。

⚠️ 两个实测到的坑（详见 docs/GUIDE.md 第 7 章）：
  1. `SE3.RPY(r, p, y, order='zyx')` 才等于课件手写的 `Rz(yaw)·Ry(pitch)·Rx(roll)`；
     用默认的 `order='xyz'` 是**另一套相反的约定**（课件里两种都出现过）。
  2. `spatialmath.base.r2q()` 返回四元数是 **(w, x, y, z)**（实部在前），
     而 ROS / Eigen 惯例是 (x, y, z, w)（实部在后）—— 差一个位置，接错就静默出错。
"""

from __future__ import annotations

import numpy as np
import spatialmath.base as smb
from spatialmath import SE3

from ..contracts import MatrixBlock, Report, TextBlock, VectorBlock, fmt_vector

# 本项目的 RPY 约定（与课件 01 的手写公式一致）
RPY_ORDER = "zyx"


def run(
    x: float,
    y: float,
    z: float,
    roll_deg: float,
    pitch_deg: float,
    yaw_deg: float,
    point: tuple[float, float, float] = (0.1, 0.0, 0.0),
) -> Report:
    """@brief 一个位姿的"全身照"：同一姿态用四种表示各写一遍，再互相重建校验。

    对应课件 01_se3_basic.py。长度/角度都按人读的顺序给（角度传度数、内部转弧度），
    并且每处都标出本项目采用的约定 —— 约定接错不会报错，只会静默给出另一个姿态。

    @param x 平移 x [m]
    @param y 平移 y [m]
    @param z 平移 z [m]
    @param roll_deg 绕 x 轴转角 [°]
    @param pitch_deg 绕 y 轴转角 [°]
    @param yaw_deg 绕 z 轴转角 [°]
    @param point 用来演示点变换的测试点 [m]
    @return 报告；fields 里带齐四套表示，供 --json 或测试取值
    """
    roll, pitch, yaw = np.deg2rad([roll_deg, pitch_deg, yaw_deg])
    T = SE3(x, y, z) * SE3.RPY(roll, pitch, yaw, order=RPY_ORDER)
    target = np.asarray(point, dtype=float)

    angle, axis = smb.tr2angvec(T.R)  # ⚠️ 返回顺序是 (角度, 轴)
    quat_wxyz = smb.r2q(T.R)  # ⚠️ 实部在前
    rpy_back = T.rpy(order=RPY_ORDER)
    p_base = (T * SE3(target)).t
    # 只平移不旋转 —— 用来对照"少乘了旋转会错成什么样"
    p_translate_only = target + np.asarray(T.t)
    # 四种表示互相重建，验证"它们确实等价"
    rebuilt = max(
        float(np.abs(smb.q2r(quat_wxyz) - T.R).max()),
        float(np.abs(smb.angvec2r(angle, axis) - T.R).max()),
    )

    return Report(
        title="位姿的几种表示（spatialmath）",
        blocks=(
            TextBlock.of(
                f"六元组 (x, y, z, roll, pitch, yaw) = "
                f"({x}, {y}, {z}, {roll_deg}°, {pitch_deg}°, {yaw_deg}°)",
                f"约定：R = Rz(yaw)·Ry(pitch)·Rx(roll)，对应 "
                f"SE3.RPY(roll, pitch, yaw, order={RPY_ORDER!r})",
            ),
            MatrixBlock("1) 旋转矩阵 R", T.R),
            TextBlock.of(
                "2) R 是不是一个合法旋转？（库算出来的也照验不误）",
                f"   正交性  ‖RᵀR - I‖ = {np.abs(T.R.T @ T.R - np.eye(3)).max():.3e}",
                f"   det(R) = {np.linalg.det(T.R):.12f}",
            ),
            MatrixBlock("3) 齐次变换矩阵 ^A T_B", T.A),
            VectorBlock("4) 点变换 p_base", p_base),
            VectorBlock("   只平移不旋转会得到", p_translate_only, suffix="　← 少乘了旋转就是错的"),
            TextBlock.of(
                "5) 同一个姿态的另外三种写法（互相重建）",
                f"   轴角  ：绕 {fmt_vector(axis)} 转 {np.rad2deg(angle):.4f}°",
                f"   四元数：(w, x, y, z) = {fmt_vector(quat_wxyz, 6)}   ← 实部在前！",
                f"   RPY 回代：{fmt_vector(np.rad2deg(rpy_back), 4)}°（order={RPY_ORDER}）",
                f"   重建出的 R 与原始 R 最大偏差 = {rebuilt:.3e}",
            ),
        ),
        fields={
            "xyz": [x, y, z],
            "rpy_deg": [roll_deg, pitch_deg, yaw_deg],
            "rpy_order": RPY_ORDER,
            "R": T.R,
            "A": T.A,
            "point": target.tolist(),
            "p_base": p_base,
            "axis": axis,
            "angle_rad": angle,
            "quaternion_wxyz": quat_wxyz,
            "rebuild_gap": rebuilt,
        },
    )
