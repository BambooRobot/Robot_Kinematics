"""@file transforms.py
@brief 相机目标 → 机器人本体 —— 对应课件 02_transform_chain_camera_to_base.py 与练习 1。

核心一句话：同一个杯子，换个 frame 表达，数值就变了；杯子一动没动。
"""

from __future__ import annotations

import numpy as np

from ..contracts import MatrixBlock, Report, TableBlock, TextBlock, VectorBlock, fmt_vector
from ..core.se3 import SE3

DEG = np.deg2rad


def build_chain(case) -> tuple[SE3, SE3, SE3]:
    """返回 (^base T_camera, ^camera T_cup, ^base T_cup)。

    这是这个用例里唯一的"数学"，放在函数里而不是塞进报告函数，
    既方便测试，也方便别的用例（比如将来做抓取规划）复用。
    """
    T_base_camera = SE3.Trans(*case.camera_xyz) * SE3.Rz(DEG(case.camera_yaw_deg))
    T_camera_cup = SE3.Trans(*case.cup_xyz)
    return T_base_camera, T_camera_cup, T_base_camera * T_camera_cup


def case_report(case) -> Report:
    T_base_camera, T_camera_cup, T_base_cup = build_chain(case)
    return Report(
        title=f"坐标变换链：{case.case_id}",
        blocks=(
            TextBlock.of(
                f"算例说明：{case.description}",
                "",
                "相机装在本体上，杯子在相机视野里，机械臂却只认本体坐标系：",
                "  ^base T_cup = ^base T_camera · ^camera T_cup",
            ),
            MatrixBlock(
                f"1) ^base T_camera（相机相对本体：平移 + 绕 z 转 {case.camera_yaw_deg}°）",
                T_base_camera.A,
            ),
            MatrixBlock("2) ^camera T_cup（杯子相对相机，纯平移）", T_camera_cup.A),
            MatrixBlock("3) ^base T_cup（相乘的结果）", T_base_cup.A),
            VectorBlock("杯子在 base frame 下的位置 xyz", T_base_cup.t),
            TextBlock.of(
                "",
                "⚠️ 相乘顺序不能反：^base T_camera · ^camera T_cup 才是“先把杯子表达成相机里的量，",
                "   再把这个量表达成本体里的量”。反过来乘得到的是另一个物理含义。",
            ),
        ),
        fields={
            "case_id": case.case_id,
            "base_camera": T_base_camera.A,
            "camera_cup": T_camera_cup.A,
            "base_cup": T_base_cup.A,
            "cup_in_base": T_base_cup.t,
        },
    )


def all_cases_report(cases: list, reference_id: str = "ppt_case") -> Report:
    """把整张算例表跑一遍，并与基准算例对比 —— 回答"杯子没动，数值为什么变了"。"""
    results = {case.case_id: build_chain(case)[2] for case in cases}
    rows = tuple(
        (case.case_id, fmt_vector(results[case.case_id].t), case.description) for case in cases
    )

    blocks: list = [TableBlock(("算例", "杯子在 base 下的位置", "说明"), rows, ("<", ">", "<"))]
    reference = results.get(reference_id)
    if reference is not None:
        deltas = tuple(
            (
                case.case_id,
                fmt_vector(results[case.case_id].t - reference.t),
            )
            for case in cases
            if case.case_id != reference_id
        )
        blocks += [
            TextBlock.of(
                "",
                f"以 {reference_id} 为基准（位置 {fmt_vector(reference.t)}）：",
            ),
            TableBlock(("算例", "相对基准的变化量"), deltas, ("<", ">")),
            TextBlock.of(
                "",
                "★ 杯子/相机参数改了，同一个物理目标的坐标表达自然跟着变 —— 杯子一动没动。",
            ),
        ]

    return Report(
        title="练习 1：改一个量，观察 base frame 下的结果怎么变",
        blocks=tuple(blocks),
        fields={
            "cases": {
                case.case_id: {
                    "cup_in_base": results[case.case_id].t,
                    "description": case.description,
                }
                for case in cases
            },
            "reference_id": reference_id,
        },
    )
