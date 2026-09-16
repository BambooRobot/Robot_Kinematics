"""@file batch.py
@brief 批量跑完算例集 —— 对应课件 11_ppt_cases_batch.py，并补上它的缺口。

课件版本只读了 coordinate_cases.csv 和 two_link_cases.csv，把 panda_cases.csv 落在一边
（那个文件声明了却没有任何脚本读它）。本项目把 Panda 段补上。

⚠️ 前三段的**文本格式**是刻意与课件保持一致的：课件交付物
   `outputs/ppt_cases_batch_result.txt` 是这门课的验收材料，本项目要能逐字符对上它
   （见 docs/NUMBERS.md）。所以这里保留了课件那套 `- case_id: ...` 的写法。
"""

from __future__ import annotations

import numpy as np

from ..contracts import CaseSource, Report, TextBlock, fmt_condition
from ..core import robots
from ..core.planar2r import Planar2R
from ..core.singularity import analysis_of

DEG = np.deg2rad
OUTPUT_NAME = "ppt_cases_batch_result.txt"


def run(source: CaseSource) -> Report:
    lines: list[str] = []

    lines.append("1) 坐标变换：camera frame -> base frame")
    for case in source.coordinate_cases():
        base = _coordinate_case(case)
        lines.append(f"- {case.case_id}: cup_base={np.round(base, 4)}")

    lines.append("")
    lines.append("2) 二连杆 FK / IK / 奇异点")
    for case in source.two_link_cases():
        lines.append(f"- {case.case_id}: {_two_link_case(case)}")

    lines.append("")
    lines.append("3) Franka Panda（课件未纳入这一段，本项目补上）")
    robot = robots.panda(robots.TCP)
    for case in source.panda_cases():
        lines.append(f"- {case.case_id}: {_panda_case(robot, case)}")

    return Report(
        title="批量算例报告（第 1、2 段与课件输出逐字符一致）",
        blocks=(TextBlock(tuple(lines)),),
        output_name=OUTPUT_NAME,
        fields={"lines": list(lines)},
    )


def _coordinate_case(case) -> np.ndarray:
    from spatialmath import SE3

    T_base_camera = SE3.Trans(*case.camera_xyz) * SE3.Rz(DEG(case.camera_yaw_deg))
    return np.asarray((T_base_camera * SE3.Trans(*case.cup_xyz)).t, dtype=float)


def _two_link_case(case) -> str:
    arm = Planar2R(l1=case.l1, l2=case.l2)
    if case.kind == "fk":
        q1, q2 = (DEG(value) for value in case.q_deg)
        tool = arm.fk(q1, q2)
        return f"FK tool={np.round(tool, 4)}, detJ={arm.det_jacobian(q2):.6f}"

    solutions = arm.ik(*case.target)
    if not solutions:
        return "IK no solution"
    text = "; ".join(f"({np.rad2deg(q1):.2f}°, {np.rad2deg(q2):.2f}°)" for q1, q2 in solutions)
    return f"IK sols={text}"


def _panda_case(robot, case) -> str:
    # 位置与雅可比都用**夹爪 TCP 帧**（= 库的默认末端）：两者必须同帧，否则数字没法对照。
    # 课件的 run_logs 记的也是这一帧。
    tool = robots.end_pose(robot, case.q, robots.TCP)
    report = analysis_of(robot, case.q)
    state = "奇异" if report.is_singular else ("接近奇异" if report.is_near_singular else "正常")
    return (
        f"tool={np.round(tool.t, 4)}, σmin={report.sigma_min:.3e}, "
        f"cond={fmt_condition(report.condition)}, {state}"
    )
