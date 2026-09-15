"""@file planar.py
@brief 二连杆的 FK / IK / 雅可比 / 奇异点 —— 对应课件 03~07 五个脚本。

课件把它们拆成五个脚本（各自简单，但同一个公式抄了多份）；本项目把数学收在
core/planar2r.py 一份实现里，这里只负责"报什么"。
"""

from __future__ import annotations

import numpy as np

from ..contracts import (
    MatrixBlock,
    Report,
    TableBlock,
    TextBlock,
    TwoLinkChartBlock,
    fmt_condition,
    fmt_vector,
)
from ..core.exceptions import SingularPoseError, UnreachableTargetError
from ..core.planar2r import Planar2R
from ..core.singularity import analyze

DEG = np.deg2rad

# 课件 07 扫描的四个姿态：(q1, q2, 名称)
SWEEP_POSES = [
    (0.0, 90.0, "普通姿态"),
    (0.0, 30.0, "开始接近伸直"),
    (0.0, 10.0, "接近奇异"),
    (0.0, 0.0, "完全伸直/奇异"),
]


def fk_report(arm: Planar2R, q1_deg: float, q2_deg: float, chart: bool = True) -> Report:
    q1, q2 = DEG(q1_deg), DEG(q2_deg)
    base, elbow, tool = arm.joint_points(q1, q2)

    blocks: list = [
        TextBlock.of(
            f"输入：q1={q1_deg:.1f}°，q2={q2_deg:.1f}°（L1={arm.l1}, L2={arm.l2}）",
            "",
            "x = L1·cos(q1) + L2·cos(q1+q2)",
            "y = L1·sin(q1) + L2·sin(q1+q2)",
            "",
            f"base  = {fmt_vector(base)}",
            f"elbow = {fmt_vector(elbow)}   ← 第一段由 q1 决定方向",
            f"tool  = {fmt_vector(tool)}   ← 第二段由 q1+q2 决定方向",
            "",
            "装配关系：base → joint1(q1) → link1(L1) → joint2(q2) → link2(L2) → tool",
            f"末端到 base 的距离 = {np.linalg.norm(tool):.4f} m"
            f"（可达范围 [{abs(arm.l1 - arm.l2):.4f}, {arm.l1 + arm.l2:.4f}]）",
        )
    ]
    if chart:
        blocks.append(TwoLinkChartBlock(points=(base, elbow, tool)))

    return Report(
        title="二连杆正运动学（关节角 → 末端位置）",
        blocks=tuple(blocks),
        fields={
            "q_deg": [q1_deg, q2_deg],
            "l1": arm.l1,
            "l2": arm.l2,
            "base": base,
            "elbow": elbow,
            "tool": tool,
            "det_jacobian": arm.det_jacobian(q2),
        },
    )


def ik_report(arm: Planar2R, x: float, y: float) -> Report:
    solutions = arm.ik(x, y)
    if not solutions:
        raise UnreachableTargetError(
            f"目标 ({x}, {y}) 超出二连杆的工作空间"
            f"（可达范围 [{abs(arm.l1 - arm.l2):.4f}, {arm.l1 + arm.l2:.4f}]，"
            f"目标距离 {np.hypot(x, y):.4f}）"
        )

    rows = []
    for index, (q1, q2) in enumerate(solutions, start=1):
        tool = arm.fk(q1, q2)
        rows.append(
            (
                f"解 {index}",
                f"{np.rad2deg(q1):.2f}°",
                f"{np.rad2deg(q2):.2f}°",
                fmt_vector(tool),
                "✔" if np.allclose(tool, [x, y], atol=1e-9) else "✘",
            )
        )
    label = "肘上 / 肘下" if len(solutions) == 2 else "（退化为一组）"

    return Report(
        title="二连杆逆运动学（末端位置 → 关节角）",
        blocks=(
            TextBlock.of(
                f"目标点：({x}, {y})",
                "",
                "cos(q2) = (x² + y² - L1² - L2²) / (2·L1·L2)",
                "q2 = ±arccos(cos(q2))          ← ± 号对应两组解",
                "q1 = atan2(y, x) - atan2(L2·sin(q2), L1 + L2·cos(q2))",
            ),
            TableBlock(
                ("解", "q1", "q2", "FK 回验末端", "回到目标"),
                tuple(rows),
                ("<", ">", ">", ">", "^"),
            ),
            TextBlock.of(
                "",
                f"共 {len(solutions)} 组解（{label}）—— 这就是 IK 的多解。",
                "⚠️ IK 求出 q 之后必须送回 FK 验算：不验证就交给机械臂，是最危险的省事。",
            ),
        ),
        fields={
            "target": [x, y],
            "solutions_rad": [[q1, q2] for q1, q2 in solutions],
            "solutions_deg": [
                [float(np.rad2deg(q1)), float(np.rad2deg(q2))] for q1, q2 in solutions
            ],
        },
    )


def jacobian_report(
    arm: Planar2R,
    q1_deg: float,
    q2_deg: float,
    dx: tuple[float, float],
    det_eps: float = 1e-9,
    cond_warn: float = 100.0,
) -> Report:
    q1, q2 = DEG(q1_deg), DEG(q2_deg)
    J = arm.jacobian(q1, q2)
    report = analyze(J, det_eps=det_eps, cond_warn=cond_warn)

    blocks: list = [
        TextBlock.of(
            f"当前 q1={q1_deg:.1f}°，q2={q2_deg:.1f}°",
            "",
            "J = [ ∂x/∂q1  ∂x/∂q2 ]        Δx ≈ J(q)·Δq",
            "    [ ∂y/∂q1  ∂y/∂q2 ]",
        ),
        MatrixBlock("", J),
        TextBlock.of(
            f"期望末端小位移 dx = {fmt_vector(dx)}",
            f"det(J) = {arm.det_jacobian(q2):.6f}   （闭式：L1·L2·sin(q2)）",
            report.summary(),
            f"最难运动的方向：{np.round(report.worst_direction, 4)}",
        ),
    ]

    dq = None
    try:
        dq = arm.solve_step(q1, q2, np.array(dx), det_eps=det_eps)
        blocks.append(
            TextBlock.of(
                "",
                f"解得关节微调量 dq = {fmt_vector(dq)} rad = {fmt_vector(np.rad2deg(dq))} deg",
                f"用 J 反推末端位移 J·dq = {fmt_vector(J @ dq)}（应等于 dx）",
                "⚠️ J 是当前姿态下的局部线性近似：步长一大就不准了，所以它只适合“小步微调”。",
            )
        )
    except SingularPoseError as exc:
        blocks.append(TextBlock.of("", f"⚠️ {exc}"))

    return Report(
        title="二连杆雅可比（末端想微调，关节该怎么动）",
        blocks=tuple(blocks),
        fields={
            "q_deg": [q1_deg, q2_deg],
            "jacobian": J,
            "dx": list(dx),
            "det_jacobian": arm.det_jacobian(q2),
            "condition": report.condition,
            "manipulability": report.manipulability,
            "sigma_min": report.sigma_min,
            "is_singular": report.is_singular,
            "dq_rad": dq,
            "dq_deg": None if dq is None else np.rad2deg(dq),
        },
    )


def singularity_sweep(arm: Planar2R, det_eps: float = 1e-9) -> Report:
    """课件 07 的四姿态扫描表，落盘到 outputs/。"""
    rows = []
    sweep_fields = []
    for q1_deg, q2_deg, name in SWEEP_POSES:
        q1, q2 = DEG(q1_deg), DEG(q2_deg)
        tool = arm.fk(q1, q2)
        report = analyze(arm.jacobian(q1, q2), det_eps=det_eps)
        rows.append(
            (
                name,
                f"{q1_deg:.0f}°",
                f"{q2_deg:.0f}°",
                fmt_vector(tool),
                f"{arm.det_jacobian(q2):.6f}",
                fmt_condition(report.condition),
            )
        )
        sweep_fields.append(
            {
                "name": name,
                "q_deg": [q1_deg, q2_deg],
                "tool": tool,
                "det_jacobian": arm.det_jacobian(q2),
                "condition": report.condition,
                "manipulability": report.manipulability,
                "is_singular": report.is_singular,
            }
        )

    return Report(
        title="奇异点扫描：从普通姿态到完全伸直",
        blocks=(
            TextBlock.of(
                "公式：det(J) = L1 · L2 · sin(q2)。所以 q2 = 0° 或 180° 时两根连杆共线，det(J) = 0。",
            ),
            TableBlock(
                ("姿态", "q1", "q2", "tool(x,y)", "det(J)", "cond(J)"),
                tuple(rows),
                ("<", ">", ">", ">", ">", ">"),
            ),
            TextBlock.of(
                "怎么读这张表：",
                "1. q2 越接近 0°，两根杆越接近共线，det(J) 越接近 0。",
                "2. cond(J) 越大，同样的末端微小移动需要的关节速度越大（放大倍数 = 1/σ_min）。",
                "3. 完全伸直时，末端沿连杆方向继续向外的瞬时运动能力丧失，只能主要沿切向动。",
                "4. 这就是 IK 在奇异点附近容易跳、抖、速度放大，甚至解不出关节微调量的根源。",
            ),
        ),
        output_name="two_link_singularity_summary.txt",
        fields={"sweep": sweep_fields},
    )
