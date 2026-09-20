"""@file planar.py

@brief 二连杆的 FK / IK / 雅可比 / 奇异点 —— 对应课件 03~07 五个脚本。

【本用例的分工】二连杆有闭式解，而库只给数值解（见 core/planar2r.py 的说明），所以：
  * FK / IK（含"肘上/肘下"两组解）= 手推公式（core/planar2r.py）
  * 雅可比 = 库的 `robot.jacob0(q)`；行列式仍用闭式 `L1·L2·sin(q2)` 与库对照
  * 奇异点体检 = core/singularity.py（numpy SVD）
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
from ..core import robots
from ..core.exceptions import UnreachableTargetError
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
    """@brief 正运动学报告：关节角 → 三个关节点，并顺手和库的 FK 对一遍。

    @param arm 二连杆几何（L1、L2）
    @param q1_deg 关节 1 角度 [°]
    @param q2_deg 关节 2 角度 [°]
    @param chart 是否附一张字符画（终端里也能看出姿态）
    @return 报告；fields["library_gap"] 是闭式解与库 FK 的偏差 —— 两个独立来源互为验证
    """
    q1, q2 = DEG(q1_deg), DEG(q2_deg)
    base, elbow, tool = arm.joint_points(q1, q2)
    library_fk = np.asarray(robots.planar(arm.l1, arm.l2).fkine([q1, q2]).t)[:2]

    blocks: list = [
        TextBlock.of(
            f"输入：q1={q1_deg:.1f}°，q2={q2_deg:.1f}°（L1={arm.l1:g}, L2={arm.l2:g}）",
            "",
            "闭式解（手推）:  x = L1·cos(q1) + L2·cos(q1+q2)",
            "                y = L1·sin(q1) + L2·sin(q1+q2)",
            "",
            f"base  = {fmt_vector(base)}",
            f"elbow = {fmt_vector(elbow)}   ← 第一段由 q1 决定方向",
            f"tool  = {fmt_vector(tool)}   ← 第二段由 q1+q2 决定方向",
            "",
            "装配关系：base → joint1(q1) → link1(L1) → joint2(q2) → link2(L2) → tool",
            f"末端到 base 的距离 = {np.linalg.norm(tool):.4f} m"
            f"（可达范围 [{abs(arm.l1 - arm.l2):.4f}, {arm.l1 + arm.l2:.4f}]）",
            f"与库 FK 的偏差 = {np.abs(library_fk - tool).max():.3e}（闭式解与库互为验证）",
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
            "library_gap": float(np.abs(library_fk - tool).max()),
        },
    )


def ik_report(arm: Planar2R, x: float, y: float) -> Report:
    """@brief 逆运动学报告：目标点 → **全部**关节解（肘上 / 肘下），再逐组回代 FK 验算。

    @param arm 二连杆几何
    @param x 目标点 x [m]
    @param y 目标点 y [m]
    @return 报告；fields["solutions_deg"] 按肘上、肘下的顺序给出两组解
    @throws UnreachableTargetError 目标落在 [|L1-L2|, L1+L2] 之外 —— 物理上够不着，不是 bug
    """
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
                "闭式解（余弦定理）：cos(q2) = (x² + y² - L1² - L2²) / (2·L1·L2)",
                "                   q2 = ±arccos(cos(q2))    ← ± 号对应两组解",
                "                   q1 = atan2(y, x) - atan2(L2·sin(q2), L1 + L2·cos(q2))",
                "",
                "⚠️ 这一步**必须**用手推公式：库的 `ikine_LM` 只返回一组解，",
                "   而“肘上/肘下两组解”正是本节的考点。",
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
    """@brief 雅可比报告：末端想微调 dx，关节该动多少 —— 并在奇异位姿上如实报告解不出来。

    矩阵由库的 `jacob0` 给出，行列式同时用闭式 L1·L2·sin(q2) 对照（两者应一致）。
    非奇异时解 J·dq = dx 给出 dq；奇异时不硬解，只说明"这是能力丧失，不是程序错误"。

    @param arm 二连杆几何
    @param q1_deg 当前关节 1 角度 [°]
    @param q2_deg 当前关节 2 角度 [°]
    @param dx 期望的末端小位移 (dx, dy) [m]
    @param det_eps σ_min 低于它就判为奇异
    @param cond_warn 条件数高于它就判为"接近奇异"
    @return 报告；fields["dq_deg"] 在奇异位姿下为 None
    """
    q1, q2 = DEG(q1_deg), DEG(q2_deg)
    robot = robots.planar(arm.l1, arm.l2)
    J = np.asarray(robot.jacob0([q1, q2]))[:2, :]  # 库的几何雅可比，取平面两行
    report = analyze(J, det_eps=det_eps, cond_warn=cond_warn)
    det_closed_form = arm.det_jacobian(q2)
    det_library = float(np.linalg.det(J))

    blocks: list = [
        TextBlock.of(
            f"当前 q1={q1_deg:.1f}°，q2={q2_deg:.1f}°",
            "",
            "J = ∂(末端位置)/∂(关节角)   —— 由库的 robot.jacob0(q) 给出",
        ),
        MatrixBlock("", J),
        TextBlock.of(
            f"期望末端小位移 dx = {fmt_vector(dx)}",
            f"det(J)：闭式 L1·L2·sin(q2) = {det_closed_form:.6f}，"
            f"库算的 det = {det_library:.6f}（两者应一致）",
            report.summary(),
            f"最难运动的方向：{np.round(report.worst_direction, 4)}",
        ),
    ]

    dq = None
    if report.is_singular:
        blocks.append(
            TextBlock.of(
                "",
                f"⚠️ 当前位姿接近奇异（q2={q2_deg:.1f}°，det(J)={det_closed_form:.3e}）："
                "末端在该方向上的瞬时运动能力丧失，解不出可用的关节微调量。",
                "   这是物理限制，不是程序错误 —— FK 照样算得出来，只是“往某个方向再挪一点”做不到。",
            )
        )
    else:
        dq = np.linalg.solve(J, np.array(dx, dtype=float))  # 解线性方程组用 numpy
        blocks.append(
            TextBlock.of(
                "",
                f"解得关节微调量 dq = {fmt_vector(dq)} rad = {fmt_vector(np.rad2deg(dq))} deg",
                f"用 J 反推末端位移 J·dq = {fmt_vector(J @ dq)}（应等于 dx）",
                "⚠️ J 是当前姿态下的局部线性近似：步长一大就不准了，所以它只适合“小步微调”。",
            )
        )

    return Report(
        title="二连杆雅可比（末端想微调，关节该怎么动）",
        blocks=tuple(blocks),
        fields={
            "q_deg": [q1_deg, q2_deg],
            "jacobian": J,
            "dx": list(dx),
            "det_jacobian": det_closed_form,
            "det_library": det_library,
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
    robot = robots.planar(arm.l1, arm.l2)
    rows, sweep_fields = [], []
    for q1_deg, q2_deg, name in SWEEP_POSES:
        q1, q2 = DEG(q1_deg), DEG(q2_deg)
        tool = arm.fk(q1, q2)
        J = np.asarray(robot.jacob0([q1, q2]))[:2, :]
        report = analyze(J, det_eps=det_eps)
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
                "（雅可比由库的 jacob0 给出，行列式与闭式公式对照 —— 两者应一致）",
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
