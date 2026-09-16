"""@file panda.py
@brief Franka Panda 的 FK / IK / 雅可比 —— 对应课件 08~10 三个脚本。

课件这三个脚本本来就是"调用库"（robot.fkine / ikine_LM / jacob0），
本项目改成调库之后，这一节和课件的口径就完全一致了。

【本用例真正要处理的是"约定"，不是数学】：
  * 末端帧：`Panda()` 自带夹爪，`fkine(q)` 默认返回**夹爪 TCP**；
    要法兰必须显式 `end="panda_link8"` —— 这就是那个著名的 0.1034 m 差异的来源；
  * RPY 约定：必须 `order='zyx'` 才等于课件手写的 `Rz·Ry·Rx`（默认的 `'xyz'` 是另一套）。
"""

from __future__ import annotations

import numpy as np
import roboticstoolbox as rtb
from spatialmath import SE3

from ..contracts import (
    HeadingBlock,
    MatrixBlock,
    Report,
    TableBlock,
    TextBlock,
    VectorBlock,
    fmt_condition,
    fmt_vector,
)
from ..core import robots
from ..core.singularity import (
    SingularityReport,
    amplification,
    analysis_of,
    describe_worst_direction,
    manipulability,
)

# 本项目采用的 RPY 约定：R = Rz(yaw)·Ry(pitch)·Rx(roll)
RPY_ORDER = "zyx"
DEG = np.deg2rad

# 课件 data/panda_cases.csv 里的三个姿态
Q_ZERO = np.zeros(7)
Q_PPT_GOAL = np.array([0.0, -0.4, 0.0, -2.2, 0.0, 2.0, 0.7853981634])
Q_IK_SEED = np.array([0.0, -0.3, 0.0, -2.2, 0.0, 2.0, 0.8])

# 课件 run_logs/08_franka_panda_fk.log 记录的零位姿（**夹爪 TCP 帧**）
COURSE_LOG_ZERO_TCP = (0.088, 0.0, 0.8226)


def fk_report(robot: rtb.Robot, q: np.ndarray, frame: str = robots.FLANGE) -> Report:
    q = np.asarray(q, dtype=float).reshape(7)
    T = robots.end_pose(robot, q, frame)
    limits = robots.joint_limits(robot)
    over = robots.out_of_limits(q, limits)
    tcp_pose = robots.end_pose(robot, q, robots.TCP)

    blocks: list = [
        TextBlock.of(
            f"模型：{robot.name}",
            f"q = {fmt_vector(q)} rad，即 {fmt_vector(np.rad2deg(q), 1)} deg",
            f"关节数 n = {robot.n}",
        ),
        MatrixBlock("末端位姿 ^base T_ee", T.A),
        VectorBlock("位置 xyz", T.t, suffix=" m"),
        MatrixBlock("旋转矩阵 R", T.R),
        TextBlock.of(
            f"RPY（order={RPY_ORDER}，即 R = Rz·Ry·Rx）= "
            f"{fmt_vector(np.rad2deg(T.rpy(order=RPY_ORDER)), 3)} deg",
            "",
            "模型自检：",
            f"  · 关节限位来自库的 `robot.qlim`，共 {len(limits) if limits is not None else 0} 个关节",
            f"  · 雅可比形状 {np.asarray(robot.jacob0(q)).shape}（6×n：3 线速度 + 3 角速度）",
        ),
    ]

    if over:
        names = ", ".join(robots.joint_names(robot)[i] for i in over)
        blocks.append(
            TextBlock.of(
                f"⚠️ 关节限位提示：{names} 超出 Franka 数据表的限位。",
                "   数学上 FK 对任何 q 都算得出来，但真机摆不出这个姿态 ——",
                "   课件的“全零姿态”就属于这种参考位形（q4 的限位区间是 [-3.0718, -0.0698]）。",
            )
        )

    if frame == robots.FLANGE and np.allclose(q, Q_ZERO):
        blocks.append(
            TextBlock.of(
                "对照课件 run_logs/08_franka_panda_fk.log 的零位姿记录：",
                f"  课件记录（夹爪 TCP 帧）t = {fmt_vector(COURSE_LOG_ZERO_TCP)}",
                f"  本项目的夹爪帧       t = {fmt_vector(tcp_pose.t)}",
                f"  本项目的法兰帧       t = {fmt_vector(T.t)}",
                "",
                "★ 同一台机器人、同一组关节角，两个数只差“末端在哪”的约定：",
                "  库的 `fkine(q)` 默认返回**夹爪 TCP**（法兰 + 103.4mm + 绕 z 转 45°），",
                '  要法兰必须显式 `end="panda_link8"`。课件日志用的是默认末端，所以是 0.8226。',
                "  两边都没错 —— 对不上时第一件事是问“末端是哪个坐标系”。",
                "  想直接看夹爪帧：`rkin panda-fk --case zero --frame tcp`。",
            )
        )

    return Report(
        title="Franka Panda 正运动学（7 个关节角 → 末端位姿）",
        blocks=tuple(blocks),
        fields={
            "model": robot.name,
            "frame": frame,
            "q": q,
            "q_deg": np.rad2deg(q),
            "A": T.A,
            "t": np.asarray(T.t, dtype=float),
            "R": T.R,
            "rpy_deg": np.rad2deg(T.rpy(order=RPY_ORDER)),
            "out_of_limits": over,
            "tcp_t": np.asarray(tcp_pose.t, dtype=float),
        },
    )


def ik_report(
    robot: rtb.Robot,
    target: SE3,
    seed: np.ndarray,
    max_iter: int = 200,
    tol: float = 1e-9,
    show_redundancy: bool = True,
    frame: str = robots.FLANGE,
) -> Report:
    """数值 IK：直接调库的 `ikine_LM`（Levenberg-Marquardt）。"""
    seed = np.asarray(seed, dtype=float).reshape(7)
    solution = _solve(robot, target, seed, frame=frame, max_iter=max_iter, tol=tol)
    q = np.asarray(solution.q, dtype=float)
    reached = robots.end_pose(robot, q, frame)

    blocks: list = [
        MatrixBlock("目标位姿 ^base T_ee", target.A),
        TextBlock.of(
            f"初值 q0 = {fmt_vector(seed)}",
            f"求解器：robot.ikine_LM（阻尼最小二乘 / LM 迭代），max_iter={max_iter}，tol={tol:.0e}",
            "",
            f"是否收敛：{bool(solution.success)}"
            + (f"（迭代 {solution.iterations} 次）" if hasattr(solution, "iterations") else ""),
        ),
        VectorBlock("求得 q", q, suffix=" rad"),
        VectorBlock("      =", np.rad2deg(q), digits=2, suffix=" deg"),
        TextBlock.of(
            "FK 回验（把解出的 q 送回 FK，看是否回到目标位姿）",
            f"  位置误差   {np.linalg.norm(target.t - reached.t):.3e} m",
            f"  姿态误差   {np.linalg.norm(target.rpy(order=RPY_ORDER) - reached.rpy(order=RPY_ORDER)):.3e} rad",
            f"  位置 xyz = {fmt_vector(reached.t)}",
            "⚠️ 数值 IK 不保证成功：初值太远、目标不可达、或者正处在奇异点附近都可能不收敛。",
        ),
    ]

    redundancy_blocks: tuple = ()
    results: list = []
    if show_redundancy:
        seeds = [seed, *[_random_seed(robot, i) for i in range(3)]]
        results = [_solve(robot, target, s, frame=frame, max_iter=max_iter, tol=tol) for s in seeds]
        rows = tuple(
            (
                f"初值 {i + 1}",
                fmt_vector(np.asarray(r.q)[:3], 2) + " …",
                "✔" if r.success else "✘",
                f"{np.linalg.norm(target.t - robots.end_pose(robot, np.asarray(r.q), frame).t):.2e}",
            )
            for i, r in enumerate(results)
        )
        redundancy_blocks = (
            HeadingBlock("7 自由度为什么有无穷多组解（冗余）"),
            TextBlock.of("同一个末端位姿，可以从不同初值收敛到不同的关节角配置："),
            TableBlock(
                ("起点", "解出的 q（前 3 个关节）", "收敛", "位置误差"), rows, ("<", "<", "^", ">")
            ),
            TextBlock.of(
                "多出来的那一个自由度就是“零空间”：不动末端，也能改变关节姿态。",
                "工程上用它来避障、远离关节限位、保持姿态舒服（抓取流水线的解择优就用了这一点）。",
            ),
        )

    title = "Franka Panda 逆运动学（目标位姿 → 关节角）"
    if show_redundancy:
        title += "与冗余"
    return Report(
        title=title,
        blocks=(*blocks, *redundancy_blocks),
        fields={
            "target": target.A,
            "seed": seed,
            "success": bool(solution.success),
            "iterations": int(getattr(solution, "iterations", -1)),
            "q": q,
            "q_deg": np.rad2deg(q),
            "position_error": float(np.linalg.norm(target.t - reached.t)),
            "redundancy": [
                {"seed_index": i + 1, "q_first3": np.asarray(r.q)[:3], "success": bool(r.success)}
                for i, r in enumerate(results)
            ],
        },
    )


def jacobian_report(robot: rtb.Robot, q: np.ndarray) -> Report:
    q = np.asarray(q, dtype=float).reshape(7)
    J = np.asarray(robot.jacob0(q), dtype=float)
    report = analysis_of(robot, q)

    return Report(
        title="Franka Panda 几何雅可比（6×7）",
        blocks=(
            VectorBlock("q", q, suffix=" rad"),
            MatrixBlock("J =", J),
            TextBlock.of(
                "前 3 行：关节角速度对末端线速度的影响；后 3 行：对末端角速度的影响。",
                "（由库的 `robot.jacob0(q)` 给出）",
                "",
                _singularity_lines(robot, q, report),
                "",
                "6×n 而不是 n×n：末端有 6 个自由度（3 平移 + 3 转动），Panda 有 7 个关节，",
                "所以雅可比是 6 行 7 列 —— 列比行多，正是“冗余”的数学表现。",
            ),
        ),
        fields={
            "q": q,
            "jacobian": J,
            "sigma_max": report.sigma_max,
            "sigma_min": report.sigma_min,
            "condition": report.condition,
            "manipulability": report.manipulability,
            "manipulability_library": manipulability(robot, q),
            "is_singular": report.is_singular,
            "is_near_singular": report.is_near_singular,
            "worst_direction": report.worst_direction,
        },
    )


def singular_pose_report(robot: rtb.Robot) -> Report:
    """对比几个典型姿态的局部运动能力，落盘 outputs/。"""
    poses = (
        ("zero（课件姿态 1）", Q_ZERO),
        ("ppt_goal（课件姿态 2）", Q_PPT_GOAL),
        ("ik_seed（课件初值）", Q_IK_SEED),
    )
    rows, entries = [], []
    for name, q in poses:
        report = analysis_of(robot, q)
        state = (
            "奇异" if report.is_singular else ("接近奇异" if report.is_near_singular else "正常")
        )
        rows.append(
            (
                name,
                f"{report.sigma_min:.3e}",
                fmt_condition(report.condition),
                f"{report.manipulability:.3e}",
                state,
            )
        )
        entries.append(
            {
                "name": name,
                "sigma_min": report.sigma_min,
                "condition": report.condition,
                "manipulability": report.manipulability,
                "is_singular": report.is_singular,
            }
        )

    return Report(
        title="Panda 奇异姿态对比",
        blocks=(
            TextBlock.of("（雅可比与可操作度均来自库：robot.jacob0 / robot.manipulability）"),
            TableBlock(
                ("姿态", "σ_min", "cond(J)", "可操作度", "判定"),
                tuple(rows),
                ("<", ">", ">", ">", "^"),
            ),
            TextBlock.of(
                "可操作度 = σ₁σ₂…σᵣ = √det(J·Jᵀ)：= 0 表示至少一个方向彻底动不了。",
                "课件里的“全零姿态”就是奇异的 —— 手臂竖直、腕部对齐，这是它的物理含义。",
            ),
        ),
        output_name="panda_singularity_summary.txt",
        fields={"poses": entries},
    )


# ── 内部工具 ────────────────────────────────────────────────────────────────


def _solve(
    robot: rtb.Robot, target: SE3, seed: np.ndarray, *, frame: str, max_iter: int, tol: float
):
    """调库求解，并把末端帧与迭代参数传下去。

    ⚠️ `ikine_LM` 的 `end=` 参数决定它把哪个坐标系当末端 —— 不给就会用默认的夹爪 TCP，
       解出来的关节角与我们的法兰帧目标对不上。
    """
    kwargs = {"q0": np.asarray(seed, dtype=float).reshape(-1), "ilimit": max_iter, "tol": tol}
    end = robots.panda_end(frame)
    if end is not None:
        kwargs["end"] = end
    return robot.ikine_LM(target, **kwargs)


def _random_seed(robot: rtb.Robot, index: int) -> np.ndarray:
    rng = np.random.default_rng(index + 1)
    return rng.uniform(-1.5, 1.5, size=robot.n)


def _singularity_lines(robot: rtb.Robot, q: np.ndarray, report: SingularityReport) -> str:
    state = (
        "奇异（至少有一个方向彻底动不了）"
        if report.is_singular
        else ("接近奇异" if report.is_near_singular else "正常位姿")
    )
    return "\n".join(
        [
            f"σ_max = {report.sigma_max:.4e}",
            f"σ_min = {report.sigma_min:.4e}   ← 最坏方向放大倍数 1/σ_min = {amplification(report):.4e}",
            f"条件数 cond(J) = {report.condition:.4f}",
            f"可操作度 = {report.manipulability:.4e}"
            f"（库的 manipulability 给 {manipulability(robot, q):.4e}，两者一致）",
            f"判定：{state}",
            f"最难运动的方向：{describe_worst_direction(report.worst_direction)}",
        ]
    )
