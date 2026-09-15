"""@file panda.py
@brief Franka Panda 的 FK / IK / 雅可比 —— 对应课件 08~10 三个脚本。

课件这三个脚本是"调用库"（robot.fkine / ikine_LM / jacob0）；本项目换成自研实现，
所以报告里打出来的每个数都能在 core/ 里找到来源。

⚠️ 末端帧由调用方决定：传法兰链就是法兰帧，传 panda_hand_tcp_chain() 就是夹爪 TCP 帧。
   这个用例不认识"frame 参数"，它只管把拿到的链算清楚 —— 免得出现"传了链又被换掉"的怪事。
"""

from __future__ import annotations

import numpy as np

from ..contracts import (
    MatrixBlock,
    Report,
    TableBlock,
    TextBlock,
    VectorBlock,
    fmt_condition,
    fmt_vector,
)
from ..core import panda
from ..core.chain import SerialChain
from ..core.ik_solvers import nullspace_improvement_bound, random_seeds, solve_dls, solve_multi_seed
from ..core.rotations import matrix_to_rpy
from ..core.se3 import SE3
from ..core.singularity import analyze, describe_worst_direction

DEG = np.deg2rad


def fk_report(chain: SerialChain, q: np.ndarray) -> Report:
    q = np.asarray(q, dtype=float).reshape(7)
    T = chain.fk(q)
    rpy = matrix_to_rpy(T.R)
    out_of_limits = panda.joints_out_of_limits(q)
    dh_deviation = float(np.abs(chain.fk(q).A - panda.panda_dh_robot().fk(q).A).max())

    blocks: list = [
        TextBlock.of(
            f"模型：{chain.name}",
            f"q = {fmt_vector(q)} rad，即 {fmt_vector(np.rad2deg(q), 1)} deg",
            f"关节数 n = {chain.n_joints}（课件：7 自由度协作机械臂）",
        ),
        MatrixBlock("末端位姿 ^base T_ee", T.A),
        VectorBlock("位置 xyz", T.t, suffix=" m"),
        MatrixBlock("旋转矩阵 R（每一个元素都是自研 SE3 算出来的，不依赖任何机器人库）", T.R),
        TextBlock.of(
            f"RPY（xyz 顺序，本项目约定 R = Rz·Ry·Rx）= {fmt_vector(np.rad2deg(rpy), 3)} deg",
            "",
            "模型自检：",
            f"  · ETS（URDF 写法）与 MDH（教材写法）两套参数的 FK 结果最大偏差 = {dh_deviation:.3e}",
            "  · 几何雅可比与数值微分的最大偏差（见 `rkin panda-jac`）= 约 1e-9 量级",
        ),
    ]

    if out_of_limits:
        blocks.append(
            TextBlock.of(
                f"⚠️ 关节限位提示：{', '.join(out_of_limits)} 超出 Franka 数据表给的限位。",
                "   数学上 FK 对任何 q 都算得出来，但真机摆不出这个姿态 ——",
                "   课件的“全零姿态”就属于这种参考位形（q4 的限位区间是 [-3.0718, -0.0698]）。",
            )
        )

    if np.allclose(q, panda.PANDA_Q_ZERO):
        hand = panda.panda_hand_tcp_chain().fk(q)
        blocks.append(
            TextBlock.of(
                "对照课件 run_logs/08_franka_panda_fk.log 的零位姿记录：",
                f"  课件记录（夹爪 TCP 帧）t = {fmt_vector(panda.COURSE_LOG_ZERO_POSE_TOOL_T)}",
                f"  本项目加夹爪后        t = {fmt_vector(hand.t)}",
                f"  本项目法兰帧          t = {fmt_vector(T.t)}",
                "",
                "★ 三个数只差一个“末端在哪”的约定：",
                "  法兰（panda_link8）→ 夹爪 TCP 需要再往前 103.4mm 并绕 z 转 -45°",
                "  （Franka 官方夹爪 TCP 约定，也就是 roboticstoolbox 里那个 panda_hand）。",
                "  课件的 robot.fkine(q) 默认返回夹爪 TCP，所以是 0.8226；",
                "  本项目默认给法兰帧，所以是 0.9260 —— 加上夹爪偏移后与课件逐位吻合。",
                "  想直接看夹爪帧：`rkin panda-fk --case zero --frame tcp`。",
            )
        )

    return Report(
        title="Franka Panda 正运动学（7 个关节角 → 末端位姿）",
        blocks=tuple(blocks),
        fields={
            "chain": chain.name,
            "q": q,
            "q_deg": np.rad2deg(q),
            "A": T.A,
            "t": T.t,
            "R": T.R,
            "rpy_deg": np.rad2deg(rpy),
            "out_of_limits": out_of_limits,
            "ets_mdh_deviation": dh_deviation,
        },
    )


def ik_report(
    chain: SerialChain,
    target: SE3,
    seed: np.ndarray,
    dls_lambda: float = 0.05,
    max_iter: int = 200,
    tol: float = 1e-9,
    show_redundancy: bool = True,
    prefer_config: np.ndarray | None = None,
) -> Report:
    seed = np.asarray(seed, dtype=float).reshape(7)
    result = solve_dls(chain, target, seed, dls_lambda=dls_lambda, max_iter=max_iter, tol=tol)

    # 零空间次要任务：7 自由度下同一个位姿有无穷多组解，用多出来的自由度
    # 把关节角拉向参考姿态（离限位更远、姿态更舒服），而不改变末端位姿。
    guided = None
    q_ref: np.ndarray | None = None
    if prefer_config is not None:
        q_ref = np.asarray(prefer_config, dtype=float).reshape(7)
        guided = solve_dls(
            chain,
            target,
            seed,
            dls_lambda=dls_lambda,
            max_iter=max_iter,
            tol=tol,
            q_ref=q_ref,
            nullspace_gain=0.5,
        )
        result = guided  # 报告里给出的解 = 加了次要任务的那个

    blocks: list = [
        MatrixBlock("目标位姿 ^base T_ee", target.A),
        TextBlock.of(
            f"初值 q0 = {fmt_vector(seed)}",
            f"阻尼系数 λ = {dls_lambda}，最大迭代 {max_iter}，收敛判据 {tol:.1e}",
            "",
            result.summary(),
        ),
        VectorBlock("求得 q", result.q, suffix=" rad"),
        VectorBlock("      =", np.rad2deg(result.q), digits=2, suffix=" deg"),
        TextBlock.of(
            "FK 回验（把解出的 q 送回 FK，看是否回到目标位姿）",
            f"  位置误差   {result.position_error:.3e} m",
            f"  姿态误差   {result.orientation_error:.3e} rad",
            f"  位置 xyz = {fmt_vector(chain.fk(result.q).t)}",
            "⚠️ 数值 IK 不保证成功：初值太远、目标不可达、或者正处在奇异点附近都可能不收敛。",
            "   这里能解出来，是因为目标本身就是用 FK 从某个 q 造出来的（一定可达）。",
        ),
    ]

    if guided is not None and q_ref is not None:
        plain = solve_dls(chain, target, seed, dls_lambda=dls_lambda, max_iter=max_iter, tol=tol)
        plain_distance = float(np.linalg.norm(plain.q - q_ref))
        guided_distance = float(np.linalg.norm(guided.q - q_ref))
        movable, bound = nullspace_improvement_bound(chain, plain.q, q_ref)
        blocks += [
            TextBlock.of("", f"参考姿态 q_ref = {fmt_vector(q_ref)}"),
            TableBlock(
                ("", "与 q_ref 的关节角距离", "末端位置误差", "姿态误差"),
                (
                    (
                        "不加次要任务",
                        f"{plain_distance:.4f} rad",
                        f"{plain.position_error:.2e} m",
                        f"{plain.orientation_error:.2e} rad",
                    ),
                    (
                        "加了次要任务",
                        f"{guided_distance:.4f} rad",
                        f"{guided.position_error:.2e} m",
                        f"{guided.orientation_error:.2e} rad",
                    ),
                ),
                ("<", ">", ">", ">"),
            ),
            TextBlock.of(
                f"实际缩短 {plain_distance - guided_distance:.5f} rad；"
                f"按零空间方向估算的上限 {bound:.5f} rad。",
                f"（能动分量 |N·(q_ref−q)| = {movable:.4f} rad —— 注意它不是缩短量，见下）",
                "",
                "★ 为什么这里几乎没省下距离：7 个关节对 6 维任务，零空间只有 1 维。",
                "  偏好里只有落在这一维上的那部分能动（上行的“能动分量”）；",
                "  而距离能缩短多少，取决于这点分量与整条差值向量的夹角 ——",
                "  两者接近正交时，能动分量再大也几乎不缩短距离（本例就是如此）。",
                "  所以“用冗余自由度满足次要偏好”不是万能的：先看这两个数，再决定值不值得做。",
            ),
        ]

    redundancy_blocks: tuple = ()
    rows: list = []
    results: list = []
    if show_redundancy:
        seeds = [seed, *random_seeds(chain, 3, seed=11, scale=1.2)]
        results = solve_multi_seed(
            chain, target, seeds, dls_lambda=dls_lambda, max_iter=max_iter, tol=tol
        )
        rows = [
            (
                f"初值 {index}",
                fmt_vector(res.q[:3], 2) + " …",
                "✔" if res.success else "✘",
                f"{res.residual:.2e}",
            )
            for index, res in enumerate(results, start=1)
        ]
        redundancy_blocks = (
            TextBlock.of(
                "同一个末端位姿，可以从不同初值收敛到不同的关节角配置：",
            ),
            TableBlock(
                ("起点", "解出的 q（前 3 个关节）", "收敛", "残差"),
                tuple(rows),
                ("<", "<", "^", ">"),
            ),
            TextBlock.of(
                "多出来的那一个自由度就是“零空间”：不动末端，也能改变关节姿态。",
                "工程上用它来避障、远离关节限位、保持姿态舒服（`--prefer-config` 就是这个用途）。",
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
            "success": result.success,
            "iterations": result.iterations,
            "q": result.q,
            "q_deg": np.rad2deg(result.q),
            "position_error": result.position_error,
            "orientation_error": result.orientation_error,
            "residual": result.residual,
            "prefer_config": None if prefer_config is None else prefer_config,
            "redundancy": [
                {
                    "seed_index": i + 1,
                    "q_first3": r.q[:3],
                    "success": r.success,
                    "residual": r.residual,
                }
                for i, r in enumerate(results)
            ],
        },
    )


def jacobian_report(
    chain: SerialChain, q: np.ndarray, det_eps: float = 1e-9, cond_warn: float = 100.0
) -> Report:
    q = np.asarray(q, dtype=float).reshape(7)
    J = chain.jacobian(q)
    report = analyze(J, det_eps=det_eps, cond_warn=cond_warn)
    amplification = (1.0 / report.sigma_min) if report.sigma_min > 0 else float("inf")
    state = (
        "奇异（至少有一个方向彻底动不了）"
        if report.is_singular
        else ("接近奇异" if report.is_near_singular else "正常位姿")
    )

    return Report(
        title="Franka Panda 几何雅可比（6×7）",
        blocks=(
            VectorBlock("q", q, suffix=" rad"),
            MatrixBlock("J =", J),
            TextBlock.of(
                "前 3 行：关节角速度对末端线速度的影响；后 3 行：对末端角速度的影响。",
                "第 7 列只有角速度、没有线速度：工具坐标系就挂在第 7 关节的轴上。",
                "",
                f"σ_max = {report.sigma_max:.4e}",
                f"σ_min = {report.sigma_min:.4e}   ← 最坏方向上的放大倍数 = 1/σ_min = {amplification:.4e}",
                f"条件数 cond(J) = {report.condition:.4f}",
                f"可操作度 = {report.manipulability:.4e}",
                f"判定：{state}",
                f"最难运动的方向：{describe_worst_direction(report.worst_direction)}",
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
            "is_singular": report.is_singular,
            "is_near_singular": report.is_near_singular,
            "worst_direction": report.worst_direction,
        },
    )


def singular_pose_report(chain: SerialChain, det_eps: float = 1e-9) -> Report:
    """对比几个典型姿态的局部运动能力，落盘 outputs/。"""
    poses = (
        ("zero（课件姿态 1）", panda.PANDA_Q_ZERO),
        ("ppt_goal（课件姿态 2）", panda.PANDA_Q_PPT_GOAL),
        ("ik_seed（课件初值）", panda.PANDA_Q_IK_SEED),
    )
    rows, entries = [], []
    for name, q in poses:
        report = analyze(chain.jacobian(q), det_eps=det_eps)
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
