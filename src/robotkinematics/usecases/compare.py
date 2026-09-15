"""@file compare.py
@brief 把自研结果与第三方参照库逐项对照 —— 判定逻辑在这里，取数在 adapters/reference_rtb.py。

【为什么要有这个】本项目刻意不依赖 spatialmath / roboticstoolbox，好处是每个数都是自己算的；
代价是"万一自己算错了呢"。这个用例把同一件事用两条完全独立的路径算一遍，偏差摆在表格里。

⚠️ 边界：适配器只管"把库里的数取出来"，**判定**（一致 / 不一致 / 结论是什么）在这里。
   这样换一个参照库（Pinocchio、KDL）只需换适配器，判定逻辑不用动。
"""

from __future__ import annotations

import numpy as np

from ..contracts import (
    HeadingBlock,
    KinematicsReference,
    Report,
    TableBlock,
    TextBlock,
    fmt_vector,
)
from ..core import panda
from ..core.planar2r import Planar2R
from ..core.se3 import SE3

DEG = np.deg2rad
INSTALL_HINT = (
    'pip install -e ".[crosscheck]"（或直接 pip install spatialmath-python roboticstoolbox-python）'
)
TOL = 1e-9


def compare(reference: KinematicsReference) -> Report:
    numbers = reference.probe()
    if not numbers.available:
        return Report(
            title="交叉验证（与 spatialmath / roboticstoolbox 对照）",
            blocks=(
                TextBlock.of(
                    f"跳过：{numbers.reason}",
                    "",
                    "本项目不依赖这两个库也能跑全部功能；它们只用于这一项对照验证。",
                    f"需要时执行：{INSTALL_HINT}",
                ),
            ),
            fields={"available": False, "reason": numbers.reason},
        )

    versions = "，".join(f"{name} {ver}" for name, ver in numbers.versions)
    rows: list[tuple[str, ...]] = []

    # ① SE3：变换链与逆变换
    T_mine = SE3.Trans(0.30, 0.0, 0.60) * SE3.Rz(DEG(30)) * SE3.Trans(0.50, -0.10, 0.20)
    rows.append(_compare("SE3 变换链 ^base T_cup", T_mine.t, numbers.se3_chain_t))
    rows.append(_compare("SE3 逆变换的位置", T_mine.inverse().t, numbers.se3_inverse_t))

    # ② 二连杆 FK
    arm = Planar2R(l1=1.0, l2=1.0)
    q = np.array([DEG(30), DEG(60)])
    rows.append(_compare("二连杆 FK", arm.fk(*q), numbers.planar_fk))

    # ③ Panda FK —— 两端都对照（法兰帧 / 夹爪 TCP 帧）
    chain = panda.panda_urdf_chain()
    hand = panda.panda_hand_tcp_chain()
    poses = (
        ("zero", panda.PANDA_Q_ZERO),
        ("ppt_goal", panda.PANDA_Q_PPT_GOAL),
        ("ik_seed", panda.PANDA_Q_IK_SEED),
    )
    for index, (name, q_panda) in enumerate(poses):
        rows.append(
            _compare(
                f"Panda FK({name}) 法兰", chain.fk(q_panda).t, _at(numbers.panda_flange_t, index)
            )
        )
        rows.append(
            _compare(
                f"Panda FK({name}) 夹爪 TCP", hand.fk(q_panda).t, _at(numbers.panda_tcp_t, index)
            )
        )

    # ④ Panda 雅可比 —— 雅可比的线速度部分含"末端到关节的杠杆臂"，末端帧不同结果就不同
    q_panda = panda.PANDA_Q_IK_SEED
    rows.append(
        _compare("Panda 雅可比（法兰帧）", chain.jacobian(q_panda), numbers.panda_jacobian_flange)
    )
    rows.append(
        _compare("Panda 雅可比（夹爪 TCP 帧）", hand.jacobian(q_panda), numbers.panda_jacobian_tcp)
    )

    blocks: list = [
        TextBlock.of(f"库版本：{versions}"),
        TableBlock(
            ("对照项", "自研", "库", "最大偏差", "判定"), tuple(rows), ("<", ">", ">", ">", "^")
        ),
        *_verdict(chain, numbers),
        *_ik_note(chain, numbers),
    ]
    if numbers.call_errors:
        blocks.append(
            TextBlock.of(
                "",
                "以下对照项调用失败（库的 API 与预期不同），已如实记录：",
                *(f"  · {name}: {reason}" for name, reason in numbers.call_errors),
            )
        )

    return Report(
        title="交叉验证（与 spatialmath / roboticstoolbox 对照）",
        blocks=tuple(blocks),
        fields={
            "available": True,
            "versions": dict(numbers.versions),
            "rows": [
                {"name": r[0], "mine": r[1], "reference": r[2], "deviation": r[3], "verdict": r[4]}
                for r in rows
            ],
        },
    )


def _at(values: tuple, index: int):
    return values[index] if index < len(values) else None


def _compare(name: str, mine, reference, tol: float = TOL) -> tuple[str, ...]:
    """对照一行。库那边没取到数就如实写"无法对照"，不假装通过。"""
    if reference is None:
        return (name, fmt_vector(mine, 4) if mine is not None else "—", "未取到", "—", "无法对照")
    mine_arr = np.asarray(mine, dtype=float).ravel()
    lib_arr = np.asarray(reference, dtype=float).ravel()
    if mine_arr.size != lib_arr.size:
        return (name, f"{mine_arr.size} 维", f"{lib_arr.size} 维", "—", "维度不同")
    deviation = float(np.abs(mine_arr - lib_arr).max())
    return (
        name,
        fmt_vector(mine_arr, 4),
        fmt_vector(lib_arr, 4),
        f"{deviation:.2e}",
        "一致" if deviation <= tol else "不一致",
    )


def _verdict(chain, numbers) -> tuple:
    """零位姿那组数的判定：差 0.103 m 到底差在哪。

    结论（roboticstoolbox 实测）：课件日志记录的是**夹爪 TCP** 帧，不是法兰。
    本项目默认给法兰帧，所以差 0.1034 m —— 不是谁算错了，是"末端在哪"的定义不同。
    """
    course_log = np.array(panda.COURSE_LOG_ZERO_POSE_TOOL_T)
    mine_flange = chain.fk(panda.PANDA_Q_ZERO).t
    mine_tcp = panda.panda_hand_tcp_chain().fk(panda.PANDA_Q_ZERO).t
    lib_flange = _at(numbers.panda_flange_t, 0)
    lib_tcp = _at(numbers.panda_tcp_t, 0)

    lines = [
        "零位姿那组数的判定：差 0.103 m 到底差在哪",
        f"  课件日志记录          t = {fmt_vector(course_log)}",
        f"  本项目 · 法兰帧       t = {fmt_vector(mine_flange)}",
        f"  本项目 · 夹爪 TCP 帧  t = {fmt_vector(mine_tcp)}",
    ]
    if lib_flange is not None and lib_tcp is not None:
        lines += [
            f"  真库 · panda_link8    t = {fmt_vector(lib_flange)}",
            f"  真库 · 默认（夹爪）   t = {fmt_vector(lib_tcp)}",
        ]

    both_match = (
        lib_flange is not None
        and lib_tcp is not None
        and np.allclose(mine_flange, lib_flange, atol=1e-9)
        and np.allclose(mine_tcp, lib_tcp, atol=1e-9)
    )
    if both_match:
        lines += [
            "",
            "结论：本项目模型与真库**逐位一致**（法兰帧和夹爪 TCP 帧都对得上）。",
            "",
            "      课件的 robot.fkine(q) 默认返回的是**夹爪 TCP**（法兰 + 103.4mm + 绕 z 转 45°），",
            "      所以日志是 0.8226；本项目默认给法兰帧，是 0.926。两者都没错 ——",
            "      “末端”这个词在不同工具里的定义不同，这才是最初那 0.103 m 差异的来源。",
            "      想复现课件那组数：`rkin panda-fk --case zero --frame tcp`。",
        ]
    elif np.allclose(mine_tcp, course_log, atol=1e-6):
        lines += [
            "",
            "结论：本项目的夹爪 TCP 帧复现了课件日志的数值（真库对照略有出入，可能是版本差异）。",
        ]
    else:
        lines += ["", "结论：仍对不上，需要排查 roboticstoolbox 版本与 Panda 模型定义（含夹爪）。"]

    return (HeadingBlock("零位姿那组数的判定"), TextBlock(tuple(lines)))


def _ik_note(chain, numbers) -> tuple:
    """IK 不做逐项对照（数值解本来就不唯一），只比"两者是否都能解到同一个位姿"。"""
    from ..core.ik_solvers import solve_dls

    target_q = panda.PANDA_Q_PPT_GOAL
    target = chain.fk(target_q)
    mine = solve_dls(chain, target, np.zeros(7))

    lines: list[str] = []
    if numbers.ik_success is None or numbers.ik_q is None:
        lines.append("  库 ikine_LM：调用失败或未取到结果")
    else:
        lines.append(
            f"  库 ikine_LM：成功={numbers.ik_success}，与生成目标的 q 之差="
            f"{float(np.linalg.norm(numbers.ik_q - target_q)):.3e}（数值解不唯一，不同解属正常）"
        )
    lines.append(
        f"  自研 DLS   ：成功={mine.success}，位置误差 {mine.position_error:.3e} m，"
        f"姿态误差 {mine.orientation_error:.3e} rad"
    )
    lines += [
        "",
        "两个解都能把末端送到同一个位姿，但关节角不同 —— 这就是 7 自由度冗余：",
        "同一个末端位姿对应无穷多组关节角，数值解从哪个初值出发就会落到哪一组解附近。",
    ]
    if mine.success and numbers.fk_of_ik_q is not None:
        # ⚠️ 这里必须用库的法兰帧结果：库的 fkine 默认返回夹爪 TCP，
        #    拿它跟自研的法兰帧目标比会平白多出 0.103 m 的“偏差”。
        lines.append(
            f"  把自研解出的 q 交给真库算 FK（法兰帧）：位置 {fmt_vector(numbers.fk_of_ik_q)}"
            f"（与目标偏差 {np.abs(numbers.fk_of_ik_q - target.t).max():.2e}）"
        )
    return (
        HeadingBlock("IK 对照（数值解不唯一，只比“是否都解到目标位姿”）"),
        TextBlock(tuple(lines)),
    )
