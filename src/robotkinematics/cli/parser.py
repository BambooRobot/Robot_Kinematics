"""@file parser.py
@brief 命令行参数定义 —— 这里只有"有哪些开关"，没有任何业务逻辑。

业务装配在 cli/commands.py（组合根），程序入口在 cli/main.py。
"""

from __future__ import annotations

import argparse

from .. import __version__


def _add_config_overrides(parser: argparse.ArgumentParser, suppress: bool = False) -> None:
    """挂上"覆盖配置文件"的一组开关。

    这些开关在主编译器和每个子命令上各挂一份，于是
    `rkin --l1 2 fk ...` 和 `rkin fk --l1 2 ...` 都能用。
    子命令上那份必须用 SUPPRESS 当默认值：否则子命令解析时会用默认值
    把主编译器已经读到的值冲掉（argparse 的经典坑）。
    """
    default = argparse.SUPPRESS if suppress else None
    suffix = "" if suppress else "（也可写在子命令之后）"
    parser.add_argument(
        "--l1", type=float, default=default, help=f"覆盖二连杆第一节长度 [m]{suffix}"
    )
    parser.add_argument(
        "--l2", type=float, default=default, help=f"覆盖二连杆第二节长度 [m]{suffix}"
    )
    parser.add_argument(
        "--dls-lambda", type=float, dest="dls_lambda", default=default, help="覆盖 IK 阻尼系数 λ"
    )
    parser.add_argument(
        "--max-iter", type=int, dest="max_iter", default=default, help="覆盖 IK 最大迭代次数"
    )
    parser.add_argument("--tol", type=float, default=default, help="覆盖 IK 收敛判据")
    parser.add_argument(
        "--det-eps", type=float, dest="det_eps", default=default, help="覆盖奇异判定阈值"
    )
    parser.add_argument(
        "--cond-warn", type=float, dest="cond_warn", default=default, help="覆盖条件数告警阈值"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="rkin",
        description="抓取任务流水线：七道工序把机器人运动学的七个知识点串成一条链"
        "（运动学由 spatialmath + roboticstoolbox 提供）",
    )
    parser.add_argument(
        "--config",
        default=None,
        help="配置文件路径（默认 configs/default.yaml）。命令行参数优先级高于配置文件",
    )
    parser.add_argument(
        "--format", choices=["text", "json"], default="text", help="输出格式（默认 text）"
    )
    parser.add_argument("--version", action="version", version=f"rkin {__version__}")
    _add_config_overrides(parser)

    sub = parser.add_subparsers(dest="command", required=True, metavar="<命令>")

    sub.add_parser("check", help="环境自检")

    p_pose = sub.add_parser("pose", help="位姿表示：六元组 → R/T，以及 RPY/轴角/四元数互转")
    p_pose.add_argument(
        "--xyz", nargs=3, type=float, default=[0.4, 0.2, 0.3], metavar=("X", "Y", "Z")
    )
    p_pose.add_argument(
        "--rpy",
        nargs=3,
        type=float,
        default=[0.0, 0.0, 90.0],
        metavar=("R", "P", "Y"),
        help="roll/pitch/yaw，单位是度",
    )
    p_pose.add_argument(
        "--point", nargs=3, type=float, default=[0.1, 0.0, 0.0], metavar=("X", "Y", "Z")
    )

    p_cam = sub.add_parser("cam", help="相机目标 → 机器人本体（坐标变换链）")
    p_cam.add_argument("--case", default="ppt_case", help="算例 id；用 --all 跑整张表")
    p_cam.add_argument(
        "--all", action="store_true", help="跑 data/cases/coordinate_cases.csv 全部算例"
    )

    p_fk = sub.add_parser("fk", help="二连杆正运动学")
    p_fk.add_argument("--q1", type=float, default=0.0, help="关节角，单位度")
    p_fk.add_argument("--q2", type=float, default=90.0, help="关节角，单位度")
    p_fk.add_argument("--no-ascii", action="store_true", help="不打印字符画")
    p_fk.add_argument("--plot", action="store_true", help="额外输出一张 2D 姿态图")

    p_ik = sub.add_parser("ik", help="二连杆逆运动学")
    p_ik.add_argument("--target", nargs=2, type=float, required=True, metavar=("X", "Y"))

    p_jac = sub.add_parser("jac", help="二连杆雅可比与关节微调")
    p_jac.add_argument("--q1", type=float, default=0.0, help="关节角，单位度")
    p_jac.add_argument("--q2", type=float, default=90.0, help="关节角，单位度")
    p_jac.add_argument("--dx", nargs=2, type=float, default=[0.0, 0.1], metavar=("DX", "DY"))

    p_sing = sub.add_parser("sing", help="二连杆奇异点扫描")
    p_sing.add_argument("--q2", type=float, default=None, help="只看某个 q2（度）")
    p_sing.add_argument("--q1", type=float, default=0.0, help="配合 --q2 使用")

    p_pfk = sub.add_parser("panda-fk", help="Panda 正运动学")
    p_pfk.add_argument(
        "--q", nargs=7, type=float, default=None, metavar=tuple(f"Q{i}" for i in range(1, 8))
    )
    p_pfk.add_argument("--case", default="ppt_goal", help="用 data/cases/panda_cases.csv 里的姿态")
    p_pfk.add_argument("--plot", action="store_true", help="额外输出骨架图")
    p_pfk.add_argument(
        "--frame",
        choices=["flange", "tcp"],
        default="flange",
        help="末端帧：flange=法兰(panda_link8，默认)，tcp=夹爪 TCP",
    )

    p_pik = sub.add_parser("panda-ik", help="Panda 数值逆运动学")
    p_pik.add_argument("--case", default="ppt_goal", help="目标位姿取自该姿态的 FK 结果")
    p_pik.add_argument(
        "--pose",
        nargs=6,
        type=float,
        default=None,
        metavar=("X", "Y", "Z", "R", "P", "Y2"),
        help="直接给目标位姿，角度单位是度",
    )
    p_pik.add_argument("--seed", nargs=7, type=float, default=None, metavar="Q")
    p_pik.add_argument(
        "--prefer-config",
        nargs=7,
        type=float,
        default=None,
        dest="prefer_config",
        metavar="Q",
        help="次要任务：让解更靠近这个参考姿态（零空间优化）",
    )
    p_pik.add_argument("--no-redundancy", action="store_true", help="不演示多初值冗余")

    p_pjac = sub.add_parser("panda-jac", help="Panda 6×7 几何雅可比")
    p_pjac.add_argument("--q", nargs=7, type=float, default=None, metavar="Q")
    p_pjac.add_argument("--case", default="ik_seed", help="用 data/cases/panda_cases.csv 里的姿态")

    sub.add_parser("panda-sing", help="对比 Panda 三个典型姿态的局部运动能力")

    p_anim = sub.add_parser("anim", help="关节扫动动画（GIF）")
    p_anim.add_argument("--sweep", choices=["q1", "q2"], default="q2")
    p_anim.add_argument("--frames", type=int, default=60)
    p_anim.add_argument("--q1", type=float, default=0.0, help="固定 q1（度）；扫 q2 时生效")
    p_anim.add_argument("--q2", type=float, default=0.0, help="固定 q2（度）；扫 q1 时生效")

    p_batch = sub.add_parser("batch", help="批量跑完算例集并落盘")
    p_batch.add_argument("--cases", default="all", choices=["all"], help="目前只支持 all")

    p_pick = sub.add_parser("pick", help="抓取任务流水线：观测 → 关节解（含失败分类）")
    p_pick.add_argument(
        "--observe",
        nargs=3,
        type=float,
        required=True,
        metavar=("X", "Y", "Z"),
        help="观测到的目标点（默认在相机坐标系下，见 --mode）",
    )
    p_pick.add_argument(
        "--arm",
        choices=["panda", "planar"],
        default="panda",
        help="panda=7 自由度真机规模；planar=二连杆教学对照（跑同一条流水线）",
    )
    p_pick.add_argument(
        "--mode",
        choices=["camera", "base"],
        default="camera",
        help="观测所在的坐标系：camera（默认）会先做变换链，base 表示已经是本体坐标",
    )
    p_pick.add_argument(
        "--rpy",
        nargs=3,
        type=float,
        default=[180.0, 0.0, 0.0],
        metavar=("R", "P", "Y"),
        help="抓取姿态（度），默认 180/0/0 = 工具 z 朝下（自上而下抓）",
    )
    p_pick.add_argument(
        "--prefer-config",
        nargs=7,
        type=float,
        default=None,
        dest="prefer_config",
        metavar="Q",
        help="次要任务：多解时优先靠近这个参考姿态（仅 panda）",
    )
    p_pick.add_argument("--seeds", type=int, default=None, help="覆盖多初值个数")
    p_pick.add_argument("--plot", action="store_true", help="额外输出一张任务图")

    # 每个子命令也挂一份覆盖开关，这样 `rkin fk --l1 2 ...` 也能用
    for sub_parser in sub.choices.values():
        _add_config_overrides(sub_parser, suppress=True)
    return parser
