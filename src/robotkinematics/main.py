"""@file main.py

@brief **唯一入口 = 组合根**：这个程序能干什么、用到哪些实现，都写在这一个文件里。

【四段结构，从上往下读就是它的骨架】

  ① 命令表（`command_table`）—— 命令名 → 处理函数。一眼看全"这个程序有哪些能力"。
  ② 装配（`Context` / `build_context`）—— **全工程唯一一处"选实现"的地方**：
     配置、渲染器、模型、算例来源、绘图器都在这一个函数里被选中并注入。
     除了这个文件，任何地方都不该出现"造一个具体实现"。
  ③ 入口流程（`main`）—— 解析 → 装配 → 查表 → 执行 → 渲染，以及唯一的异常出口。
  ④ 处理函数（`cmd_*`）—— 一条命令一个函数，只做一件事：把参数翻译成对
     `usecases/` 提供方函数的调用，并把结果包成 `CommandOutput`。

【为什么处理函数在文件末尾、命令表在开头】读代码的人先要的是"全貌"，再是"细节"。
`command_table()` 之所以是个函数而不是模块级字典，是因为 Python 要先定义后引用
（表要指向下面那些函数）—— 这是语法约束，不是设计选择。

【异常策略】core / usecases / adapters 遇错就抛 `KinematicsError`，不兜底也不打印；
这里作为唯一的顶层统一捕获，写 stderr 并返回非零退出码（走 stderr 而不是 stdout，
这样 `rkin fk ... > out.txt` 重定向时错误仍显示在终端）。
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
from spatialmath import SE3

from . import __version__
from .adapters.cases_csv import CsvCaseSource
from .adapters.config_yaml import Config
from .adapters.render_json import JsonRenderer
from .adapters.render_text import TextRenderer
from .contracts import Renderer, Report
from .core import robots
from .core.exceptions import KinematicsError
from .core.planar2r import Planar2R
from .usecases import batch, env_check, planar, pose, transforms
from .usecases import panda as panda_uc
from .usecases import pick as pick_uc

# ══════════════════════════════════════════════════════════════════════════
# ① 命令表：命令 → 处理函数
# ══════════════════════════════════════════════════════════════════════════


def command_table() -> dict[str, Callable[[argparse.Namespace, Context], CommandOutput]]:
    """@brief 这个程序的全部能力：命令名 → 处理函数。

    【读法】左边是用户在命令行敲的，右边是处理函数（定义在文件末尾）。
    处理函数的 docstring 里写着它调用哪个提供方函数 —— 表和函数合起来就是
    "入口用到哪些接口"的完整答案。

    @return 命令名 → 处理函数；每条命令一个函数，签名统一为 (args, ctx)
    """
    return {
        "check": cmd_check,
        "pose": cmd_pose,
        "cam": cmd_cam,
        "fk": cmd_fk,
        "ik": cmd_ik,
        "jac": cmd_jac,
        "sing": cmd_sing,
        "panda-fk": cmd_panda_fk,
        "panda-ik": cmd_panda_ik,
        "panda-jac": cmd_panda_jac,
        "panda-sing": cmd_panda_sing,
        "anim": cmd_anim,
        "batch": cmd_batch,
        "pick": cmd_pick,
    }


# ══════════════════════════════════════════════════════════════════════════
# ② 装配：全工程唯一一处"选实现"
# ══════════════════════════════════════════════════════════════════════════

# 配置文件里的哪一项对应哪个命令行开关（命令行覆盖优先）
CONFIG_OVERRIDES = {
    "l1": "planar_l1",
    "l2": "planar_l2",
    "dls_lambda": "ik_dls_lambda",
    "max_iter": "ik_max_iter",
    "tol": "ik_tol",
    "step": "ik_step",
    "det_eps": "singularity_det_eps",
    "cond_warn": "singularity_cond_warn",
}


@dataclass(frozen=True)
class CommandOutput:
    """命令产出：一份报告（可能没有）+ 若干附加行（比如"图已保存到哪"）。"""

    report: Report | None = None
    extra_lines: tuple[str, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class Context:
    """① 装配的产物：这一次运行所依赖的全部具体实现。

    处理函数只接收它、不自己造东西 —— 想换实现（比如换成 JSON 渲染器、
    换个算例来源），只改 `build_context` 一处。
    """

    config: Config
    renderer: Renderer
    source: CsvCaseSource
    arm: Planar2R

    def plotter(self):
        """出图器：**用到时才 import matplotlib** —— 其余命令不该为它付出启动代价。

        所以这里是工厂方法而不是字段：越晚 import，命令启动越快。
        """
        from .adapters.plot_mpl import MatplotlibPlotter

        return MatplotlibPlotter(outputs_dir=self.config.outputs_dir_path)


def build_context(args: argparse.Namespace) -> Context:
    """@brief 读配置、按命令行覆盖、把这次运行要用的实现全部造好。

    @param args argparse 解析出来的参数
    @return 装配好的上下文（配置 / 渲染器 / 模型 / 算例来源）
    """
    config = Config.from_yaml(args.config).override(
        **{
            target: getattr(args, source)
            for source, target in CONFIG_OVERRIDES.items()
            if getattr(args, source, None) is not None
        }
    )
    renderer: Renderer = (
        JsonRenderer(outputs_dir=config.outputs_dir_path)
        if args.format == "json"
        else TextRenderer(outputs_dir=config.outputs_dir_path)
    )
    return Context(
        config=config,
        renderer=renderer,
        source=CsvCaseSource(config.cases_dir_path),
        arm=Planar2R(l1=config.planar.l1, l2=config.planar.l2),
    )


# ══════════════════════════════════════════════════════════════════════════
# ③ 入口流程
# ══════════════════════════════════════════════════════════════════════════


def main(argv: list[str] | None = None) -> int:
    """@brief 程序入口：解析 → 装配 → 查表 → 执行 → 渲染 → 打印。

    @param argv 命令行参数；None 表示取 sys.argv
    @return 进程退出码：0 成功、1 领域错误、130 被 Ctrl-C 中断
    """
    args = build_parser().parse_args(argv)
    try:
        context = build_context(args)
        handler = command_table().get(args.command)
        if handler is None:  # 只有命令表与 argparse 不同步时才会发生
            raise KinematicsError(f"未知命令: {args.command}")
        output = handler(args, context)
    except KinematicsError as exc:
        print(f"错误: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("\n已中断", file=sys.stderr)
        return 130

    parts = []
    if output.report is not None:
        parts.append(context.renderer.render(output.report))
    parts.extend(output.extra_lines)
    print("\n".join(parts))
    return 0


# ══════════════════════════════════════════════════════════════════════════
# ④ 处理函数：一条命令一个
#
# 每个函数只做一件事：把命令行参数翻译成对 usecases/ 提供方函数的调用，
# 需要副作用（出图）时顺手做掉，并把"图存哪了"作为附加行返回。
# 这里不出现任何数学、不认识终端布局 —— 那两件事分别是 core 与 adapters 的活。
# ══════════════════════════════════════════════════════════════════════════


def _panda_q(args: argparse.Namespace, ctx: Context) -> np.ndarray:
    """取 Panda 关节角：`--q` 直接给，否则按 `--case` 从算例 CSV 里取。

    ⚠️ 用 getattr：panda-fk / panda-jac / panda-ik 共用本函数，而 panda-ik 没有 `--q`
       （它的输入是目标位姿，不是关节角）。
    """
    explicit = getattr(args, "q", None)
    if explicit is not None:
        return np.asarray(explicit, dtype=float)
    cases = ctx.source.panda_cases()
    for case in cases:
        if case.case_id == args.case:
            return case.q
    available = ", ".join(c.case_id for c in cases)
    raise KinematicsError(f"没有 Panda 姿态 {args.case!r}；可用：{available}")


def cmd_check(args: argparse.Namespace, ctx: Context) -> CommandOutput:
    """环境自检 → `usecases.env_check.run`。"""
    return CommandOutput(env_check.run())


def cmd_pose(args: argparse.Namespace, ctx: Context) -> CommandOutput:
    """位姿表示互转 → `usecases.pose.run`。"""
    x, y, z = args.xyz
    roll_deg, pitch_deg, yaw_deg = args.rpy
    return CommandOutput(pose.run(x, y, z, roll_deg, pitch_deg, yaw_deg, point=tuple(args.point)))


def cmd_cam(args: argparse.Namespace, ctx: Context) -> CommandOutput:
    """相机 → 本体变换链 → `usecases.transforms.case_report / all_cases_report`。"""
    cases = ctx.source.coordinate_cases()
    if args.all:
        return CommandOutput(transforms.all_cases_report(cases))
    for case in cases:
        if case.case_id == args.case:
            return CommandOutput(transforms.case_report(case))
    raise KinematicsError(f"没有算例 {args.case!r}；可用：{', '.join(c.case_id for c in cases)}")


def cmd_fk(args: argparse.Namespace, ctx: Context) -> CommandOutput:
    """二连杆正解 → `usecases.planar.fk_report`（可选出图 → adapters.plot_mpl）。"""
    report = planar.fk_report(ctx.arm, args.q1, args.q2, chart=not args.no_ascii)
    extra: list[str] = []
    if args.plot:
        path = ctx.plotter().plot_two_link(ctx.arm, args.q1, args.q2)
        extra.append(f"姿态图已保存：{path}")
    return CommandOutput(report, tuple(extra))


def cmd_ik(args: argparse.Namespace, ctx: Context) -> CommandOutput:
    """二连杆逆解（两组解）→ `usecases.planar.ik_report`。"""
    return CommandOutput(planar.ik_report(ctx.arm, *args.target))


def cmd_jac(args: argparse.Namespace, ctx: Context) -> CommandOutput:
    """二连杆雅可比与微调 → `usecases.planar.jacobian_report`。"""
    return CommandOutput(
        planar.jacobian_report(
            ctx.arm,
            args.q1,
            args.q2,
            tuple(args.dx),
            det_eps=ctx.config.singularity.det_eps,
            cond_warn=ctx.config.singularity.cond_warn,
        )
    )


def cmd_sing(args: argparse.Namespace, ctx: Context) -> CommandOutput:
    """二连杆奇异点体检：给了 --q2 看单个位姿，否则扫四个典型姿态。

    两种输入分别落到 `usecases.planar.jacobian_report` 与 `singularity_sweep`。
    """
    if args.q2 is not None:
        return CommandOutput(
            planar.jacobian_report(
                ctx.arm,
                args.q1,
                args.q2,
                (0.0, 0.0),
                det_eps=ctx.config.singularity.det_eps,
                cond_warn=ctx.config.singularity.cond_warn,
            )
        )
    return CommandOutput(planar.singularity_sweep(ctx.arm, det_eps=ctx.config.singularity.det_eps))


def cmd_panda_fk(args: argparse.Namespace, ctx: Context) -> CommandOutput:
    """Panda 正解（法兰或夹爪帧）→ `core.robots.panda` + `usecases.panda.fk_report`。"""
    robot = robots.panda(args.frame)
    q = _panda_q(args, ctx)
    report = panda_uc.fk_report(robot, q, frame=args.frame)
    extra: list[str] = []
    if args.plot:
        path = ctx.plotter().plot_panda_skeleton(robot, q)
        extra.append(f"骨架图已保存：{path}")
    return CommandOutput(report, tuple(extra))


def cmd_panda_ik(args: argparse.Namespace, ctx: Context) -> CommandOutput:
    """Panda 数值逆解 → `usecases.panda.ik_report`（末端固定取夹爪 TCP）。"""
    robot = robots.panda(robots.TCP)  # 库的 IK 只对默认末端求解，见 usecases/panda.py
    if args.pose is not None:
        x, y, z, roll_deg, pitch_deg, yaw_deg = args.pose
        target = SE3(x, y, z) * SE3.RPY(
            *np.deg2rad([roll_deg, pitch_deg, yaw_deg]), order=panda_uc.RPY_ORDER
        )
    else:
        target = robots.end_pose(robot, _panda_q(args, ctx), robots.TCP)

    seed = np.asarray(args.seed, dtype=float) if args.seed is not None else panda_uc.Q_IK_SEED
    return CommandOutput(
        panda_uc.ik_report(
            robot,
            target,
            seed,
            max_iter=ctx.config.ik.max_iter,
            tol=ctx.config.ik.tol,
            show_redundancy=not args.no_redundancy,
            frame=robots.TCP,
        )
    )


def cmd_panda_jac(args: argparse.Namespace, ctx: Context) -> CommandOutput:
    """Panda 6×7 雅可比 → `usecases.panda.jacobian_report`。"""
    return CommandOutput(panda_uc.jacobian_report(robots.panda(), _panda_q(args, ctx)))


def cmd_panda_sing(args: argparse.Namespace, ctx: Context) -> CommandOutput:
    """Panda 三个典型姿态对比 → `usecases.panda.singular_pose_report`。"""
    return CommandOutput(panda_uc.singular_pose_report(robots.panda()))


def cmd_anim(args: argparse.Namespace, ctx: Context) -> CommandOutput:
    """关节扫动 GIF → `adapters.plot_mpl.MatplotlibPlotter.animate_two_link`。"""
    path = ctx.plotter().animate_two_link(
        ctx.arm, sweep=args.sweep, frames=args.frames, q1_deg=args.q1, q2_deg=args.q2
    )
    return CommandOutput(None, (f"动画已保存：{path}",))


def cmd_batch(args: argparse.Namespace, ctx: Context) -> CommandOutput:
    """批量算例报告 → `usecases.batch.run`。"""
    return CommandOutput(batch.run(ctx.source))


def cmd_pick(args: argparse.Namespace, ctx: Context) -> CommandOutput:
    """抓取流水线（七道工序 + 失败分类）→ `usecases.pick.pick / to_report`。

    这一条是唯一需要较真装配的：`--arm planar` 时机器人是二连杆、闭式解当暖启动；
    否则是 Panda 的夹爪帧（抓东西的不是法兰）。
    """
    if args.arm == "planar":
        robot = robots.planar(l1=ctx.arm.l1, l2=ctx.arm.l2)
        planar_arm: Planar2R | None = ctx.arm
    else:
        robot = robots.panda(robots.TCP)
        planar_arm = None

    prefer = None
    if args.prefer_config is not None:
        if args.arm != "panda":
            raise KinematicsError("--prefer-config 只在 --arm panda 下有意义（二连杆没有冗余）")
        prefer = np.asarray(args.prefer_config, dtype=float)

    task = pick_uc.PickTask(
        observation=np.asarray(args.observe, dtype=float),
        camera_xyz=(ctx.config.camera.x, ctx.config.camera.y, ctx.config.camera.z),
        camera_yaw_deg=ctx.config.camera.yaw_deg,
        orientation_rpy_deg=tuple(args.rpy),
        arm=args.arm,
        mode=args.mode,
        prefer_config=prefer,
        planar=planar_arm,
    )
    params = pick_uc.PickParams(
        seeds=args.seeds or ctx.config.pick.seeds,
        residual_tol=ctx.config.pick.residual_tol,
        unreachable_residual=ctx.config.pick.unreachable_residual,
        min_sigma=ctx.config.pick.min_sigma,
        workspace_samples=ctx.config.pick.workspace_samples,
        workspace_seed=ctx.config.pick.workspace_seed,
        limit_margin_deg=ctx.config.pick.limit_margin_deg,
        max_iter=ctx.config.ik.max_iter,
        ik_tol=ctx.config.ik.tol,
    )
    result = pick_uc.pick(robot, task, params)
    report = pick_uc.to_report(task, result)

    extra: list[str] = []
    if args.plot:
        path = ctx.plotter().plot_pick_result(task, result, robot)
        extra.append(f"任务图已保存：{path}")
    return CommandOutput(report, tuple(extra))


# ══════════════════════════════════════════════════════════════════════════
# 参数定义（输入契约）：这里只有"有哪些开关"，没有任何业务逻辑
# ══════════════════════════════════════════════════════════════════════════


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
    """@brief 定义全部子命令与开关（这里只有"有哪些参数"，没有业务）。

    @return 组装好的 argparse 解析器；每个子命令都挂了一份"覆盖配置文件"的开关
    """
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
