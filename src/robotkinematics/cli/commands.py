"""@file commands.py
@brief 组合根：把配置 + 适配器 + 用例拼成一条命令。

【这里是什么】每个子命令一个函数，只做三件事：
  1. 从配置造出模型与适配器（机器人模型、算例来源、渲染器、绘图器）；
  2. 调用对应的用例拿到 Report；
  3. 需要副作用（出图）时顺手做掉，并把"图存哪了"作为附加行返回。

用例不知道终端、不知道文件、不知道 matplotlib —— 想确认"`rkin panda-ik` 到底用了哪些
实现"，只看这个文件就够了。
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
from spatialmath import SE3

from ..adapters.cases_csv import CsvCaseSource
from ..contracts import Report
from ..core import robots
from ..core.exceptions import KinematicsError
from ..core.planar2r import Planar2R
from ..usecases import batch, env_check, planar, pose, transforms
from ..usecases import panda as panda_uc
from ..usecases import pick as pick_uc


@dataclass(frozen=True)
class CommandOutput:
    """命令产出：一份报告（可能没有）+ 若干附加行（比如"图已保存"）。"""

    report: Report | None = None
    extra_lines: tuple[str, ...] = field(default_factory=tuple)


def dispatch(args, config) -> CommandOutput:
    handler = _HANDLERS.get(args.command)
    if handler is None:
        raise KinematicsError(f"未知命令: {args.command}")
    return handler(args, config)


def _make_arm(config) -> Planar2R:
    return Planar2R(l1=config.planar.l1, l2=config.planar.l2)


def _make_source(config) -> CsvCaseSource:
    return CsvCaseSource(config.cases_dir_path)


def _make_plotter(config):
    """出图时才导入 matplotlib —— 其余命令不该为它付出"启动就加载"的代价。"""
    from ..adapters.plot_mpl import MatplotlibPlotter

    return MatplotlibPlotter(outputs_dir=config.outputs_dir_path)


def _resolve_panda_q(args, config) -> np.ndarray:
    """--q 直接给；否则从 panda_cases.csv 里按 --case 取。

    ⚠️ 用 getattr：这个 helper 被 panda-fk / panda-jac / panda-ik 共用，
       而 panda-ik 没有 --q（它的目标是位姿，不是关节角）。
    """
    explicit = getattr(args, "q", None)
    if explicit is not None:
        return np.asarray(explicit, dtype=float)
    source = _make_source(config)
    cases = source.panda_cases()
    for case in cases:
        if case.case_id == args.case:
            return case.q
    available = ", ".join(c.case_id for c in cases)
    raise KinematicsError(f"没有 Panda 姿态 {args.case!r}；可用：{available}")


# ── 各子命令 ────────────────────────────────────────────────────────────────


def _cmd_check(args, config) -> CommandOutput:
    return CommandOutput(env_check.run())


def _cmd_pose(args, config) -> CommandOutput:
    x, y, z = args.xyz
    roll_deg, pitch_deg, yaw_deg = args.rpy
    return CommandOutput(pose.run(x, y, z, roll_deg, pitch_deg, yaw_deg, point=tuple(args.point)))


def _cmd_cam(args, config) -> CommandOutput:
    cases = _make_source(config).coordinate_cases()
    if args.all:
        return CommandOutput(transforms.all_cases_report(cases))
    for case in cases:
        if case.case_id == args.case:
            return CommandOutput(transforms.case_report(case))
    raise KinematicsError(f"没有算例 {args.case!r}；可用：{', '.join(c.case_id for c in cases)}")


def _cmd_fk(args, config) -> CommandOutput:
    arm = _make_arm(config)
    report = planar.fk_report(arm, args.q1, args.q2, chart=not args.no_ascii)
    extra: list[str] = []
    if args.plot:
        path = _make_plotter(config).plot_two_link(arm, args.q1, args.q2)
        extra.append(f"姿态图已保存：{path}")
    return CommandOutput(report, tuple(extra))


def _cmd_ik(args, config) -> CommandOutput:
    return CommandOutput(planar.ik_report(_make_arm(config), *args.target))


def _cmd_jac(args, config) -> CommandOutput:
    return CommandOutput(
        planar.jacobian_report(
            _make_arm(config),
            args.q1,
            args.q2,
            tuple(args.dx),
            det_eps=config.singularity.det_eps,
            cond_warn=config.singularity.cond_warn,
        )
    )


def _cmd_sing(args, config) -> CommandOutput:
    arm = _make_arm(config)
    if args.q2 is not None:
        return CommandOutput(
            planar.jacobian_report(
                arm,
                args.q1,
                args.q2,
                (0.0, 0.0),
                det_eps=config.singularity.det_eps,
                cond_warn=config.singularity.cond_warn,
            )
        )
    return CommandOutput(planar.singularity_sweep(arm, det_eps=config.singularity.det_eps))


def _cmd_panda_fk(args, config) -> CommandOutput:
    robot = robots.panda(args.frame)
    q = _resolve_panda_q(args, config)
    report = panda_uc.fk_report(robot, q, frame=args.frame)
    extra: list[str] = []
    if args.plot:
        path = _make_plotter(config).plot_panda_skeleton(robot, q)
        extra.append(f"骨架图已保存：{path}")
    return CommandOutput(report, tuple(extra))


def _cmd_panda_ik(args, config) -> CommandOutput:
    robot = robots.panda(robots.TCP)  # 见 panda_uc 说明：库的 IK 只对默认末端求解
    if args.pose is not None:
        x, y, z, roll_deg, pitch_deg, yaw_deg = args.pose
        target = SE3(x, y, z) * SE3.RPY(
            *np.deg2rad([roll_deg, pitch_deg, yaw_deg]), order=panda_uc.RPY_ORDER
        )
    else:
        target = robots.end_pose(robot, _resolve_panda_q(args, config), robots.TCP)

    seed = np.asarray(args.seed, dtype=float) if args.seed is not None else panda_uc.Q_IK_SEED
    return CommandOutput(
        panda_uc.ik_report(
            robot,
            target,
            seed,
            max_iter=config.ik.max_iter,
            tol=config.ik.tol,
            show_redundancy=not args.no_redundancy,
            frame=robots.TCP,
        )
    )


def _cmd_panda_jac(args, config) -> CommandOutput:
    return CommandOutput(panda_uc.jacobian_report(robots.panda(), _resolve_panda_q(args, config)))


def _cmd_panda_sing(args, config) -> CommandOutput:
    return CommandOutput(panda_uc.singular_pose_report(robots.panda()))


def _cmd_anim(args, config) -> CommandOutput:
    path = _make_plotter(config).animate_two_link(
        _make_arm(config), sweep=args.sweep, frames=args.frames, q1_deg=args.q1, q2_deg=args.q2
    )
    return CommandOutput(None, (f"动画已保存：{path}",))


def _cmd_batch(args, config) -> CommandOutput:
    return CommandOutput(batch.run(_make_source(config)))


def _cmd_pick(args, config) -> CommandOutput:
    """抓取流水线：组合根负责把"哪台机器人 + 什么任务"装配好，用例跑七道工序。"""
    if args.arm == "planar":
        planar_arm = _make_arm(config)
        robot = robots.planar(l1=planar_arm.l1, l2=planar_arm.l2)
    else:
        planar_arm = None
        robot = robots.panda(robots.TCP)  # 抓取的末端是夹爪，不是法兰

    prefer = None
    if args.prefer_config is not None:
        if args.arm != "panda":
            raise KinematicsError("--prefer-config 只在 --arm panda 下有意义（二连杆没有冗余）")
        prefer = np.asarray(args.prefer_config, dtype=float)

    task = pick_uc.PickTask(
        observation=np.asarray(args.observe, dtype=float),
        camera_xyz=(config.camera.x, config.camera.y, config.camera.z),
        camera_yaw_deg=config.camera.yaw_deg,
        orientation_rpy_deg=tuple(args.rpy),
        arm=args.arm,
        mode=args.mode,
        prefer_config=prefer,
        planar=planar_arm,
    )
    params = pick_uc.PickParams(
        seeds=args.seeds or config.pick.seeds,
        residual_tol=config.pick.residual_tol,
        unreachable_residual=config.pick.unreachable_residual,
        min_sigma=config.pick.min_sigma,
        workspace_samples=config.pick.workspace_samples,
        workspace_seed=config.pick.workspace_seed,
        limit_margin_deg=config.pick.limit_margin_deg,
        max_iter=config.ik.max_iter,
        ik_tol=config.ik.tol,
    )
    result = pick_uc.pick(robot, task, params)
    report = pick_uc.to_report(task, result)

    extra: list[str] = []
    if args.plot:
        path = _make_plotter(config).plot_pick_result(task, result, robot)
        extra.append(f"任务图已保存：{path}")
    return CommandOutput(report, tuple(extra))


_HANDLERS: dict[str, Callable] = {
    "check": _cmd_check,
    "pose": _cmd_pose,
    "cam": _cmd_cam,
    "fk": _cmd_fk,
    "ik": _cmd_ik,
    "jac": _cmd_jac,
    "sing": _cmd_sing,
    "panda-fk": _cmd_panda_fk,
    "panda-ik": _cmd_panda_ik,
    "panda-jac": _cmd_panda_jac,
    "panda-sing": _cmd_panda_sing,
    "anim": _cmd_anim,
    "batch": _cmd_batch,
    "pick": _cmd_pick,
}
