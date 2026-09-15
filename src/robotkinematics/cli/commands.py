"""@file commands.py
@brief 组合根：把配置 + 适配器 + 用例拼成一条命令。

【这里是什么】每个子命令一个函数，只做三件事：
  1. 从配置造出需要的适配器（算例来源、渲染器、绘图器、参照库）；
  2. 调用对应的用例拿到 Report；
  3. 需要副作用（出图）时顺手做掉，并把"图存哪了"作为附加行返回。

用例不知道终端、不知道文件、不知道 matplotlib —— 想确认"`rkin panda-ik` 到底用了哪些
实现"，只看这个文件就够了。
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np

from ..adapters.cases_csv import CsvCaseSource
from ..adapters.reference_rtb import RtbReference
from ..contracts import Report
from ..core import panda
from ..core.exceptions import KinematicsError
from ..core.ik_solvers import planar_chain
from ..core.planar2r import Planar2R
from ..core.se3 import SE3
from ..usecases import batch, compare, env_check, planar, pose, transforms
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
    """出图时才导入 matplotlib —— 其余命令不该为它付出"启动就加载"的代价。

    这也让无图形环境（容器、CI）跑非绘图命令时完全不碰 matplotlib，
    那些环境里的 3D 模块问题就不会影响到普通命令。
    """
    from ..adapters.plot_mpl import MatplotlibPlotter

    return MatplotlibPlotter(outputs_dir=config.outputs_dir_path)


def _resolve_panda_q(args, config) -> np.ndarray:
    """--q 直接给；否则从 panda_cases.csv 里按 --case 取。

    ⚠️ 用 getattr：这个helper被 panda-fk / panda-jac / panda-ik 共用，
       而 panda-ik 没有 --q（它的目标是位姿，不是关节角）。直接写 args.q 会在那条命令上炸。
    """
    explicit = getattr(args, "q", None)
    if explicit is not None:
        return np.asarray(explicit, dtype=float)
    source = _make_source(config)
    for case in source.panda_cases():
        if case.case_id == args.case:
            return case.q
    available = ", ".join(c.case_id for c in source.panda_cases())
    raise KinematicsError(f"没有 Panda 姿态 {args.case!r}；可用：{available}")


def _panda_chain(frame: str):
    """末端帧决定用哪条链 —— 由调用方解析，用例不需要知道 frame 这回事。"""
    return panda.panda_hand_tcp_chain() if frame == "tcp" else panda.panda_urdf_chain()


# ── 各子命令 ────────────────────────────────────────────────────────────────


def _cmd_check(args, config) -> CommandOutput:
    return CommandOutput(env_check.run())


def _cmd_pose(args, config) -> CommandOutput:
    # ⚠️ 显式展开而不是 `*args.xyz`：同时用「可变位置参数 + 关键字参数」时，
    #    类型检查无法确定解包会不会把 point 也占了位，会被判为"重复传参"。
    x, y, z = args.xyz
    roll_deg, pitch_deg, yaw_deg = args.rpy
    return CommandOutput(pose.run(x, y, z, roll_deg, pitch_deg, yaw_deg, point=tuple(args.point)))


def _cmd_cam(args, config) -> CommandOutput:
    source = _make_source(config)
    cases = source.coordinate_cases()
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
    chain = _panda_chain(args.frame)
    q = _resolve_panda_q(args, config)
    report = panda_uc.fk_report(chain, q)
    extra: list[str] = []
    if args.plot:
        # 出图是"呈现方式"的选择，属于命令行这一层，用例不知道它的存在
        path = _make_plotter(config).plot_panda_skeleton(chain, q)
        extra.append(f"骨架图已保存：{path}")
    return CommandOutput(report, tuple(extra))


def _cmd_panda_ik(args, config) -> CommandOutput:
    chain = panda.panda_urdf_chain()
    if args.pose is not None:
        x, y, z, roll_deg, pitch_deg, yaw_deg = args.pose
        target = SE3.from_pose6(x, y, z, *np.deg2rad([roll_deg, pitch_deg, yaw_deg]))
    else:
        target = chain.fk(_resolve_panda_q(args, config))

    seed = np.asarray(args.seed, dtype=float) if args.seed is not None else panda.PANDA_Q_IK_SEED
    prefer = np.asarray(args.prefer_config, dtype=float) if args.prefer_config is not None else None
    return CommandOutput(
        panda_uc.ik_report(
            chain,
            target,
            seed,
            dls_lambda=config.ik.dls_lambda,
            max_iter=config.ik.max_iter,
            tol=config.ik.tol,
            show_redundancy=not args.no_redundancy,
            prefer_config=prefer,
        )
    )


def _cmd_panda_jac(args, config) -> CommandOutput:
    return CommandOutput(
        panda_uc.jacobian_report(
            panda.panda_urdf_chain(),
            _resolve_panda_q(args, config),
            det_eps=config.singularity.det_eps,
            cond_warn=config.singularity.cond_warn,
        )
    )


def _cmd_panda_sing(args, config) -> CommandOutput:
    return CommandOutput(
        panda_uc.singular_pose_report(panda.panda_urdf_chain(), det_eps=config.singularity.det_eps)
    )


def _cmd_anim(args, config) -> CommandOutput:
    # 纯副作用命令：没有报告，只有产物路径
    path = _make_plotter(config).animate_two_link(
        _make_arm(config), sweep=args.sweep, frames=args.frames, q1_deg=args.q1, q2_deg=args.q2
    )
    return CommandOutput(None, (f"动画已保存：{path}",))


def _cmd_batch(args, config) -> CommandOutput:
    return CommandOutput(batch.run(_make_source(config)))


def _cmd_pick(args, config) -> CommandOutput:
    """抓取流水线：组合根负责把"哪台机器人 + 什么参数"装配好，用例只管跑七道工序。"""
    if args.arm == "planar":
        arm = _make_arm(config)
        chain = planar_chain(l1=arm.l1, l2=arm.l2)
        limits = None  # 二连杆模型没有限位信息，如实传 None（而不是编一组）
    else:
        arm = None
        chain = panda.panda_urdf_chain()
        limits = np.array(panda.PANDA_JOINT_LIMITS)

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
        limits=limits,
        planar=arm,
        mode=args.mode,
        prefer_config=prefer,
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
    )
    result = pick_uc.pick(chain, task, params)
    report = pick_uc.to_report(task, result)

    extra: list[str] = []
    if args.plot:
        path = _make_plotter(config).plot_pick_result(task, result, chain)
        extra.append(f"任务图已保存：{path}")
    return CommandOutput(report, tuple(extra))


def _cmd_crosscheck(args, config) -> CommandOutput:
    return CommandOutput(compare.compare(RtbReference()))


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
    "crosscheck": _cmd_crosscheck,
}
