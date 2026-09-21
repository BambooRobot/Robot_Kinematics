"""@file pick_steps.py

@brief 抓取流水线的**七道工序**：每一道一个函数，外加"失败时怎么归类"。机器人固定为 Panda。

七道工序（每一道对应一个知识点），以及每道工序由谁实现：

  ① 观测 → 本体位姿      位姿与坐标系、变换链        → spatialmath 的 SE3
  ② 可达性预筛            工作空间不是球              → 自研采样（库没有这个 API）
  ③ 多初值 IK             IK 的多解 / 无解 / 迭代     → robot.ikine_LM
  ④ 雅可比体检            雅可比与奇异点              → robot.jacob0 + numpy SVD
  ⑤ 限位校验              真机约束（关节限位）        → robot.qlim
  ⑥ 解择优                冗余与零空间                → 自研评分（库不提供）
  ⑦ 微动可行性校验        雅可比干活：解一次小位移    → robot.jacob0 + numpy.linalg.solve

【约定】每道工序的签名统一为 `(robot, task, params, state) -> Failure | None`：

  * 成功：把结果写进 `state` 的相关字段，并把这道工序的 `PickStep` 追加到 `state.steps`；
  * 失败：返回一个 `Failure`（有名字的结论 + 一句人话），**整条流水线就此停下**。
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import roboticstoolbox as rtb
from spatialmath import SE3

from ..contracts import fmt_vector
from ..core import robots, workspace
from .pick_types import (
    DEG,
    OUTCOME_ALL_LIMIT_VIOLATED,
    OUTCOME_ALL_NEAR_SINGULAR,
    OUTCOME_MICRO_MOTION_INFEASIBLE,
    OUTCOME_NOT_IN_WORKSPACE,
    OUTCOME_POSE_NOT_REACHABLE,
    OUTCOME_RESIDUAL_TOO_LARGE,
    RPY_ORDER,
    SCORE_WEIGHTS,
    Failure,
    PickCandidate,
    PickParams,
    PickStep,
    PickTask,
    PipelineState,
)

#: 一道工序：成功返回 None（结果写进 state），失败返回 Failure（流水线停下）
Step = Callable[[rtb.Robot, PickTask, PickParams, PipelineState], "Failure | None"]


def step_locate(
    robot: rtb.Robot, task: PickTask, params: PickParams, state: PipelineState
) -> Failure | None:
    """① 观测 → 本体位姿（知识点：位姿与坐标系、变换链）。"""
    observation = np.asarray(task.observation, dtype=float).reshape(3)

    if task.mode == "camera":
        T_base_camera = SE3.Trans(*task.camera_xyz) * SE3.Rz(DEG(task.camera_yaw_deg))
        point_base = np.asarray((T_base_camera * SE3(observation)).t, dtype=float)
        detail = (
            f"相机观测 {fmt_vector(observation)} → 本体 {fmt_vector(point_base)}"
            f"（^base T_camera · p_obs）"
        )
    else:
        point_base = observation
        detail = f"观测已在本体坐标系：{fmt_vector(point_base)}"

    roll, pitch, yaw = DEG(task.orientation_rpy_deg)
    target = SE3(point_base) * SE3.RPY(roll, pitch, yaw, order=RPY_ORDER)

    state.target = target
    state.steps.append(
        PickStep(
            name="① 观测 → 本体位姿",
            detail=detail,
            numbers={
                "observation": observation,
                "target_position": np.asarray(target.t, dtype=float),
                "target_distance": float(np.linalg.norm(target.t)),
            },
        )
    )
    return None


def step_prescreen(
    robot: rtb.Robot, task: PickTask, params: PickParams, state: PipelineState
) -> Failure | None:
    """② 可达性预筛（知识点：工作空间不是球）。"""
    assert state.target is not None
    limits = robots.joint_limits(robot)
    reach = workspace.directional_reach(
        robot,
        np.asarray(state.target.t, dtype=float),
        samples=params.workspace_samples,
        seed=params.workspace_seed,
        limits=limits,
    )
    distance = float(np.linalg.norm(state.target.t))
    margin = reach.margin(distance)
    inside = margin >= 0.0
    detail = f"{reach.summary()}；目标距离 {distance:.4f} m → " + (
        f"在界内（余量 {margin:+.4f} m）" if inside else f"**超出 {abs(margin):.4f} m**"
    )

    state.reach = reach
    state.steps.append(
        PickStep(
            name="② 可达性预筛",
            detail=detail,
            numbers={
                "direction_reach": reach.radius,
                "target_distance": distance,
                "margin": margin,
                "inside": inside,
            },
        )
    )
    return None


def step_solve(
    robot: rtb.Robot, task: PickTask, params: PickParams, state: PipelineState
) -> Failure | None:
    """③ 多初值 IK（知识点：IK 的多解 / 无解 / 数值迭代）。"""
    assert state.target is not None
    n = robot.n
    candidates: list[PickCandidate] = []
    best_residual = float("inf")

    seeds = [task.home(n)]
    rng = np.random.default_rng(params.workspace_seed + 1)
    seeds += [rng.uniform(-1.5, 1.5, size=n) for _ in range(max(params.seeds - 1, 0))]

    for index, seed in enumerate(seeds):
        # ⚠️ 不传 end：库的 IK 只对模型默认末端求解。抓取的末端就该是夹爪（见 PickTask.frame）
        solution = robot.ikine_LM(
            state.target,
            q0=np.asarray(seed, dtype=float).reshape(n),
            ilimit=params.max_iter,
            tol=params.ik_tol,
        )
        q = np.asarray(solution.q, dtype=float).reshape(n)
        residual, pos_err, ori_err = _task_error(robot, state.target, q, task)
        best_residual = min(best_residual, residual)
        if residual <= params.residual_tol:
            candidates.append(_candidate(q, index, residual, pos_err, ori_err))

    detail = (
        f"{len(seeds)} 个初值 → 收敛 {len(candidates)} 个"
        f"（判据：任务残差 < {params.residual_tol:.0e}，最佳 {best_residual:.2e}）"
    )
    numbers = {
        "seeds": len(seeds),
        "converged": len(candidates),
        "best_residual": best_residual,
        "residual_tol": params.residual_tol,
    }

    state.candidates = candidates
    state.best_residual = best_residual
    state.steps.append(PickStep(name="③ 多初值 IK", detail=detail, numbers=numbers))

    if not candidates:
        assert state.reach is not None
        classified = classify_no_candidate(task, state.reach, state.target, best_residual, params)
        return Failure(*classified)
    return None


def _candidate(
    q: np.ndarray, index: int, residual: float, pos_err: float, ori_err: float
) -> PickCandidate:
    """IK 阶段产出的候选：体检字段先占位，等 ④ 那一步填。"""
    return PickCandidate(
        q=q,
        seed_index=index,
        residual=residual,
        position_error=pos_err,
        orientation_error=ori_err,
        sigma_min=float("nan"),
        condition=float("nan"),
        manipulability=float("nan"),
        limit_margin=float("inf"),
    )


def step_inspect(
    robot: rtb.Robot, task: PickTask, params: PickParams, state: PipelineState
) -> Failure | None:
    """④ 雅可比体检 与 ⑤ 限位校验。"""
    from ..core.singularity import analyze

    limits = robots.joint_limits(robot)
    margin_required = DEG(params.limit_margin_deg)
    inspected: list[PickCandidate] = []
    near_singular = over_limit = 0
    sigma_values: list[float] = []

    for candidate in state.candidates:
        J = np.asarray(robot.jacob0(candidate.q), dtype=float)
        report = analyze(J)
        limit_margin = robots.limit_margin(candidate.q, limits)
        sigma_values.append(report.sigma_min)

        rejected = ""
        if report.sigma_min < params.min_sigma:
            rejected = OUTCOME_ALL_NEAR_SINGULAR
            near_singular += 1
        elif limit_margin < margin_required:
            rejected = OUTCOME_ALL_LIMIT_VIOLATED
            over_limit += 1

        inspected.append(
            PickCandidate(
                q=candidate.q,
                seed_index=candidate.seed_index,
                residual=candidate.residual,
                position_error=candidate.position_error,
                orientation_error=candidate.orientation_error,
                sigma_min=report.sigma_min,
                condition=report.condition,
                manipulability=report.manipulability,
                limit_margin=limit_margin,
                score=score_candidate(
                    candidate.q, report.sigma_min, report.sigma_max, limit_margin, task
                ),
                rejected_by=rejected,
            )
        )

    state.candidates = inspected
    state.steps.append(
        PickStep(
            name="④ 雅可比体检",
            detail=(
                f"σ_min 区间 [{min(sigma_values):.4f}, {max(sigma_values):.4f}]，"
                f"接近奇异 {near_singular} 组（阈值 {params.min_sigma}）"
            ),
            numbers={
                "sigma_min_range": (min(sigma_values), max(sigma_values)),
                "near_singular": near_singular,
            },
        )
    )
    state.steps.append(
        PickStep(
            name="⑤ 限位校验",
            detail=(
                "该模型没有限位信息，跳过"
                if limits is None
                else f"越限位 {over_limit} 组（余量要求 {params.limit_margin_deg:.1f}°）"
            ),
            numbers={"over_limit": over_limit, "checked": limits is not None},
        )
    )

    if not any(c.survived for c in inspected):
        return Failure(classify_all_rejected(inspected), reject_reason(inspected))
    return None


def step_choose(
    robot: rtb.Robot, task: PickTask, params: PickParams, state: PipelineState
) -> Failure | None:
    """⑥ 解择优（知识点：冗余与零空间）。"""
    survivors = [c for c in state.candidates if c.survived]
    best = max(survivors, key=lambda c: c.score)
    state.best = best
    state.steps.append(
        PickStep(
            name="⑥ 解择优",
            detail=(
                f"存活 {len(survivors)} 组，按「离限位余量 {SCORE_WEIGHTS[0]}"
                f" + σ_min {SCORE_WEIGHTS[1]} + 离偏好位形 {SCORE_WEIGHTS[2]}」评分择优"
                f"（偏好位形 = {'显式指定' if task.prefer_config is not None else '初始位形'}）："
                f"选中第 {best.seed_index + 1} 个初值的那组"
            ),
            numbers={
                "survivors": len(survivors),
                "chosen_seed_index": best.seed_index,
                "score": best.score,
                "weights": SCORE_WEIGHTS,
            },
        )
    )
    return None


def score_candidate(
    q: np.ndarray, sigma_min: float, sigma_max: float, limit_margin: float, task: PickTask
) -> float:
    """解择优的评分：限位余量 + σ_min + 离偏好位形，各项归一化后加权。"""
    w_limit, w_sigma, w_prefer = SCORE_WEIGHTS
    limit_score = (
        0.5
        if not np.isfinite(limit_margin)
        else float(np.clip(limit_margin / (np.pi / 4), 0.0, 1.0))
    )
    sigma_score = 0.0 if sigma_max <= 0 else float(np.clip(sigma_min / sigma_max, 0.0, 1.0))
    prefer_distance = float(np.linalg.norm(q - task.preferred(q.size)))
    prefer_score = float(np.clip(1.0 - prefer_distance / np.pi, 0.0, 1.0))
    return w_limit * limit_score + w_sigma * sigma_score + w_prefer * prefer_score


def step_micro_motion(
    robot: rtb.Robot, task: PickTask, params: PickParams, state: PipelineState
) -> Failure | None:
    """⑦ 微动可行性校验：用雅可比解一次小位移。"""
    assert state.best is not None
    J = np.asarray(robot.jacob0(state.best.q), dtype=float)[:6, :]
    delta = np.zeros(6)
    delta[2] = -params.micro_step_m  # 末端沿 z 压 1mm

    dq = np.linalg.solve(J @ J.T + (0.05**2) * np.eye(6), delta)
    dq = J.T @ dq

    joint_step_deg = float(np.rad2deg(np.linalg.norm(dq)))
    feasible = joint_step_deg <= params.max_joint_step_deg
    detail = (
        f"末端微动 {params.micro_step_m * 1000:.1f} mm → 关节需动 {joint_step_deg:.3f}°"
        f"（上限 {params.max_joint_step_deg:.1f}°）：{'可行' if feasible else '**不可行**'}"
    )
    state.steps.append(
        PickStep(
            name="⑦ 微动可行性校验",
            detail=detail,
            numbers={
                "micro_step_m": params.micro_step_m,
                "joint_step_deg": joint_step_deg,
                "max_joint_step_deg": params.max_joint_step_deg,
                "feasible": feasible,
                "dq": dq,
            },
        )
    )
    if not feasible:
        return Failure(OUTCOME_MICRO_MOTION_INFEASIBLE, detail)
    return None


PIPELINE: tuple[Step, ...] = (
    step_locate,
    step_prescreen,
    step_solve,
    step_inspect,
    step_choose,
    step_micro_motion,
)


def classify_no_candidate(
    task: PickTask,
    reach: workspace.ReachEstimate,
    target: SE3,
    best_residual: float,
    params: PickParams,
) -> tuple[str, str]:
    """没有任何候选解时，区分"够不着"和"没解出来"。"""
    distance = float(np.linalg.norm(target.t))
    margin = reach.margin(distance)
    if margin < 0:
        return (
            OUTCOME_NOT_IN_WORKSPACE,
            f"目标超出工作空间：{reach.summary()}，目标距离 {distance:.4f} m，"
            f"沿该方向差 {abs(margin):.4f} m（换方向或挪机器人，换初值没用）",
        )
    if best_residual > params.unreachable_residual:
        return (
            OUTCOME_POSE_NOT_REACHABLE,
            f"位置在可达范围内（余量 {margin:+.4f} m），但指定姿态达不到：最佳任务残差 "
            f"{best_residual:.4f} > {params.unreachable_residual}。"
            f"换个抓取姿态（例如侧向接近）再试 —— 位置可达不代表姿态可达",
        )
    return (
        OUTCOME_RESIDUAL_TOO_LARGE,
        f"差一点就能收敛：最佳任务残差 {best_residual:.2e}，高于要求 {params.residual_tol:.0e}。"
        f"换更多初值或放宽精度可能成功",
    )


def classify_all_rejected(candidates: list[PickCandidate]) -> str:
    """所有候选都被淘汰时的结论：单一原因就报那个原因，混合原因时优先报更严重的。"""
    reasons = {c.rejected_by for c in candidates}
    if len(reasons) == 1:
        return reasons.pop()
    return (
        OUTCOME_ALL_NEAR_SINGULAR
        if OUTCOME_ALL_NEAR_SINGULAR in reasons
        else OUTCOME_ALL_LIMIT_VIOLATED
    )


def reject_reason(candidates: list[PickCandidate]) -> str:
    """全被淘汰时的人话理由：每一类各有多少组。"""
    over = sum(1 for c in candidates if c.rejected_by == OUTCOME_ALL_LIMIT_VIOLATED)
    near = sum(1 for c in candidates if c.rejected_by == OUTCOME_ALL_NEAR_SINGULAR)
    return (
        f"{len(candidates)} 组解全部被淘汰：越关节限位 {over} 组、接近奇异 {near} 组。"
        f"机械臂能到达这个位姿，但没有一组是能安全执行的"
    )


def _task_error(
    robot: rtb.Robot, target: SE3, q: np.ndarray, task: PickTask
) -> tuple[float, float, float]:
    """返回 (任务残差, 位置误差, 姿态误差)。"""
    from spatialmath.base import tr2angvec

    reached = robots.end_pose(robot, q, task.frame)
    dp = np.asarray(target.t, dtype=float) - np.asarray(reached.t, dtype=float)
    position_error = float(np.linalg.norm(dp))
    angle, axis = tr2angvec(np.asarray(target.R) @ np.asarray(reached.R).T)
    orientation_error = float(abs(angle))
    return (
        float(np.linalg.norm(np.concatenate([dp, axis * angle]))),
        position_error,
        orientation_error,
    )
