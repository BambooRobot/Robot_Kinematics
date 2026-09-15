"""@file pick.py
@brief 抓取任务流水线 —— 把七个知识点串成一条工序链。

这是本项目里唯一"总入口"式的用例。它回答的不是"某个数是多少"，而是
**"这次抓取能不能做，不能做差在哪"**。

七道工序（每一道对应第一章的一个知识点）：

  ① 观测 → 本体位姿      位姿与坐标系、变换链
  ② 可达性预筛            真机工作空间（不是球）
  ③ 多初值 IK              逆运动学（多解 / 无解 / 数值迭代）
  ④ 雅可比体检            雅可比与奇异点
  ⑤ 限位校验              真机约束（关节限位）
  ⑥ 解择优                冗余与零空间
  ⑦ 微动可行性校验        雅可比作为"干活的工具"：对选中的解解一次小位移

【为什么要分这么细】每一步都能独立失败、独立验证。失败时给出的是**有名字、有数字**的
结论（见 OUTCOMES），而不是一句"失败了" —— 这正是"项目"和"demo"的区别。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..contracts import HeadingBlock, Report, TableBlock, TextBlock, TwoLinkChartBlock, fmt_vector
from ..core.chain import SerialChain
from ..core.ik_solvers import pose_error, random_seeds, solve_dls
from ..core.planar2r import Planar2R
from ..core.rotations import rpy_to_matrix
from ..core.se3 import SE3
from ..core.singularity import analyze, describe_worst_direction
from ..core.workspace import ReachEstimate, directional_reach

DEG = np.deg2rad

# ── 六类结论 ────────────────────────────────────────────────────────────────
OUTCOME_REACHABLE = "可执行"
OUTCOME_NOT_IN_WORKSPACE = "不在工作空间"
OUTCOME_POSE_NOT_REACHABLE = "姿态不可达"
OUTCOME_RESIDUAL_TOO_LARGE = "收敛精度不足"
OUTCOME_ALL_LIMIT_VIOLATED = "所有解越关节限位"
OUTCOME_ALL_NEAR_SINGULAR = "所有解接近奇异"
OUTCOME_MICRO_MOTION_INFEASIBLE = "微动不可行"

OUTCOMES = (
    OUTCOME_REACHABLE,
    OUTCOME_NOT_IN_WORKSPACE,
    OUTCOME_POSE_NOT_REACHABLE,
    OUTCOME_RESIDUAL_TOO_LARGE,
    OUTCOME_ALL_LIMIT_VIOLATED,
    OUTCOME_ALL_NEAR_SINGULAR,
    OUTCOME_MICRO_MOTION_INFEASIBLE,
)

# 择优评分权重：离限位余量 0.5、σ_min 0.3、与偏好姿态接近度 0.2
SCORE_WEIGHTS = (0.5, 0.3, 0.2)


@dataclass(frozen=True)
class PickParams:
    """流水线参数（来自 configs/default.yaml 的 pick 段）。"""

    seeds: int = 24
    residual_tol: float = 1e-6
    unreachable_residual: float = 0.02
    min_sigma: float = 0.02
    workspace_samples: int = 8000
    workspace_seed: int = 0
    limit_margin_deg: float = 0.0
    max_iter: int = 200  # 单次 IK 的迭代上限
    micro_step_m: float = 0.001  # 微动校验的末端步长（默认 1mm）
    max_joint_step_deg: float = 5.0  # 微动对应的关节角上限（超过就说明接近奇异了）


@dataclass(frozen=True)
class PickTask:
    """一次抓取任务：抓哪里、用什么姿态、用哪台机器人。"""

    observation: np.ndarray  # 相机坐标系下的观测点 [m]
    camera_xyz: tuple[float, float, float]
    camera_yaw_deg: float
    orientation_rpy_deg: tuple[float, float, float] = (180.0, 0.0, 0.0)
    arm: str = "panda"  # "panda" | "planar"
    limits: np.ndarray | None = None  # (n, 2) 关节限位；None 表示不检查
    planar: Planar2R | None = None
    mode: str = "base"  # "base"：观测已在 base frame；"camera"：观测在相机 frame
    prefer_config: np.ndarray | None = None  # 显式偏好；为空时用 q_home 兜底
    q_home: np.ndarray | None = None  # 初始位形；同时作为"就近择优"的默认偏好

    def home(self, n_joints: int) -> np.ndarray:
        """初始位形：没给就是零位形。"""
        if self.q_home is None:
            return np.zeros(n_joints)
        return np.asarray(self.q_home, dtype=float).reshape(n_joints)

    def preferred(self, n_joints: int) -> np.ndarray:
        """择优时偏好靠近哪个位形：显式偏好优先，否则用初始位形。

        ⚠️ 这条默认很重要：不做"就近择优"的话，数值 IK 可能给出关节角转了两圈多的解
           —— 运动学上完全正确，真机上却是灾难。实测二连杆就出现过 q=(-13.6, 15.1) rad。
        """
        if self.prefer_config is not None:
            return np.asarray(self.prefer_config, dtype=float).reshape(n_joints)
        return self.home(n_joints)

    @property
    def task_rows(self) -> slice:
        """参与求解的位姿维度：二连杆只动得了平面里的两个方向。"""
        return slice(0, 2) if self.arm == "planar" else slice(0, 6)


@dataclass(frozen=True)
class PickCandidate:
    """一个候选解，以及它为什么存活 / 为什么被淘汰。"""

    q: np.ndarray
    seed_index: int
    residual: float
    position_error: float
    orientation_error: float
    sigma_min: float
    condition: float
    manipulability: float
    limit_margin: float  # 离最近的关节限位还有多少 [rad]；无限制信息时为 inf
    score: float = 0.0
    rejected_by: str = ""  # "" = 存活

    @property
    def survived(self) -> bool:
        return not self.rejected_by


@dataclass(frozen=True)
class PickStep:
    """一道工序的结果：做了什么、结论、关键数字。"""

    name: str
    detail: str
    numbers: dict = field(default_factory=dict)


@dataclass(frozen=True)
class PickResult:
    outcome: str
    reason: str
    target: SE3
    q: np.ndarray | None
    candidates: tuple[PickCandidate, ...]
    steps: tuple[PickStep, ...]
    reach: ReachEstimate | None = None


# ── 流水线 ──────────────────────────────────────────────────────────────────


def pick(chain: SerialChain, task: PickTask, params: PickParams) -> PickResult:
    steps: list[PickStep] = []

    # ① 观测 → 本体位姿
    target, step = _step_locate(task)
    steps.append(step)

    # ② 可达性预筛
    reach, step = _step_prescreen(chain, task, target, params)
    steps.append(step)

    # ③ 多初值 IK
    candidates, _seeds, best_residual, step = _step_solve(chain, task, target, params)
    steps.append(step)

    if not candidates:
        outcome, reason = _classify_no_candidate(task, reach, target, best_residual, params)
        return PickResult(outcome, reason, target, None, (), tuple(steps), reach)

    # ④ 雅可比体检 + ⑦ 限位检查（都是对候选逐个做的，合并成一次遍历）
    candidates, jacobian_step, limits_step = _step_inspect(chain, task, candidates, params)
    steps.append(jacobian_step)

    # ⑤ 限位校验（与 ④ 一起遍历算出，但作为独立工序报告）
    steps.append(limits_step)

    survivors = [c for c in candidates if c.survived]
    if not survivors:
        outcome = _classify_all_rejected(candidates)
        return PickResult(
            outcome,
            _reject_reason(candidates),
            target,
            None,
            tuple(candidates),
            tuple(steps),
            reach,
        )

    # ⑥ 解择优
    best = max(survivors, key=lambda c: c.score)
    steps.append(_step_choose(candidates, best, task))

    # ⑦ 微动可行性校验：对选中的那组做最后确认（能不能"压下去 1mm 再合爪"）
    feasible, micro, step = _step_micro_motion(chain, task, best, params)
    steps.append(step)
    if not feasible:
        return PickResult(
            OUTCOME_MICRO_MOTION_INFEASIBLE,
            micro["detail"],
            target,
            None,
            tuple(candidates),
            tuple(steps),
            reach,
        )

    travel = float(np.linalg.norm(best.q - task.home(best.q.size)))
    pose_note = (
        f"位置误差 {best.position_error:.2e} m、姿态误差 {best.orientation_error:.2e} rad"
        if task.arm != "planar"
        else f"位置误差 {best.position_error:.2e} m（平面机械臂无姿态自由度，姿态不参与判定）"
    )
    reason = (
        f"从 {len(candidates)} 个候选解中选出评分最高的一组：离限位余量 "
        f"{np.rad2deg(best.limit_margin):.1f}°、σ_min={best.sigma_min:.4f}、"
        f"离初始位形 {travel:.3f} rad；任务残差 {best.residual:.2e}（{pose_note}）"
    )
    return PickResult(
        OUTCOME_REACHABLE, reason, target, best.q, tuple(candidates), tuple(steps), reach
    )


# ── 各道工序 ────────────────────────────────────────────────────────────────


def _step_locate(task: PickTask) -> tuple[SE3, PickStep]:
    """① 观测 → 本体位姿（知识点：位姿与坐标系、变换链）。"""
    observation = np.asarray(task.observation, dtype=float).reshape(3)

    if task.mode == "camera":
        T_base_camera = SE3.Trans(*task.camera_xyz) * SE3.Rz(DEG(task.camera_yaw_deg))
        point_base = T_base_camera.apply(observation)
        detail = (
            f"相机观测 {fmt_vector(observation)} → 本体 {fmt_vector(point_base)}"
            f"（^base T_camera · p_obs）"
        )
    else:
        point_base = observation
        detail = f"观测已在本体坐标系：{fmt_vector(point_base)}"

    R = rpy_to_matrix(*DEG(task.orientation_rpy_deg))

    if task.arm == "planar":
        # 二连杆只在 x-y 平面里动：z 与姿态都没有自由度，如实说明而不是假装约束满足
        projected = np.array([point_base[0], point_base[1], 0.0])
        ignored = float(point_base[2])
        target = SE3(np.eye(3), projected)
        detail += (
            f"；二连杆只有平面两个自由度 → 投影到 x-y 平面，忽略 z={ignored:+.4f} m 与姿态要求"
        )
    else:
        target = SE3(R, point_base)

    return target, PickStep(
        name="① 观测 → 本体位姿",
        detail=detail,
        numbers={
            "observation": observation,
            "target_position": target.t,
            "target_distance": float(np.linalg.norm(target.t)),
            "target_R": target.R,
        },
    )


def _step_prescreen(
    chain: SerialChain, task: PickTask, target: SE3, params: PickParams
) -> tuple[ReachEstimate, PickStep]:
    """② 可达性预筛（知识点：工作空间不是球）。"""
    reach = directional_reach(
        chain,
        target.t,
        samples=params.workspace_samples,
        seed=params.workspace_seed,
        limits=task.limits,
    )
    distance = float(np.linalg.norm(target.t))
    margin = reach.margin(distance)
    inside = margin >= 0.0
    detail = f"{reach.summary()}；目标距离 {distance:.4f} m → " + (
        f"在界内（余量 {margin:+.4f} m）" if inside else f"**超出 {abs(margin):.4f} m**"
    )
    return reach, PickStep(
        name="② 可达性预筛",
        detail=detail,
        numbers={
            "direction_reach": reach.radius,
            "target_distance": distance,
            "margin": margin,
            "inside": inside,
        },
    )


def _step_solve(
    chain: SerialChain, task: PickTask, target: SE3, params: PickParams
) -> tuple[list[PickCandidate], list[np.ndarray], float, PickStep]:
    """③ 多初值 IK（知识点：逆运动学的多解 / 无解 / 数值迭代）。"""
    seeds = [task.home(chain.n_joints)]
    seeds += _analytic_seeds(task, target)
    if params.seeds > 1:
        seeds += random_seeds(chain, params.seeds - 1, seed=params.workspace_seed + 1, scale=1.5)

    candidates: list[PickCandidate] = []
    best_residual = float("inf")
    solved = 0
    for index, seed in enumerate(seeds):
        result = solve_dls(
            chain,
            target,
            seed,
            max_iter=params.max_iter,
            tol=params.residual_tol,
            rows=task.task_rows,
        )
        task_residual = _task_residual(chain, target, result.q, task.task_rows)
        best_residual = min(best_residual, task_residual)
        if result.success:
            solved += 1
            candidates.append(
                PickCandidate(
                    q=result.q,
                    seed_index=index,
                    residual=task_residual,
                    position_error=result.position_error,
                    orientation_error=result.orientation_error,
                    sigma_min=float("nan"),
                    condition=float("nan"),
                    manipulability=float("nan"),
                    limit_margin=float("inf"),
                )
            )

    # 二连杆：数值流水线的解要与闭式解对上（对照组的意义就在这里）
    analytic_note = ""
    if task.arm == "planar" and task.planar is not None and candidates:
        analytic = task.planar.ik(float(target.t[0]), float(target.t[1]))
        if analytic:
            matches = [
                min(float(np.linalg.norm(c.q - np.array(sol))) for sol in analytic)
                for c in candidates
            ]
            analytic_note = f"；与闭式解最近距离 {min(matches):.2e} rad"

    detail = (
        f"{len(seeds)} 个初值 → 收敛 {solved} 个"
        f"（判据：残差 < {params.residual_tol:.0e}，最佳残差 {best_residual:.2e}）{analytic_note}"
    )
    return (
        candidates,
        seeds,
        best_residual,
        PickStep(
            name="③ 多初值 IK",
            detail=detail,
            numbers={
                "seeds": len(seeds),
                "converged": solved,
                "best_residual": best_residual,
                "residual_tol": params.residual_tol,
            },
        ),
    )


def _task_residual(chain: SerialChain, target: SE3, q: np.ndarray, rows: slice) -> float:
    """任务残差：只统计"这次任务真正要求的那几维"的误差。

    ⚠️ 不能拿完整 6 维位姿误差来判定：二连杆只有平面两个自由度，姿态误差恒为 ~1.45 rad，
       用它判定会把"位置解得很好"误判成失败（实测踩过）。位置/姿态误差仍然如实分开报告。
    """
    return float(np.linalg.norm(pose_error(chain.fk(q), target)[rows]))


def _analytic_seeds(task: PickTask, target: SE3) -> list[np.ndarray]:
    """二连杆的闭式解作为暖启动 —— 有解析解就该用，这也是"对照组"的意义。"""
    if task.arm != "planar" or task.planar is None:
        return []
    return [np.array(sol) for sol in task.planar.ik(float(target.t[0]), float(target.t[1]))]


def _step_inspect(
    chain: SerialChain, task: PickTask, candidates: list[PickCandidate], params: PickParams
) -> tuple[list[PickCandidate], PickStep, PickStep]:
    """④ 雅可比体检 与 ⑦ 限位校验（知识点：奇异点、真机约束）。

    两者都是"逐个候选"的检查，所以放在一次遍历里算；但**作为两道工序分别报告** ——
    它们的物理含义完全不同：一个是"这组解好不好动"，一个是"真机能不能摆出来"。
    """
    margin_required = DEG(params.limit_margin_deg)
    inspected: list[PickCandidate] = []
    near_singular = 0
    over_limit = 0

    for candidate in candidates:
        report = analyze(chain.jacobian(candidate.q))
        limit_margin = _limit_margin(candidate.q, task.limits)

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
                score=_score(candidate.q, report.sigma_min, report.sigma_max, task),
                rejected_by=rejected,
            )
        )

    sigma_values = [c.sigma_min for c in inspected]
    worst = min(inspected, key=lambda c: c.sigma_min)
    jacobian_step = PickStep(
        name="④ 雅可比体检",
        detail=(
            f"σ_min 区间 [{min(sigma_values):.4f}, {max(sigma_values):.4f}]，"
            f"最差一组的最难方向：{describe_worst_direction(analyze(chain.jacobian(worst.q)).worst_direction)}；"
            f"接近奇异 {near_singular} 组"
        ),
        numbers={
            "sigma_min_range": (min(sigma_values), max(sigma_values)),
            "near_singular": near_singular,
        },
    )
    limits_step = PickStep(
        name="⑤ 限位校验",
        detail=(
            "无限位信息，跳过"
            if task.limits is None
            else f"越限位 {over_limit} 组（余量要求 {params.limit_margin_deg:.1f}°）"
        ),
        numbers={"over_limit": over_limit, "checked": task.limits is not None},
    )
    return inspected, jacobian_step, limits_step


def _step_micro_motion(
    chain: SerialChain, task: PickTask, best: PickCandidate, params: PickParams
) -> tuple[bool, dict, PickStep]:
    """⑦ 微动可行性校验：用雅可比解一次小位移（知识点：雅可比干活）。

    抓取前总要"往下压一点再合爪"。这一步就是在算：这 1mm 压下去，关节要动多少？
    如果算出来的关节角大得离谱（接近奇异点），这组解在实际中就不能用。
    """
    J = chain.jacobian(best.q)[task.task_rows, :]
    rows = J.shape[0]
    delta = np.zeros(rows)
    delta[2 if rows > 2 else 0] = -params.micro_step_m  # 平面情形退化为沿 x/y 的 1mm

    # 阻尼最小二乘求 dq（与 IK 同一套办法，这里只走一步）
    JJt = J @ J.T
    dq = J.T @ np.linalg.solve(JJt + (0.05**2) * np.eye(rows), delta)

    joint_step_deg = float(np.rad2deg(np.linalg.norm(dq)))
    feasible = joint_step_deg <= params.max_joint_step_deg
    detail = (
        f"末端微动 {params.micro_step_m * 1000:.1f} mm → 关节需动 {joint_step_deg:.3f}°"
        f"（上限 {params.max_joint_step_deg:.1f}°）：{'可行' if feasible else '**不可行**'}"
    )
    return (
        feasible,
        {"detail": detail, "joint_step_deg": joint_step_deg},
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
        ),
    )


def _step_choose(candidates: list[PickCandidate], best: PickCandidate, task: PickTask) -> PickStep:
    """⑥ 解择优（知识点：冗余与零空间）。"""
    survivors = [c for c in candidates if c.survived]
    detail = (
        f"存活 {len(survivors)} 组，按「离限位余量 {SCORE_WEIGHTS[0]} + σ_min {SCORE_WEIGHTS[1]}"
        f" + 离偏好位形 {SCORE_WEIGHTS[2]}」评分择优"
        f"（偏好位形 = {'显式指定' if task.prefer_config is not None else '初始位形'}）："
        f"选中第 {best.seed_index + 1} 个初值的那组"
    )
    return PickStep(
        name="⑥ 解择优",
        detail=detail,
        numbers={
            "survivors": len(survivors),
            "chosen_seed_index": best.seed_index,
            "score": best.score,
            "weights": SCORE_WEIGHTS,
        },
    )


# ── 判定与打分 ──────────────────────────────────────────────────────────────


def _classify_no_candidate(
    task: PickTask,
    reach: ReachEstimate,
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
    label = "最佳任务残差"
    if best_residual > params.unreachable_residual:
        if task.arm == "planar":
            # 平面机械臂没有姿态要求，解不出来就是位置解不出来
            return (
                OUTCOME_RESIDUAL_TOO_LARGE,
                f"位置在可达范围内（余量 {margin:+.4f} m）却没解出来：{label} "
                f"{best_residual:.4f} > {params.unreachable_residual}，属于数值求解问题而非物理限制",
            )
        return (
            OUTCOME_POSE_NOT_REACHABLE,
            f"位置在可达范围内（余量 {margin:+.4f} m），但指定姿态达不到：{label} "
            f"{best_residual:.4f} > {params.unreachable_residual}。"
            f"换个抓取姿态（例如侧向接近）再试 —— 位置可达不代表姿态可达",
        )
    return (
        OUTCOME_RESIDUAL_TOO_LARGE,
        f"差一点就能收敛：{label} {best_residual:.2e}，高于要求 {params.residual_tol:.0e}。"
        f"换更多初值或放宽精度可能成功",
    )


def _classify_all_rejected(candidates: list[PickCandidate]) -> str:
    reasons = {c.rejected_by for c in candidates}
    if len(reasons) == 1:
        return reasons.pop()
    return (
        OUTCOME_ALL_NEAR_SINGULAR
        if OUTCOME_ALL_NEAR_SINGULAR in reasons
        else OUTCOME_ALL_LIMIT_VIOLATED
    )


def _reject_reason(candidates: list[PickCandidate]) -> str:
    over = sum(1 for c in candidates if c.rejected_by == OUTCOME_ALL_LIMIT_VIOLATED)
    near = sum(1 for c in candidates if c.rejected_by == OUTCOME_ALL_NEAR_SINGULAR)
    return (
        f"{len(candidates)} 组解全部被淘汰：越关节限位 {over} 组、接近奇异 {near} 组。"
        f"机械臂能到达这个位姿，但没有一组是能安全执行的"
    )


def _limit_margin(q: np.ndarray, limits: np.ndarray | None) -> float:
    """离最近关节限位还有多少 [rad]；越界时为负。无限制信息时返回 inf。"""
    if limits is None:
        return float("inf")
    limits = np.asarray(limits, dtype=float)
    lower = q - limits[:, 0]
    upper = limits[:, 1] - q
    return float(min(lower.min(), upper.min()))


def _score(q: np.ndarray, sigma_min: float, sigma_max: float, task: PickTask) -> float:
    """择优评分：各项都归一化到 [0,1]，权重写在 SCORE_WEIGHTS 里（可审计）。"""
    w_limit, w_sigma, w_prefer = SCORE_WEIGHTS

    margin = _limit_margin(q, task.limits)
    limit_score = 0.5 if not np.isfinite(margin) else float(np.clip(margin / (np.pi / 4), 0.0, 1.0))
    sigma_score = 0.0 if sigma_max <= 0 else float(np.clip(sigma_min / sigma_max, 0.0, 1.0))

    # 偏好项：离"偏好位形"（显式偏好，否则初始位形）越近越好。
    # 没有这一项的话，数值解可能给出关节转两圈多的解 —— 运动学对、真机不能用。
    distance = float(np.linalg.norm(q - task.preferred(q.size)))
    prefer_score = float(np.clip(1.0 - distance / np.pi, 0.0, 1.0))

    return w_limit * limit_score + w_sigma * sigma_score + w_prefer * prefer_score


# ── 报告 ────────────────────────────────────────────────────────────────────


def to_report(task: PickTask, result: PickResult) -> Report:
    """把任务结果渲染成报告（终端 / JSON 都由适配器决定）。"""
    outcome_line = f"结论：{result.outcome}"
    blocks: list = [
        TextBlock.of(
            f"机械臂：{task.arm}",
            f"目标位姿：位置 {fmt_vector(result.target.t)}",
            outcome_line,
            f"原因：{result.reason}",
        ),
        HeadingBlock("七道工序"),
        TableBlock(
            ("工序", "结论", "关键数字"),
            tuple((s.name, s.detail, _numbers_summary(s.numbers)) for s in result.steps),
            ("<", "<", "<"),
        ),
    ]

    if result.q is not None:
        blocks += [
            HeadingBlock("输出"),
            TextBlock.of(
                f"关节解 q = {fmt_vector(result.q)} rad，{fmt_vector(np.rad2deg(result.q), 1)} deg"
            ),
        ]
    if len(result.candidates) > 1:
        blocks += [
            HeadingBlock("候选解对比"),
            TableBlock(
                ("初值", "残差", "σ_min", "离限位", "评分", "结论"),
                tuple(
                    (
                        f"#{c.seed_index + 1}",
                        f"{c.residual:.2e}",
                        f"{c.sigma_min:.4f}" if np.isfinite(c.sigma_min) else "—",
                        f"{np.rad2deg(c.limit_margin):.1f}°"
                        if np.isfinite(c.limit_margin)
                        else "—",
                        f"{c.score:.3f}",
                        c.rejected_by or "存活",
                    )
                    for c in sorted(result.candidates, key=lambda c: -c.score)
                ),
                ("<", ">", ">", ">", ">", "<"),
            ),
        ]
    if task.arm == "planar" and result.q is not None:
        blocks.append(
            TwoLinkChartBlock(
                points=_planar_points(task, result.q),
                label="机械臂姿态",
            )
        )

    return Report(
        title="抓取任务流水线",
        blocks=tuple(blocks),
        fields={
            "outcome": result.outcome,
            "reason": result.reason,
            "arm": task.arm,
            "target_position": result.target.t,
            "q": result.q,
            "q_deg": None if result.q is None else np.rad2deg(result.q),
            "steps": [
                {"name": s.name, "detail": s.detail, "numbers": _jsonable(s.numbers)}
                for s in result.steps
            ],
            "candidates": [
                {
                    "seed_index": c.seed_index + 1,
                    "residual": c.residual,
                    "sigma_min": c.sigma_min,
                    "condition": c.condition,
                    "limit_margin_deg": float(np.rad2deg(c.limit_margin))
                    if np.isfinite(c.limit_margin)
                    else None,
                    "score": c.score,
                    "rejected_by": c.rejected_by,
                }
                for c in result.candidates
            ],
        },
    )


def _planar_points(task: PickTask, q: np.ndarray) -> tuple:
    arm = task.planar or Planar2R()
    return arm.joint_points(float(q[0]), float(q[1]))


def _numbers_summary(numbers: dict) -> str:
    """把关键数字压成一行 —— 小量用科学计数法，否则 1e-8 会显示成 0.0000。"""
    parts = []
    for key, value in numbers.items():
        parts.append(f"{key}={_format_number(value)}")
        if len(parts) >= 3:
            break
    return "，".join(parts)


def _format_number(value) -> str:
    if isinstance(value, np.ndarray):
        return fmt_vector(value, 3)
    if isinstance(value, tuple):
        return "[" + ", ".join(_format_number(v) for v in value) + "]"
    if isinstance(value, bool):
        return "是" if value else "否"
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, float):
        if value == 0:
            return "0"
        return f"{value:.2e}" if abs(value) < 1e-3 else f"{value:.4f}"
    return str(value)


def _jsonable(numbers: dict) -> dict:
    out = {}
    for key, value in numbers.items():
        if isinstance(value, np.ndarray):
            out[key] = value.tolist()
        elif isinstance(value, tuple):
            out[key] = [float(v) for v in value]
        elif isinstance(value, (np.floating, np.integer)):
            out[key] = float(value)
        else:
            out[key] = value
    return out
