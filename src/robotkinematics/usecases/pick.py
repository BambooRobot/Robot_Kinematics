"""@file pick.py
@brief 抓取任务流水线 —— 把七个知识点串成一条工序链。

这是本项目里唯一"总入口"式的用例。它回答的不是"某个数是多少"，而是
**"这次抓取能不能做，不能做差在哪"**。

七道工序（每一道对应第一章的一个知识点），以及每道工序由谁实现：

  ① 观测 → 本体位姿      位姿与坐标系、变换链        → spatialmath 的 SE3
  ② 可达性预筛            工作空间不是球              → 自研采样（库没有这个 API）
  ③ 多初值 IK             IK 的多解 / 无解 / 迭代     → robot.ikine_LM；二连杆另有闭式解
  ④ 雅可比体检            雅可比与奇异点              → robot.jacob0 + numpy SVD
  ⑤ 限位校验              真机约束（关节限位）        → robot.qlim
  ⑥ 解择优                冗余与零空间                → 自研评分（库不提供）
  ⑦ 微动可行性校验        雅可比干活：解一次小位移    → robot.jacob0 + numpy.linalg.solve

【为什么要分这么细】每一步都能独立失败、独立验证。失败时给出的是**有名字、有数字**的
结论（见 OUTCOMES），而不是一句"失败了" —— 这正是"项目"和"demo"的区别。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import roboticstoolbox as rtb
from spatialmath import SE3

from ..contracts import HeadingBlock, Report, TableBlock, TextBlock, TwoLinkChartBlock, fmt_vector
from ..core import robots, workspace
from ..core.planar2r import Planar2R

# ── 七类结论（一类成功 + 六类失败）─────────────────────────────────────────
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

# 择优评分权重：离限位余量 0.5、σ_min 0.3、离偏好位形 0.2
SCORE_WEIGHTS = (0.5, 0.3, 0.2)

RPY_ORDER = "zyx"  # 本项目约定：R = Rz(yaw)·Ry(pitch)·Rx(roll)
DEG = np.deg2rad


@dataclass(frozen=True)
class PickParams:
    """流水线参数（来自 configs/default.yaml 的 pick 段）。"""

    seeds: int = 24
    residual_tol: float = 1e-4  # 0.1 mm：比真机重复定位精度还严一档
    unreachable_residual: float = 0.02
    min_sigma: float = 0.02
    workspace_samples: int = 8000
    workspace_seed: int = 0
    limit_margin_deg: float = 0.0
    max_iter: int = 200  # 单次 ikine_LM 的迭代上限
    ik_tol: float = 1e-9  # 传给库的关节步长容差（不是位姿残差！见下）
    micro_step_m: float = 0.001  # 微动校验的末端步长（默认 1mm）
    max_joint_step_deg: float = 5.0  # 微动对应的关节角上限


@dataclass(frozen=True)
class PickTask:
    """一次抓取任务：抓哪里、用什么姿态、用哪台机器人。"""

    observation: np.ndarray  # 观测点（默认在相机坐标系下，见 mode）
    camera_xyz: tuple[float, float, float] = (0.30, 0.00, 0.60)
    camera_yaw_deg: float = 30.0
    orientation_rpy_deg: tuple[float, float, float] = (180.0, 0.0, 0.0)
    arm: str = "panda"  # "panda" | "planar"
    # 抓取用的末端帧：固定为**夹爪 TCP**（= 库的默认末端）。
    # ⚠️ 不能改成法兰：库的 `ikine_LM` 只对模型的默认末端求解，不接受 end 参数；
    #    而且物理上抓东西的本来就是夹爪，不是法兰。
    frame: str = robots.TCP
    mode: str = "camera"  # "camera"=观测在相机系；"base"=已在本体系
    prefer_config: np.ndarray | None = None  # 显式偏好；为空时用 q_home 兜底
    q_home: np.ndarray | None = None  # 初始位形；同时作为"就近择优"的默认偏好
    planar: Planar2R | None = None  # 二连杆的闭式解（arm="planar" 时必填）

    @property
    def is_planar(self) -> bool:
        return self.arm == "planar"

    def home(self, n_joints: int) -> np.ndarray:
        """初始位形：没给就是零位形。"""
        if self.q_home is None:
            return np.zeros(n_joints)
        return np.asarray(self.q_home, dtype=float).reshape(n_joints)

    def preferred(self, n_joints: int) -> np.ndarray:
        """择优时偏好靠近哪个位形：显式偏好优先，否则用初始位形。

        ⚠️ 这条默认很重要：不做"就近择优"的话，数值 IK 可能给出关节角转了两圈多的解
           —— 运动学上完全正确，真机上却是灾难。
        """
        if self.prefer_config is not None:
            return np.asarray(self.prefer_config, dtype=float).reshape(n_joints)
        return self.home(n_joints)


@dataclass(frozen=True)
class PickCandidate:
    """一个候选解，以及它为什么存活 / 为什么被淘汰。"""

    q: np.ndarray
    seed_index: int
    residual: float  # 任务残差（只算参与求解的那几维）
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
    reach: workspace.ReachEstimate | None = None


# ── 流水线 ──────────────────────────────────────────────────────────────────


def pick(robot: rtb.Robot, task: PickTask, params: PickParams) -> PickResult:
    steps: list[PickStep] = []

    # ① 观测 → 本体位姿
    target, step = _step_locate(task)
    steps.append(step)

    # ② 可达性预筛
    reach, step = _step_prescreen(robot, task, target, params)
    steps.append(step)

    # ③ 多初值 IK
    candidates, best_residual, step = _step_solve(robot, task, target, params)
    steps.append(step)

    if not candidates:
        outcome, reason = _classify_no_candidate(task, reach, target, best_residual, params)
        return PickResult(outcome, reason, target, None, (), tuple(steps), reach)

    # ④ 雅可比体检 + ⑤ 限位校验（一次遍历算出，作为两道工序报告）
    candidates, jacobian_step, limits_step = _step_inspect(robot, task, candidates, params)
    steps.extend((jacobian_step, limits_step))

    survivors = [c for c in candidates if c.survived]
    if not survivors:
        return PickResult(
            _classify_all_rejected(candidates),
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

    # ⑦ 微动可行性校验：对选中的那组做最后确认
    feasible, detail, step = _step_micro_motion(robot, task, best, params)
    steps.append(step)
    if not feasible:
        return PickResult(
            OUTCOME_MICRO_MOTION_INFEASIBLE,
            detail,
            target,
            None,
            tuple(candidates),
            tuple(steps),
            reach,
        )

    travel = float(np.linalg.norm(best.q - task.home(best.q.size)))
    reason = (
        f"从 {len(candidates)} 个候选解中选出评分最高的一组：离限位余量 "
        f"{np.rad2deg(best.limit_margin):.1f}°、σ_min={best.sigma_min:.4f}、"
        f"离初始位形 {travel:.3f} rad；任务残差 {best.residual:.2e}"
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
        point_base = np.asarray((T_base_camera * SE3(observation)).t, dtype=float)
        detail = (
            f"相机观测 {fmt_vector(observation)} → 本体 {fmt_vector(point_base)}"
            f"（^base T_camera · p_obs）"
        )
    else:
        point_base = observation
        detail = f"观测已在本体坐标系：{fmt_vector(point_base)}"

    if task.is_planar:
        # 二连杆只在 x-y 平面里动：z 与姿态都没有自由度，如实说明而不是假装约束满足
        ignored = float(point_base[2])
        target = SE3(point_base[0], point_base[1], 0.0)
        detail += (
            f"；二连杆只有平面两个自由度 → 投影到 x-y 平面，忽略 z={ignored:+.4f} m 与姿态要求"
        )
    else:
        roll, pitch, yaw = DEG(task.orientation_rpy_deg)
        target = SE3(point_base) * SE3.RPY(roll, pitch, yaw, order=RPY_ORDER)

    return target, PickStep(
        name="① 观测 → 本体位姿",
        detail=detail,
        numbers={
            "observation": observation,
            "target_position": np.asarray(target.t, dtype=float),
            "target_distance": float(np.linalg.norm(target.t)),
        },
    )


def _step_prescreen(
    robot: rtb.Robot, task: PickTask, target: SE3, params: PickParams
) -> tuple[workspace.ReachEstimate, PickStep]:
    """② 可达性预筛（知识点：工作空间不是球）。"""
    limits = robots.joint_limits(robot)
    reach = workspace.directional_reach(
        robot,
        np.asarray(target.t, dtype=float),
        samples=params.workspace_samples,
        seed=params.workspace_seed,
        limits=limits,
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
    robot: rtb.Robot, task: PickTask, target: SE3, params: PickParams
) -> tuple[list[PickCandidate], float, PickStep]:
    """③ 多初值 IK（知识点：IK 的多解 / 无解 / 数值迭代）。

    初值集合 = 初始位形 + **二连杆的闭式解**（有解析解就该拿它当暖启动）+ 随机构造。
    求解一律交给库的 `ikine_LM`；二连杆的闭式解另用来反向校验数值解。
    """
    n = robot.n
    candidates: list[PickCandidate] = []
    best_residual = float("inf")

    if task.is_planar:
        # 二连杆：候选解直接来自闭式解 —— 它同时给出"肘上/肘下"两组，这正是本节考点。
        # ⚠️ 不用库的 ikine_LM：给平面机器人传 mask（"只管位置"）会因库内部权重矩阵
        #    与被筛过后的误差向量维数不匹配而抛错（实测 RTB 1.4.3）；不传 mask 又要求
        #    它同时满足平面机械臂根本做不到的姿态。所以这一处的分工是刻意的。
        for index, sol in enumerate(_analytic_seeds(task, target)):
            q = np.asarray(sol, dtype=float).reshape(n)
            residual, pos_err, ori_err = _task_error(robot, target, q, task)
            best_residual = min(best_residual, residual)
            candidates.append(_candidate(q, index, residual, pos_err, ori_err))
        detail = (
            f"闭式解（余弦定理）给出 {len(candidates)} 组解，"
            f"最佳任务残差 {best_residual:.2e}（解析解是精确解，只受浮点精度限制）"
        )
        return (
            candidates,
            best_residual,
            PickStep(
                name="③ 多初值 IK",
                detail=detail,
                numbers={
                    "source": "闭式解（平面机械臂；库的 IK 无法表达“只管位置”）",
                    "solutions": len(candidates),
                    "best_residual": best_residual,
                    "residual_tol": params.residual_tol,
                },
            ),
        )

    # Panda：多初值 + 库的 ikine_LM（LM 迭代）
    seeds = [task.home(n)]
    rng = np.random.default_rng(params.workspace_seed + 1)
    seeds += [rng.uniform(-1.5, 1.5, size=n) for _ in range(max(params.seeds - 1, 0))]

    for index, seed in enumerate(seeds):
        # ⚠️ 不传 end：库的 IK 只对模型默认末端求解。抓取的末端就该是夹爪（见 PickTask.frame）
        # ⚠️ 库的 `tol` 是**关节步长**的收敛判据，位姿残差是结果而不是输入：
        #    用默认值它会停在 ~1e-6 的步长上，位姿残差能到 0.8mm；传 1e-9 才压到 1e-7 量级。
        solution = robot.ikine_LM(
            target,
            q0=np.asarray(seed, dtype=float).reshape(n),
            ilimit=params.max_iter,
            tol=params.ik_tol,
        )
        q = np.asarray(solution.q, dtype=float).reshape(n)
        residual, pos_err, ori_err = _task_error(robot, target, q, task)
        best_residual = min(best_residual, residual)
        if residual <= params.residual_tol:
            candidates.append(_candidate(q, index, residual, pos_err, ori_err))

    detail = (
        f"{len(seeds)} 个初值 → 收敛 {len(candidates)} 个"
        f"（判据：任务残差 < {params.residual_tol:.0e}，最佳 {best_residual:.2e}）"
    )
    return (
        candidates,
        best_residual,
        PickStep(
            name="③ 多初值 IK",
            detail=detail,
            numbers={
                "seeds": len(seeds),
                "converged": len(candidates),
                "best_residual": best_residual,
                "residual_tol": params.residual_tol,
            },
        ),
    )


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


def _step_inspect(
    robot: rtb.Robot, task: PickTask, candidates: list[PickCandidate], params: PickParams
) -> tuple[list[PickCandidate], PickStep, PickStep]:
    """④ 雅可比体检 与 ⑤ 限位校验（一次遍历算出，作为两道工序分别报告）。

    两者的物理含义完全不同：一个是"这组解好不好动"，一个是"真机能不能摆出来"。
    """
    from ..core.singularity import analyze  # 局部导入避免循环

    limits = robots.joint_limits(robot)
    margin_required = DEG(params.limit_margin_deg)
    inspected: list[PickCandidate] = []
    near_singular = over_limit = 0
    sigma_values: list[float] = []

    for candidate in candidates:
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
                score=_score_with_margin(
                    candidate.q, report.sigma_min, report.sigma_max, limit_margin, task
                ),
                rejected_by=rejected,
            )
        )

    jacobian_step = PickStep(
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
    limits_step = PickStep(
        name="⑤ 限位校验",
        detail=(
            "该模型没有限位信息，跳过"
            if limits is None
            else f"越限位 {over_limit} 组（余量要求 {params.limit_margin_deg:.1f}°）"
        ),
        numbers={"over_limit": over_limit, "checked": limits is not None},
    )
    return inspected, jacobian_step, limits_step


def _step_micro_motion(
    robot: rtb.Robot, task: PickTask, best: PickCandidate, params: PickParams
) -> tuple[bool, str, PickStep]:
    """⑦ 微动可行性校验：用雅可比解一次小位移（知识点：雅可比干活）。

    抓取前总要"往下压一点再合爪"。这一步就是在算：这 1mm 压下去，关节要动多少？
    如果算出来的关节角大得离谱（接近奇异点），这组解在实际中就不能用。
    """
    J = np.asarray(robot.jacob0(best.q), dtype=float)
    rows = 2 if task.is_planar else 6
    J = J[:rows, :]
    delta = np.zeros(rows)
    delta[2 if rows > 2 else 0] = -params.micro_step_m  # 平面情形退化为沿 x 的 1mm

    # 阻尼最小二乘求 dq —— 这里也交给 numpy 解线性方程组
    dq = np.linalg.solve(J @ J.T + (0.05**2) * np.eye(rows), delta)
    dq = J.T @ dq

    joint_step_deg = float(np.rad2deg(np.linalg.norm(dq)))
    feasible = joint_step_deg <= params.max_joint_step_deg
    detail = (
        f"末端微动 {params.micro_step_m * 1000:.1f} mm → 关节需动 {joint_step_deg:.3f}°"
        f"（上限 {params.max_joint_step_deg:.1f}°）：{'可行' if feasible else '**不可行**'}"
    )
    return (
        feasible,
        detail,
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


def _task_error(
    robot: rtb.Robot, target: SE3, q: np.ndarray, task: PickTask
) -> tuple[float, float, float]:
    """返回 (任务残差, 位置误差, 姿态误差)。

    ⚠️ 不能拿完整 6 维位姿误差来判定：二连杆只有平面两个自由度，姿态误差恒为 ~1.45 rad，
       用它判定会把"位置解得很好"误判成失败。所以任务残差只统计 mask 打开的那几维。
    """
    reached = robots.end_pose(robot, q, task.frame)
    dp = np.asarray(target.t, dtype=float) - np.asarray(reached.t, dtype=float)
    position_error = float(np.linalg.norm(dp))

    if task.is_planar:
        return position_error, position_error, 0.0

    from spatialmath.base import tr2angvec

    angle, axis = tr2angvec(np.asarray(target.R) @ np.asarray(reached.R).T)
    orientation_error = float(abs(angle))
    return (
        float(np.linalg.norm(np.concatenate([dp, axis * angle]))),
        position_error,
        orientation_error,
    )


def _analytic_seeds(task: PickTask, target: SE3) -> list[np.ndarray]:
    """二连杆的闭式解作为暖启动 —— 有解析解就该用。"""
    if not task.is_planar or task.planar is None:
        return []
    return [np.array(sol) for sol in task.planar.ik(float(target.t[0]), float(target.t[1]))]


def _classify_no_candidate(
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
        if task.is_planar:
            return (
                OUTCOME_RESIDUAL_TOO_LARGE,
                f"位置在可达范围内（余量 {margin:+.4f} m）却没解出来：最佳任务残差 "
                f"{best_residual:.4f} > {params.unreachable_residual}，属于数值求解问题而非物理限制",
            )
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


def _score_with_margin(
    q: np.ndarray, sigma_min: float, sigma_max: float, limit_margin: float, task: PickTask
) -> float:
    """解择优的评分：限位余量 + σ_min + 离偏好位形，各项归一化后加权。

    限位项拿满分的两种情况：没有限位信息（inf），或余量 ≥ 45°。
    """
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


# ── 报告 ────────────────────────────────────────────────────────────────────


def to_report(task: PickTask, result: PickResult) -> Report:
    """把任务结果渲染成报告（终端 / JSON 都由适配器决定）。"""
    blocks: list = [
        TextBlock.of(
            f"机械臂：{task.arm}"
            + ("（末端：夹爪 TCP）" if not task.is_planar else "（二连杆，末端=tool）"),
            f"目标位姿：位置 {fmt_vector(np.asarray(result.target.t, dtype=float))}",
            f"结论：{result.outcome}",
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
    if task.is_planar and result.q is not None:
        arm = task.planar or Planar2R()
        blocks.append(
            TwoLinkChartBlock(
                points=arm.joint_points(float(result.q[0]), float(result.q[1])),
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
            "frame": task.frame,
            "target_position": np.asarray(result.target.t, dtype=float),
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
