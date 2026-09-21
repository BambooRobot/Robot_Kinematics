"""@file pick.py

@brief 抓取任务流水线（**门面**）：七道工序串成一条链、跑它、把结论报出来。

这是本项目唯一的对外用例。它回答的不是"某个数是多少"，而是
**"这次抓取能不能做，不能做差在哪"**。机器人固定为 Franka Panda。

【本模块只有两件事】把工序连起来（`PIPELINE`）、按序执行并在失败处停下（`pick`）。
其余三件事分给同目录的三个模块：

  * `pick_types.py`  —— 词汇与数据契约：结论常量、Task / Params / Candidate / Result
  * `pick_steps.py`  —— 七道工序各自**怎么算**，以及失败时**怎么归类**
  * `pick_report.py` —— 报什么（`Report` 的结构；怎么画是适配器的事）
"""

from __future__ import annotations

import numpy as np
import roboticstoolbox as rtb

from .pick_report import to_report
from .pick_steps import (
    PIPELINE,
    classify_all_rejected,
    classify_no_candidate,
    reject_reason,
    score_candidate,
    step_choose,
    step_inspect,
    step_locate,
    step_micro_motion,
    step_prescreen,
    step_solve,
)
from .pick_types import (
    OUTCOME_ALL_LIMIT_VIOLATED,
    OUTCOME_ALL_NEAR_SINGULAR,
    OUTCOME_MICRO_MOTION_INFEASIBLE,
    OUTCOME_NOT_IN_WORKSPACE,
    OUTCOME_POSE_NOT_REACHABLE,
    OUTCOME_REACHABLE,
    OUTCOME_RESIDUAL_TOO_LARGE,
    OUTCOMES,
    RPY_ORDER,
    SCORE_WEIGHTS,
    Failure,
    PickCandidate,
    PickParams,
    PickResult,
    PickStep,
    PickTask,
    PipelineState,
)

__all__ = [
    "OUTCOMES",
    "OUTCOME_ALL_LIMIT_VIOLATED",
    "OUTCOME_ALL_NEAR_SINGULAR",
    "OUTCOME_MICRO_MOTION_INFEASIBLE",
    "OUTCOME_NOT_IN_WORKSPACE",
    "OUTCOME_POSE_NOT_REACHABLE",
    "OUTCOME_REACHABLE",
    "OUTCOME_RESIDUAL_TOO_LARGE",
    "PIPELINE",
    "RPY_ORDER",
    "SCORE_WEIGHTS",
    "Failure",
    "PickCandidate",
    "PickParams",
    "PickResult",
    "PickStep",
    "PickTask",
    "PipelineState",
    "classify_all_rejected",
    "classify_no_candidate",
    "pick",
    "reject_reason",
    "score_candidate",
    "step_choose",
    "step_inspect",
    "step_locate",
    "step_micro_motion",
    "step_prescreen",
    "step_solve",
    "to_report",
]


def pick(robot: rtb.Robot, task: PickTask, params: PickParams) -> PickResult:
    """@brief 跑完七道工序，回答"这次抓取能不能做，不能做差在哪"。

    执行器只做一件事：**按 `PIPELINE` 的顺序调用工序，遇到 `Failure` 就停**。

    @param robot Panda 模型（夹爪 TCP）
    @param task 任务：抓哪里、什么姿态
    @param params 阈值与采样参数（来自 configs/default.yaml 的 pick 段）
    @return 结论 + 理由 + 全部步骤；result.q 为 None 即不可执行，具体原因看 outcome
    """
    state = PipelineState()
    for step in PIPELINE:
        failure = step(robot, task, params, state)
        if failure is not None:
            return PickResult(
                outcome=failure.outcome,
                reason=failure.reason,
                target=state.target,  # ① 从不会失败，所以这里必定已就绪
                q=None,
                candidates=tuple(state.candidates),
                steps=tuple(state.steps),
                reach=state.reach,
            )

    best = state.best  # ⑦ 通过 ⇒ ⑥ 已经选出解
    assert best is not None
    travel = float(np.linalg.norm(best.q - task.home(best.q.size)))
    return PickResult(
        outcome=OUTCOME_REACHABLE,
        reason=(
            f"找到可执行解：任务残差 {best.residual:.2e}，"
            f"σ_min={best.sigma_min:.4f}，离限位余量 {np.rad2deg(best.limit_margin):.1f}°，"
            f"相对初始位形行程 {travel:.3f} rad"
        ),
        target=state.target,
        q=best.q,
        candidates=tuple(state.candidates),
        steps=tuple(state.steps),
        reach=state.reach,
    )
