"""@file pick.py

@brief 抓取任务流水线（**门面**）：七道工序串成一条链、跑它、把结论报出来。

这是本项目唯一"总入口"式的用例。它回答的不是"某个数是多少"，而是
**"这次抓取能不能做，不能做差在哪"**。

【本模块只有两件事】把工序连起来（`PIPELINE`）、按序执行并在失败处停下（`pick`）。
其余三件事分给同目录的三个模块，各有各的名字：

  * `pick_types.py`  —— 词汇与数据契约：结论常量、Task / Params / Candidate / Result
  * `pick_steps.py`  —— 七道工序各自**怎么算**，以及失败时**怎么归类**
  * `pick_report.py` —— 报什么（`Report` 的结构；怎么画是适配器的事）

对外只需要 `import usecases.pick`；下面把常用的名字重新导出，是为了让调用方
（入口 `main.py`、测试）不必关心这几个文件是怎么切的。
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
    于是"哪一步失败、失败叫什么名字"完全由工序决定，这里不掺业务判断 ——
    想加第八道工序，写一个同样签名的函数、插进 `PIPELINE` 即可。

    @param robot Panda（或二连杆）模型
    @param task 任务：抓哪里、什么姿态、用哪台机器人
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
    reason = (
        f"从 {len(state.candidates)} 个候选解中选出评分最高的一组：离限位余量 "
        f"{np.rad2deg(best.limit_margin):.1f}°、σ_min={best.sigma_min:.4f}、"
        f"离初始位形 {travel:.3f} rad；任务残差 {best.residual:.2e}"
    )
    return PickResult(
        outcome=OUTCOME_REACHABLE,
        reason=reason,
        target=state.target,
        q=best.q,
        candidates=tuple(state.candidates),
        steps=tuple(state.steps),
        reach=state.reach,
    )
