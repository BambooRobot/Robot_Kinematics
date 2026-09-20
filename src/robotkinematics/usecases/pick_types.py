"""@file pick_types.py

@brief 抓取流水线的**词汇与数据契约**：结论、参数、任务、候选解、工序、总产出。

这里只有"名词" —— 常量、dataclass，以及数据自己的小方法（`home` / `preferred` / `survived`）。
**一点计算都不许加进来**：算法在 `pick_steps.py`，编排在 `pick.py`，报什么在 `pick_report.py`。
这么分是因为"词汇"要被其余三个文件同时依赖，而它自己谁都不依赖（不会出现循环导入）。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from spatialmath import SE3

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
    ik_tol: float = 1e-9  # 传给库的关节步长容差（不是位姿残差！见下方 ⚠️）
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
        """这条路走闭式解（二连杆）还是数值解（Panda）。"""
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
        """这个候选是否活到了最后 —— rejected_by 为空即存活。"""
        return not self.rejected_by


@dataclass(frozen=True)
class PickStep:
    """一道工序的结果：做了什么、结论、关键数字。"""

    name: str
    detail: str
    numbers: dict = field(default_factory=dict)


@dataclass(frozen=True)
class PickResult:
    """整条流水线的产出：结论 + 理由 + 全过程。

    outcome 是给程序看的结论（取值见 OUTCOMES），reason 是给人看的一句话；
    steps 留下每道工序的数字，所以失败时能回答"卡在第几步、还差多少"，
    而不是只有一句"失败了"。q 为 None 表示没有可执行的解。
    """

    outcome: str
    reason: str
    target: SE3
    q: np.ndarray | None
    candidates: tuple[PickCandidate, ...]
    steps: tuple[PickStep, ...]
    reach: workspace.ReachEstimate | None = None


@dataclass
class PipelineState:
    """工序之间传递的状态 —— 每道工序只读写自己关心的字段。

    为什么是一个可变对象而不是层层返回值：七道工序的产出形状各不相同
    （位姿 / 可达估计 / 候选列表 / 最优解），硬凑一个统一的返回值只会让每道工序
    都去拆一个它不关心的元组。这里用"共享状态 + 显式字段"，反而每道工序的
    输入输出一眼可见（看它读写了哪些字段）。
    """

    target: SE3 | None = None
    reach: workspace.ReachEstimate | None = None
    candidates: list[PickCandidate] = field(default_factory=list)
    best_residual: float = float("inf")
    best: PickCandidate | None = None
    steps: list[PickStep] = field(default_factory=list)


@dataclass(frozen=True)
class Failure:
    """一道工序给出的**短路信号**：整条流水线停在这里，带着有名字的结论。

    短路策略因此只有一处（`pick.py` 的执行器），而不是散在七道工序里各自 `return`。
    """

    outcome: str
    reason: str
