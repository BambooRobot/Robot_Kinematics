"""@file ik_solvers.py
@brief 数值逆运动学：阻尼最小二乘（DLS / Levenberg-Marquardt）。

【和二连杆的区别】二连杆能用几何公式直接解出闭式解；7 自由度的 Panda 没有实用的闭式解，
工程上一律用数值迭代（课件里的 robot.ikine_LM 就是这一类）。

【为什么是“阻尼”最小二乘】朴素的最小二乘 dq = J⁺e 在奇异点附近会把 dq 放大到无穷大。
加上 λ 后 dq = Jᵀ(JJᵀ + λ²I)⁻¹e：σ ≫ λ 时与最小二乘几乎一致，σ → 0 时 dq 被压住不发散。
λ 就是“宁可慢一点，也别让关节疯转”的那个旋钮。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np

from .chain import SerialChain
from .exceptions import KinematicsError, NotConvergedError
from .rotations import rot_log
from .se3 import SE3


class PoseSource(Protocol):
    """IK 只需要这三件事：关节数、能算 FK、能给雅可比。

    ⚠️ n_joints 声明成只读属性：SerialChain 是用 @property 实现的，
       写成可变属性的话 mypy 会判定"只读实现不满足可变协议"而拒绝传参。
    """

    @property
    def n_joints(self) -> int: ...

    def fk(self, q: np.ndarray) -> SE3: ...

    def jacobian(self, q: np.ndarray) -> np.ndarray: ...


@dataclass(frozen=True)
class IKResult:
    """一次数值 IK 的结果。失败时 q 是最后一次迭代的值，方便看出卡在哪。

    success 只看“参与求解的那几维”（rows）是否达标；position_error / orientation_error /
    residual 则照实报告完整位姿误差 —— 平面二连杆永远匹配不了姿态，但它确实解掉了自己的任务。
    """

    q: np.ndarray
    success: bool
    iterations: int
    position_error: float
    orientation_error: float
    residual: float

    def summary(self) -> str:
        mark = "成功" if self.success else "未收敛"
        return (
            f"{mark}：迭代 {self.iterations} 次，位置误差 {self.position_error:.3e} m，"
            f"姿态误差 {self.orientation_error:.3e} rad，综合残差 {self.residual:.3e}"
        )


def pose_error(current: SE3, target: SE3) -> np.ndarray:
    """6 维位姿误差 [位置误差; 姿态误差]，两者都在 base 坐标系下表达。

    姿态误差用旋转向量（轴×角）：它的方向就是“末端该绕哪根轴转”，长度是转角，
    与几何雅可比的角速度行严格对应 —— 换成欧拉角差值就错位了。
    """
    dp = target.t - current.t
    dR = target.R @ current.R.T
    return np.concatenate([dp, rot_log(dR)])


def solve_dls(
    robot: PoseSource,
    target: SE3,
    q0: np.ndarray,
    *,
    dls_lambda: float = 0.05,
    max_iter: int = 200,
    tol: float = 1e-9,
    step: float = 1.0,
    rows: slice = slice(0, 6),
    q_ref: np.ndarray | None = None,
    nullspace_gain: float = 0.0,
    stall_patience: int = 20,
) -> IKResult:
    """阻尼最小二乘迭代求解，返回最后一次迭代的结果（成功与否都在里面）。

    rows：只对误差的哪几维做求解。平面二连杆应该传 slice(0, 2) ——
          它根本动不了 z 和三个转角，硬把 6 维误差塞进去只会自找麻烦。

    q_ref + nullspace_gain：冗余机械臂的“次要任务”。7 自由度下同一个位姿有无穷多组
          关节角，把多出来的自由度用来靠近 q_ref（比如让姿态更舒服、离限位更远）。

    ⚠️ 两个任务是**同时**迭代的，收敛判据也必须同时满足：主任务达标 + 零空间步长足够小。
       不能“先让主任务收敛、再单独走零空间”—— 零空间只在局部线性近似下成立，
       单独走一步 0.01 rad 就会带来约 1e-5 的二阶位姿漂移，任何严格的位姿校验都会把它拦下，
       次要任务就永远等于没做。同时迭代时，主任务那一项会随时把漂移修回去，这才是标准做法。
    """
    q = np.asarray(q0, dtype=float).reshape(-1).copy()
    if q.size != robot.n_joints:
        raise KinematicsError(f"初值维度 {q.size} 与关节数 {robot.n_joints} 不符")
    if q_ref is not None:
        q_ref = np.asarray(q_ref, dtype=float).reshape(-1)
        if q_ref.size != robot.n_joints:
            raise KinematicsError(f"参考姿态维度 {q_ref.size} 与关节数 {robot.n_joints} 不符")

    best: IKResult | None = None
    result = _evaluate(robot, target, q, rows, 0, tol)
    best_residual = float("inf")
    stalled = 0
    for iteration in range(max_iter + 1):
        result = _evaluate(robot, target, q, rows, iteration, tol)
        if best is None or result.residual < best.residual:
            best = result

        # 停滞检测：连续多轮都没把残差压下去，就说明这个初值落进了够不到的局部极小，
        # 再迭代只是烧时间（"目标不可达"的用例会跑满整个迭代预算）。
        # ⚠️ 用严格小于 + 相对容差：纯粹比较相等会因浮点噪声永远判不出停滞。
        if result.residual < best_residual * (1.0 - 1e-6):
            best_residual = result.residual
            stalled = 0
        else:
            stalled += 1
            if stalled >= stall_patience:
                break

        # 次要任务：把偏好中“落在零空间里”的那部分作为额外关节速度
        nullspace_step = None
        if q_ref is not None and nullspace_gain > 0.0:
            nullspace_step = nullspace_gain * (_nullspace_projector(robot, q, rows) @ (q_ref - q))

        primary_done = result.success
        secondary_done = nullspace_step is None or float(np.linalg.norm(nullspace_step)) < tol
        if primary_done and secondary_done:
            return result
        if iteration == max_iter:
            break

        e = pose_error(robot.fk(q), target)[rows]
        J = robot.jacobian(q)[rows, :]

        # 阻尼最小二乘：λ 越大越稳、越慢；λ=0 就是普通最小二乘（奇异点附近会炸）
        JJt = J @ J.T
        dq = J.T @ np.linalg.solve(JJt + (dls_lambda**2) * np.eye(JJt.shape[0]), e)
        if nullspace_step is not None:
            dq = dq + nullspace_step

        q = q + step * dq

    # 迭代次数用完都没收敛：返回历史上残差最小的那一次，而不是最后一次
    return best if best is not None else result


def nullspace_improvement_bound(
    robot: PoseSource, q: np.ndarray, q_ref: np.ndarray, rows: slice = slice(0, 6)
) -> tuple[float, float]:
    """估算次要任务最多能把“到参考姿态的距离”缩短多少 [rad]。

    返回 (能动分量, 距离缩短上限)。

    推导：记 d = q_ref − q，零空间投影 v = N·d。
    沿 v 走一步后 |d − v|² = |d|² − 2(v·d) + |v|²，而投影矩阵满足 v·d = |v|²，
    所以缩短量 = |d| − √(|d|² − |v|²) ≈ |v|²/(2|d|)（|v| ≪ |d| 时）。

    ⚠️ 关键：缩短量**不是** |v|。|v| 只是“能动的那部分偏好”，它与 |d| 的夹角决定了
       这点分量能省下多少距离：偏好方向与零空间接近正交时（工程上很常见），
       能动分量存在，但几乎不缩短距离。先算这两个数，再看实际效果，
       才知道“次要任务到底有没有生效”。
    """
    delta = np.asarray(q_ref, dtype=float).reshape(-1) - np.asarray(q, dtype=float).reshape(-1)
    movable = _nullspace_projector(robot, q, rows) @ delta
    movable_norm = float(np.linalg.norm(movable))
    distance = float(np.linalg.norm(delta))
    if distance <= 0.0:
        return movable_norm, 0.0
    bound = distance - float(np.sqrt(max(distance**2 - movable_norm**2, 0.0)))
    return movable_norm, bound


def _nullspace_projector(robot: PoseSource, q: np.ndarray, rows: slice) -> np.ndarray:
    """零空间投影矩阵 N = I − J⁺J：把任意关节方向投影到“不动末端”的子空间。

    ⚠️ 必须用未阻尼的伪逆 pinv：阻尼伪逆不再满足 J·(I − J⁺J) = 0，
       拿它做投影会让次要任务偷偷扰动主任务，迭代在目标附近卡住不收敛。
    """
    J = robot.jacobian(q)[rows, :]
    return np.eye(robot.n_joints) - np.linalg.pinv(J) @ J


def _evaluate(
    robot: PoseSource, target: SE3, q: np.ndarray, rows: slice, iterations: int, tol: float
) -> IKResult:
    full_error = pose_error(robot.fk(q), target)
    return IKResult(
        q=np.asarray(q, dtype=float).copy(),
        success=bool(np.linalg.norm(full_error[rows]) < tol),
        iterations=iterations,
        position_error=float(np.linalg.norm(full_error[:3])),
        orientation_error=float(np.linalg.norm(full_error[3:])),
        residual=float(np.linalg.norm(full_error)),
    )


def solve_dls_or_raise(robot: PoseSource, target: SE3, q0: np.ndarray, **kwargs) -> IKResult:
    """同 solve_dls，但没收敛时抛 NotConvergedError —— 给 CLI 用。"""
    result = solve_dls(robot, target, q0, **kwargs)
    if not result.success:
        raise NotConvergedError(
            f"IK 未收敛（{result.iterations} 次迭代后残差 {result.residual:.3e}）："
            "目标可能不可达、初值太远，或正处在奇异位姿附近"
        )
    return result


def solve_multi_seed(
    robot: PoseSource,
    target: SE3,
    seeds: list[np.ndarray],
    **kwargs,
) -> list[IKResult]:
    """多初值求解 —— 用来演示冗余：同一个目标位姿，可以有好几组完全不同的关节角。

    课件 09 的结论（robot.ikine_LM 从全零初值解出的 q 与生成目标的 q 不同）就是这件事。
    """
    return [solve_dls(robot, target, seed, **kwargs) for seed in seeds]


def random_seeds(
    robot: PoseSource, count: int, seed: int = 0, scale: float = 1.0
) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    return [rng.uniform(-scale, scale, size=robot.n_joints) for _ in range(count)]


def planar_chain(l1: float = 1.0, l2: float = 1.0) -> SerialChain:
    """把二连杆也写成串联链，好让数值 IK 与解析 IK 对照（测试里会断言两者一致）。

    链的展开顺序：Rz(q1) → Tx(L1) → Rz(q2) → Tx(L2)，
    正好是“先转第一节，再沿第一节走到肘，再转第二节，再沿第二节走到末端”。
    """
    from .chain import ETSLink

    return SerialChain(
        links=(
            ETSLink(offset=SE3.identity(), motion="Rz", name="q1"),
            ETSLink(offset=SE3.Trans(l1, 0.0, 0.0), motion="Rz", name="q2"),
        ),
        tool=SE3.Trans(l2, 0.0, 0.0),
        name="planar 2R (chain form)",
    )
