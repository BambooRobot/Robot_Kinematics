"""@file se3.py
@brief SE(3)：三维刚体位姿（位置 + 姿态）与齐次变换矩阵。

【为什么要有这个类】位姿 = 位置 + 姿态，两者必须一起传播：只传位置，抓紧时不知道
从哪个方向靠近；只传姿态，不知道抓哪里。SE3 把 3x3 旋转和 3 维平移绑成一个对象，
让“A 坐标系下的 B 坐标系位姿”这种话能写成 ^A T_B，并且能连乘成变换链。

【API 命名】Trans / Rx / Ry / Rz / .A / .t / .R 刻意与课件里 spatialmath 的 SE3 同名同义，
方便对照 PPT。区别只在于：这里的每一个数都是本项目自己算的，不需要装 spatialmath。
"""

from __future__ import annotations

import numpy as np

from . import rotations
from .exceptions import InvalidRotationError


class SE3:
    """刚体变换 ^A T_B：把 B 坐标系下的量映射到 A 坐标系。"""

    __slots__ = ("_R", "_t")

    def __init__(self, R: np.ndarray | None = None, t: np.ndarray | None = None) -> None:
        if R is None:
            self._R = np.eye(3)
        else:
            R = np.asarray(R, dtype=float).reshape(3, 3)
            if not rotations.is_rotation(R, tol=1e-8):
                raise InvalidRotationError("旋转部分不是合法旋转矩阵（需正交且 det=+1）")
            self._R = R.copy()
        if t is None:
            self._t = np.zeros(3)
        else:
            self._t = np.asarray(t, dtype=float).reshape(3).copy()

    # ── 构造 ────────────────────────────────────────────────────────────────

    @staticmethod
    def identity() -> SE3:
        return SE3()

    @staticmethod
    def Trans(x: float, y: float, z: float) -> SE3:
        """纯平移：只回答“在哪里”，不表达“朝哪边”。"""
        return SE3(t=np.array([x, y, z]))

    @staticmethod
    def Rx(theta: float) -> SE3:
        """绕 x 轴旋转 theta（弧度）的纯旋转。"""
        return SE3(R=rotations.rotx(theta))

    @staticmethod
    def Ry(theta: float) -> SE3:
        return SE3(R=rotations.roty(theta))

    @staticmethod
    def Rz(theta: float) -> SE3:
        return SE3(R=rotations.rotz(theta))

    @classmethod
    def from_matrix(cls, A: np.ndarray) -> SE3:
        """从 4x4 齐次变换矩阵构造。"""
        A = np.asarray(A, dtype=float)
        if A.shape != (4, 4):
            raise InvalidRotationError(f"齐次变换矩阵必须是 4x4，收到 {A.shape}")
        if not np.allclose(A[3, :], [0.0, 0.0, 0.0, 1.0], atol=1e-9):
            raise InvalidRotationError("齐次变换矩阵最后一行必须是 [0, 0, 0, 1]")
        return cls(R=A[:3, :3], t=A[:3, 3])

    @classmethod
    def from_pose6(cls, x: float, y: float, z: float, roll: float, pitch: float, yaw: float) -> SE3:
        """六元组 (x, y, z, roll, pitch, yaw) 转 SE3 —— 课件里的 pose6_to_T。"""
        return cls(R=rotations.rpy_to_matrix(roll, pitch, yaw), t=np.array([x, y, z]))

    # ── 取值 ────────────────────────────────────────────────────────────────

    @property
    def R(self) -> np.ndarray:
        """旋转部分（返回副本：外部改它不该影响本对象）。"""
        return self._R.copy()

    @property
    def t(self) -> np.ndarray:
        """平移部分，即 B 的原点在 A 中的位置 [m]。"""
        return self._t.copy()

    @property
    def A(self) -> np.ndarray:
        """4x4 齐次矩阵形式 —— 课件里 T.A 就是这个。"""
        A = np.eye(4)
        A[:3, :3] = self._R
        A[:3, 3] = self._t
        return A

    def pose6(self) -> tuple[float, float, float, float, float, float]:
        """转回六元组 (x, y, z, roll, pitch, yaw)，角度为弧度。"""
        roll, pitch, yaw = rotations.matrix_to_rpy(self._R)
        return (self._t[0], self._t[1], self._t[2], roll, pitch, yaw)

    # ── 运算 ────────────────────────────────────────────────────────────────

    def __mul__(self, other: SE3) -> SE3:
        """位姿复合：^A T_C = ^A T_B · ^B T_C。

        ⚠️ 顺序不能反。工程上不要死记公式，读成一句话：
        “这个点现在在 B 里表达，下一步算法需要它在 A 里表达，那么乘 ^A T_B。”
        """
        if not isinstance(other, SE3):
            raise TypeError(
                "SE3 只能与 SE3 相乘（复合位姿）；变换点请用 apply()，旋转方向请用 apply_direction()"
            )
        return SE3(R=self._R @ other._R, t=self._R @ other._t + self._t)

    def inverse(self) -> SE3:
        """逆变换 ^B T_A。旋转用转置（正交矩阵的逆就是转置），平移要跟着转回去。

        ⚠️ 不能只把平移取负：那样得到的是“位置反了但姿态没反”，在带旋转的链上必错。
        """
        R_inv = self._R.T
        return SE3(R=R_inv, t=-R_inv @ self._t)

    def apply(self, point: np.ndarray) -> np.ndarray:
        """把一个点从 B 坐标系变换到 A 坐标系：p_A = R·p_B + t。"""
        p = np.asarray(point, dtype=float).reshape(3)
        return self._R @ p + self._t

    def apply_direction(self, direction: np.ndarray) -> np.ndarray:
        """只旋转方向向量，不加平移。方向（速度、力、法线）不该被平移影响。"""
        v = np.asarray(direction, dtype=float).reshape(3)
        return self._R @ v

    def is_close(self, other: SE3, tol: float = 1e-9) -> bool:
        return bool(
            np.allclose(self._R, other._R, atol=tol) and np.allclose(self._t, other._t, atol=tol)
        )

    # ── 展示 ────────────────────────────────────────────────────────────────

    def __repr__(self) -> str:
        rows = []
        for i in range(3):
            vals = ", ".join(f"{v: .4f}" for v in self.__mul_row(i))
            rows.append(f"  [{vals}]")
        tx, ty, tz = self._t
        return "\n".join([f"SE3  t = [{tx: .4f}, {ty: .4f}, {tz: .4f}]", *rows])

    def __mul_row(self, i: int) -> list[float]:
        """取齐次矩阵第 i 行 —— 仅用于打印，避免 __repr__ 里重复拼矩阵。"""
        return [*self._R[i], self._t[i]]

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, SE3):
            return NotImplemented
        return self.is_close(other, tol=0.0)
