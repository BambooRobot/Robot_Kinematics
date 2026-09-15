"""@file reference_rtb.py
@brief 参照库适配器：用 spatialmath + roboticstoolbox 取一组数，交给用例去判定。

⚠️ **整个项目里只有这个文件 import 那两个库**（而且是函数内延迟导入）。
   这样没装库的环境永远不会因为缺依赖而挂，`rkin crosscheck` 也只是变成"跳过"。

⚠️ 这里只负责"把数取出来"，不做任何"一致/不一致"的判断 —— 判定在 usecases/compare.py。
   将来换成 Pinocchio 或 KDL 做参照，只需再写一个实现同一协议的适配器。
"""

from __future__ import annotations

import numpy as np

from ..contracts import ReferenceNumbers
from ..core import panda


class RtbReference:
    """spatialmath / roboticstoolbox 参照实现。"""

    def probe(self) -> ReferenceNumbers:
        spatialmath, rtb, reason = _import_optional()
        if spatialmath is None or rtb is None:
            return ReferenceNumbers(available=False, reason=reason)

        errors: list[tuple[str, str]] = []
        versions = (
            ("spatialmath", getattr(spatialmath, "__version__", "?")),
            ("roboticstoolbox", getattr(rtb, "__version__", "?")),
        )

        # ① SE3 变换链（不依赖机器人模型，出错就是环境问题）
        se3_chain_t: np.ndarray | None = None
        se3_inverse_t: np.ndarray | None = None
        try:
            T_lib = (
                spatialmath.SE3.Trans(0.30, 0.0, 0.60)
                * spatialmath.SE3.Rz(np.deg2rad(30))
                * spatialmath.SE3.Trans(0.50, -0.10, 0.20)
            )
            se3_chain_t = np.asarray(T_lib.t, dtype=float)
            se3_inverse_t = np.asarray(T_lib.inv().t, dtype=float)
        except Exception as exc:
            errors.append(("SE3 变换链", f"{type(exc).__name__}: {exc}"))
            se3_chain_t = se3_inverse_t = None

        # ② 库自带的平面二连杆模型
        try:
            planar_robot = rtb.models.DH.Planar2()
            planar_fk = np.asarray(
                planar_robot.fkine(np.array([np.deg2rad(30), np.deg2rad(60)])).t, dtype=float
            )
        except Exception as exc:
            errors.append(("二连杆 FK", f"{type(exc).__name__}: {exc}"))
            planar_fk = None

        # ③④ Panda —— 两端都取：
        #     库的默认末端是**夹爪 TCP**，要拿它跟自研的夹爪帧比；
        #     拿它直接跟法兰比会平白多出 0.1034 m。
        robot = rtb.models.Panda()
        poses = (panda.PANDA_Q_ZERO, panda.PANDA_Q_PPT_GOAL, panda.PANDA_Q_IK_SEED)

        flange_t, tcp_t = [], []
        for name, q in zip(("zero", "ppt_goal", "ik_seed"), poses, strict=True):
            try:
                flange_t.append(np.asarray(robot.fkine(q, end="panda_link8").t, dtype=float))
            except Exception as exc:
                errors.append((f"Panda FK({name}) 法兰", f"{type(exc).__name__}: {exc}"))
            try:
                tcp_t.append(np.asarray(robot.fkine(q).t, dtype=float))
            except Exception as exc:
                errors.append((f"Panda FK({name}) 夹爪 TCP", f"{type(exc).__name__}: {exc}"))

        q_jac = panda.PANDA_Q_IK_SEED
        try:
            jac_flange = np.asarray(robot.jacob0(q_jac, end="panda_link8"), dtype=float)
        except Exception as exc:
            errors.append(("Panda 雅可比（法兰帧）", f"{type(exc).__name__}: {exc}"))
            jac_flange = None
        try:
            jac_tcp = np.asarray(robot.jacob0(q_jac), dtype=float)
        except Exception as exc:
            errors.append(("Panda 雅可比（夹爪 TCP 帧）", f"{type(exc).__name__}: {exc}"))
            jac_tcp = None

        # ⑤ IK：目标用自研法兰帧的 FK 造出来，保证一定可达
        ik_success = None
        ik_q = None
        fk_of_ik_q = None
        try:
            target = panda.panda_urdf_chain().fk(panda.PANDA_Q_PPT_GOAL)
            solution = robot.ikine_LM(spatialmath.SE3(target.A), q0=np.zeros(7))
            ik_success = bool(solution.success)
            ik_q = np.asarray(solution.q, dtype=float)
        except Exception as exc:
            errors.append(("IK", f"{type(exc).__name__}: {exc}"))
        try:
            if ik_q is not None:
                fk_of_ik_q = np.asarray(robot.fkine(ik_q, end="panda_link8").t, dtype=float)
        except Exception as exc:
            errors.append(("IK 解的 FK 回验", f"{type(exc).__name__}: {exc}"))

        return ReferenceNumbers(
            available=True,
            versions=versions,
            se3_chain_t=se3_chain_t,
            se3_inverse_t=se3_inverse_t,
            planar_fk=planar_fk,
            panda_flange_t=tuple(flange_t),
            panda_tcp_t=tuple(tcp_t),
            panda_jacobian_flange=jac_flange,
            panda_jacobian_tcp=jac_tcp,
            ik_success=ik_success,
            ik_q=ik_q,
            fk_of_ik_q=fk_of_ik_q,
            call_errors=tuple(errors),
        )


def _import_optional():
    try:
        import spatialmath
    except ImportError as exc:
        return None, None, f"未安装 spatialmath（{exc}）"
    try:
        import roboticstoolbox as rtb
    except ImportError as exc:
        return None, None, f"未安装 roboticstoolbox（{exc}）"
    return spatialmath, rtb, ""
