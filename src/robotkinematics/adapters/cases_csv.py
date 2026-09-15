"""@file cases.py
@brief 读取 data/cases/*.csv：课件里的算例数据集。

三张表的列名与课件保持一致（不改列名，才能和课件输出逐条对照）：
  * coordinate_cases.csv —— camera frame 到 base frame 的坐标变换
  * two_link_cases.csv   —— 二连杆 FK / IK / 奇异点（同一张表按“填了 q 还是填了 target”分流）
  * panda_cases.csv      —— Panda 的三个演示姿态
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..core.exceptions import KinematicsError

# 课件用的是带 BOM 的 CSV，用 utf-8-sig 读，否则第一个列名会多出
_ENCODING = "utf-8-sig"


@dataclass(frozen=True)
class CoordinateCase:
    """相机在本体下的安装位姿 + 杯子在相机下的位置。"""

    case_id: str
    cup_xyz: tuple[float, float, float]
    camera_xyz: tuple[float, float, float]
    camera_yaw_deg: float
    description: str


@dataclass(frozen=True)
class TwoLinkCase:
    """二连杆算例：填了 q 就是 FK 算例，填了 target 就是 IK 算例。"""

    case_id: str
    l1: float
    l2: float
    description: str
    q_deg: tuple[float, float] | None = None
    target: tuple[float, float] | None = None

    @property
    def kind(self) -> str:
        return "fk" if self.q_deg is not None else "ik"


@dataclass(frozen=True)
class PandaCase:
    """Panda 演示姿态。"""

    case_id: str
    q: np.ndarray
    description: str


def load_coordinate_cases(directory: str | Path) -> list[CoordinateCase]:
    rows = _read_rows(Path(directory) / "coordinate_cases.csv")
    return [
        CoordinateCase(
            case_id=_text(row, "case_id"),
            cup_xyz=_triple(row, "cup_x_camera", "cup_y_camera", "cup_z_camera"),
            camera_xyz=_triple(row, "camera_x_base", "camera_y_base", "camera_z_base"),
            camera_yaw_deg=_number(row, "camera_yaw_deg"),
            description=row.get("description", ""),
        )
        for row in rows
    ]


def load_two_link_cases(directory: str | Path) -> list[TwoLinkCase]:
    rows = _read_rows(Path(directory) / "two_link_cases.csv")
    cases = []
    for row in rows:
        q = _optional_pair(row, "q1_deg", "q2_deg")
        target = _optional_pair(row, "target_x", "target_y")
        if q is None and target is None:
            raise KinematicsError(
                f"算例 {row.get('case_id')!r} 既没有关节角也没有目标点，无法判断是 FK 还是 IK"
            )
        cases.append(
            TwoLinkCase(
                case_id=_text(row, "case_id"),
                l1=_number(row, "L1"),
                l2=_number(row, "L2"),
                description=row.get("description", ""),
                q_deg=q,
                target=target,
            )
        )
    return cases


def load_panda_cases(directory: str | Path) -> list[PandaCase]:
    rows = _read_rows(Path(directory) / "panda_cases.csv")
    cases = []
    for row in rows:
        q = np.array([_number(row, f"q{i}") for i in range(1, 8)])
        cases.append(
            PandaCase(
                case_id=_text(row, "case_id"),
                q=q,
                description=row.get("description", ""),
            )
        )
    return cases


# ── 内部工具 ────────────────────────────────────────────────────────────────


def _read_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise KinematicsError(f"找不到算例文件: {path}")
    with path.open("r", encoding=_ENCODING, newline="") as fh:
        rows = [{k: (v or "").strip() for k, v in row.items() if k} for row in csv.DictReader(fh)]
    return [row for row in rows if any(row.values())]  # 跳过空行


def _text(row: dict[str, str], key: str) -> str:
    value = row.get(key, "")
    if not value:
        raise KinematicsError(f"算例缺少必填列 {key}")
    return value


def _number(row: dict[str, str], key: str) -> float:
    raw = row.get(key, "")
    if raw == "":
        raise KinematicsError(f"算例 {row.get('case_id')!r} 缺少数值列 {key}")
    try:
        return float(raw)
    except ValueError as exc:
        raise KinematicsError(f"算例 {row.get('case_id')!r} 的 {key}={raw!r} 不是数字") from exc


def _triple(row: dict[str, str], kx: str, ky: str, kz: str) -> tuple[float, float, float]:
    return (_number(row, kx), _number(row, ky), _number(row, kz))


def _optional_pair(row: dict[str, str], k1: str, k2: str) -> tuple[float, float] | None:
    """CSV 里空白表示“这一列不适合本条算例”，返回 None 表示没填。"""
    if not row.get(k1, "") or not row.get(k2, ""):
        return None
    return (_number(row, k1), _number(row, k2))


class CsvCaseSource:
    """实现 contracts.CaseSource 协议：从 data/cases/*.csv 读算例。

    用例只认识 CaseSource 这个协议，所以将来把算例换成数据库或接口，
    只需要再写一个适配器，用例一行都不用改。
    """

    def __init__(self, directory: str | Path) -> None:
        self.directory = Path(directory)

    def coordinate_cases(self) -> list[CoordinateCase]:
        return load_coordinate_cases(self.directory)

    def two_link_cases(self) -> list[TwoLinkCase]:
        return load_two_link_cases(self.directory)

    def panda_cases(self) -> list[PandaCase]:
        return load_panda_cases(self.directory)
