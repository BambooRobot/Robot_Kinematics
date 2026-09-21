"""@file cases.py

@brief 读取 data/cases/*.csv：可选算例数据集（流水线入口不依赖；供测试与扩展用）。

  * coordinate_cases.csv —— camera frame 到 base frame 的坐标变换
  * panda_cases.csv      —— Panda 的演示姿态
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..core.exceptions import KinematicsError

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
class PandaCase:
    """Panda 演示姿态。"""

    case_id: str
    q: np.ndarray
    description: str


def load_coordinate_cases(directory: str | Path) -> list[CoordinateCase]:
    """@brief 读 coordinate_cases.csv：相机→本体变换的算例。"""
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


def load_panda_cases(directory: str | Path) -> list[PandaCase]:
    """@brief 读 panda_cases.csv：Panda 演示姿态。"""
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


def _read_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise KinematicsError(f"找不到算例文件: {path}")
    with path.open("r", encoding=_ENCODING, newline="") as fh:
        rows = [{k: (v or "").strip() for k, v in row.items() if k} for row in csv.DictReader(fh)]
    return [row for row in rows if any(row.values())]


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


class CsvCaseSource:
    """实现 contracts.CaseSource 协议：从 data/cases/*.csv 读算例。"""

    def __init__(self, directory: str | Path) -> None:
        """@brief 指定算例目录。

        @param directory 存放 CSV 的目录（通常是 configs 里的 paths.cases_dir）
        """
        self.directory = Path(directory)

    def coordinate_cases(self) -> list[CoordinateCase]:
        """@brief 见 load_coordinate_cases。"""
        return load_coordinate_cases(self.directory)

    def panda_cases(self) -> list[PandaCase]:
        """@brief 见 load_panda_cases。"""
        return load_panda_cases(self.directory)
