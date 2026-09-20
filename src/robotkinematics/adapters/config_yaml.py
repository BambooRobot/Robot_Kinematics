"""@file config.py

@brief 运行时配置：读 YAML、校验取值范围、处理命令行覆盖。

三条行为是刻意设计的（与同目录的 face_recognition_app 保持一致）：
  1. 不认识的配置项要**告警**，而不是静默忽略 —— 否则拼错的配置会让人查半天。
  2. 数值要**校验范围**，把错误拦在启动阶段，而不是算到一半才崩。
  3. **命令行参数优先级高于配置文件**：配置文件写默认值，命令行做一次性试验。
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any

import yaml

from ..core.exceptions import KinematicsError

# src/robotkinematics/infrastructure/config.py -> 项目根目录
PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG_PATH = "configs/default.yaml"


def resolve_path(path: str | Path) -> Path:
    """相对路径先按当前工作目录找，找不到再回到项目根目录。

    这样既能“在项目根目录下按 README 运行”，也能从任意目录调用（比如 IDE 直接运行）。
    """
    p = Path(path)
    if p.is_absolute():
        return p
    if p.exists():
        return p.resolve()
    return (PROJECT_ROOT / p).resolve()


@dataclass
class PlanarConfig:
    """二连杆的几何参数。"""

    l1: float = 1.0
    l2: float = 1.0


@dataclass
class CameraConfig:
    """相机在本体坐标系下的安装位姿。"""

    x: float = 0.30
    y: float = 0.00
    z: float = 0.60
    yaw_deg: float = 30.0


@dataclass
class IKConfig:
    """数值 IK 的求解参数（传给库的 `ikine_LM`）。"""

    dls_lambda: float = 0.05
    max_iter: int = 200
    tol: float = 1.0e-9
    step: float = 1.0


@dataclass
class SingularityConfig:
    """奇异位姿的判定阈值。"""

    det_eps: float = 1.0e-9
    cond_warn: float = 100.0


@dataclass
class PickConfig:
    """抓取流水线的参数（rkin pick）。"""

    seeds: int = 24  # 多初值 IK 的初值个数
    residual_tol: float = 1.0e-6  # 末端位姿残差上限 [m / rad]
    unreachable_residual: float = 0.02  # 超过它判为"姿态不可达"，否则判为"精度不足"
    min_sigma: float = 0.02  # 雅可比最小奇异值下限
    workspace_samples: int = 8000  # 工作空间采样数（可达边界是估计值）
    workspace_seed: int = 0  # 采样种子，固定以保证结论可复现
    limit_margin_deg: float = 0.0  # 关节限位余量要求 [deg]


@dataclass
class PathsConfig:
    """数据与产物的目录（相对路径会先按当前目录找，再回到项目根目录）。"""

    cases_dir: str = "data/cases"
    outputs_dir: str = "outputs"


@dataclass
class Config:
    """整个项目的运行时配置，对应 configs/default.yaml 的各个段。"""

    planar: PlanarConfig = field(default_factory=PlanarConfig)
    camera: CameraConfig = field(default_factory=CameraConfig)
    ik: IKConfig = field(default_factory=IKConfig)
    singularity: SingularityConfig = field(default_factory=SingularityConfig)
    paths: PathsConfig = field(default_factory=PathsConfig)
    pick: PickConfig = field(default_factory=PickConfig)
    source: str = ""  # 实际读到的配置文件路径，便于排查“改了没生效”

    # ── 读取与校验 ──────────────────────────────────────────────────────────

    @classmethod
    def from_yaml(cls, path: str | Path | None = None) -> Config:
        """@brief 读 YAML 建配置：未识别的配置项告警、数值做范围校验。

        @param path 配置文件路径；None 表示用默认的 configs/default.yaml
        @return 校验过的 Config 实例（`config.source` 记录实际读到的文件）
        @throws KinematicsError 文件打不开、顶层不是映射、或数值越界
        """
        resolved = resolve_path(path or DEFAULT_CONFIG_PATH)
        if not resolved.exists():
            raise KinematicsError(f"无法打开配置文件: {resolved}")
        with resolved.open("r", encoding="utf-8") as fh:
            raw = yaml.safe_load(fh) or {}
        if not isinstance(raw, dict):
            raise KinematicsError(f"配置文件顶层必须是一个映射: {resolved}")

        config = cls()
        _fill(config.planar, raw.get("planar"), "planar")
        _fill(config.camera, raw.get("camera"), "camera")
        _fill(config.ik, raw.get("ik"), "ik")
        _fill(config.singularity, raw.get("singularity"), "singularity")
        _fill(config.paths, raw.get("paths"), "paths")
        _fill(config.pick, raw.get("pick"), "pick")

        known_sections = {f.name for f in fields(cls)}
        for key in raw:
            if key not in known_sections:
                _warn(f'配置段 "{key}" 无法识别，已忽略（是否拼写有误？）')

        config.validate()
        config.source = str(resolved)
        return config

    def validate(self) -> None:
        """@brief 校验各配置项的取值区间 —— 把错误拦在启动阶段，而不是算到一半才崩。

        @throws KinematicsError 任一项越界或不是数字
        """
        _check_range("planar.l1", self.planar.l1, 1e-9, 1e6)
        _check_range("planar.l2", self.planar.l2, 1e-9, 1e6)
        _check_range("ik.dls_lambda", self.ik.dls_lambda, 1e-9, 1e3)
        _check_range("ik.max_iter", self.ik.max_iter, 1, 10_000)
        _check_range("ik.tol", self.ik.tol, 1e-15, 1e3)
        _check_range("ik.step", self.ik.step, 1e-6, 1.0)
        _check_range("singularity.det_eps", self.singularity.det_eps, 1e-15, 1e3)
        _check_range("singularity.cond_warn", self.singularity.cond_warn, 1.0, 1e12)
        _check_range("pick.seeds", self.pick.seeds, 1, 1000)
        _check_range("pick.residual_tol", self.pick.residual_tol, 1e-15, 1e3)
        _check_range("pick.unreachable_residual", self.pick.unreachable_residual, 1e-9, 1e3)
        _check_range("pick.min_sigma", self.pick.min_sigma, 1e-9, 1e3)
        _check_range("pick.workspace_samples", self.pick.workspace_samples, 100, 1_000_000)
        _check_range("pick.limit_margin_deg", self.pick.limit_margin_deg, 0.0, 180.0)

    # ── 路径 ────────────────────────────────────────────────────────────────

    @property
    def cases_dir_path(self) -> Path:
        """算例 CSV 所在目录（已解析成绝对路径）。"""
        return resolve_path(self.paths.cases_dir)

    @property
    def outputs_dir_path(self) -> Path:
        """报告与图片的输出目录（已解析成绝对路径）。"""
        return resolve_path(self.paths.outputs_dir)

    # ── 命令行覆盖 ──────────────────────────────────────────────────────────

    def override(self, **kwargs: Any) -> Config:
        """把命令行传来的非 None 值写回配置（例如 l1=0.5 → config.planar.l1）。

        key 用扁平的点号形式不够直观，这里约定：planar_l1 / planar_l2 / ik_dls_lambda /
        ik_max_iter / ik_tol / ik_step / singularity_det_eps / singularity_cond_warn。
        """
        for key, value in kwargs.items():
            if value is None:
                continue
            section_name, _, field_name = key.partition("_")
            section = getattr(self, section_name, None)
            if section is None or not hasattr(section, field_name):
                raise KinematicsError(f"无法识别的配置覆盖项: {key}")
            setattr(section, field_name, value)
        self.validate()
        return self


def _fill(section: Any, values: Any, name: str) -> None:
    """把 YAML 里的一段填进 dataclass，未知键告警。"""
    if values is None:
        return
    if not isinstance(values, dict):
        raise KinematicsError(f"配置段 {name} 必须是一个映射")
    known = {f.name for f in fields(section)}
    for key, value in values.items():
        if key not in known:
            _warn(f'配置项 "{name}.{key}" 无法识别，已忽略（是否拼写有误？）')
            continue
        setattr(section, key, value)


def _check_range(name: str, value: Any, low: float, high: float) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise KinematicsError(f"配置项 {name} 必须是数字，收到 {value!r}")
    if not (low <= float(value) <= high):
        raise KinematicsError(f"配置项 {name}={value} 超出允许范围 [{low}, {high}]")


def _warn(message: str) -> None:
    # 走 stderr：这样 `rkin ... > out.txt` 重定向时警告仍然显示在终端
    print(f"警告: {message}", file=sys.stderr)
