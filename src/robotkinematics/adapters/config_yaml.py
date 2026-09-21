"""@file config.py

@brief 运行时配置：读 YAML、校验取值范围、处理命令行覆盖。

三条行为是刻意设计的：
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

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG_PATH = "configs/default.yaml"


def resolve_path(path: str | Path) -> Path:
    """相对路径先按当前工作目录找，找不到再回到项目根目录。"""
    p = Path(path)
    if p.is_absolute():
        return p
    if p.exists():
        return p.resolve()
    return (PROJECT_ROOT / p).resolve()


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

    max_iter: int = 200
    tol: float = 1.0e-9


@dataclass
class PickConfig:
    """抓取流水线的参数（rkin --observe ...）。"""

    seeds: int = 24
    residual_tol: float = 1.0e-6
    unreachable_residual: float = 0.02
    min_sigma: float = 0.02
    workspace_samples: int = 8000
    workspace_seed: int = 0
    limit_margin_deg: float = 0.0


@dataclass
class PathsConfig:
    """数据与产物的目录（相对路径会先按当前目录找，再回到项目根目录）。"""

    cases_dir: str = "data/cases"
    outputs_dir: str = "outputs"


@dataclass
class Config:
    """整个项目的运行时配置，对应 configs/default.yaml 的各个段。"""

    camera: CameraConfig = field(default_factory=CameraConfig)
    ik: IKConfig = field(default_factory=IKConfig)
    paths: PathsConfig = field(default_factory=PathsConfig)
    pick: PickConfig = field(default_factory=PickConfig)
    source: str = ""

    @classmethod
    def from_yaml(cls, path: str | Path | None = None) -> Config:
        """@brief 读 YAML 建配置：未识别的配置项告警、数值做范围校验。"""
        resolved = resolve_path(path or DEFAULT_CONFIG_PATH)
        if not resolved.exists():
            raise KinematicsError(f"无法打开配置文件: {resolved}")
        with resolved.open("r", encoding="utf-8") as fh:
            raw = yaml.safe_load(fh) or {}
        if not isinstance(raw, dict):
            raise KinematicsError(f"配置文件顶层必须是一个映射: {resolved}")

        config = cls()
        _fill(config.camera, raw.get("camera"), "camera")
        _fill(config.ik, raw.get("ik"), "ik")
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
        """@brief 校验各配置项的取值区间。"""
        _check_range("ik.max_iter", self.ik.max_iter, 1, 10_000)
        _check_range("ik.tol", self.ik.tol, 1e-15, 1e3)
        _check_range("pick.seeds", self.pick.seeds, 1, 1000)
        _check_range("pick.residual_tol", self.pick.residual_tol, 1e-15, 1e3)
        _check_range("pick.unreachable_residual", self.pick.unreachable_residual, 1e-9, 1e3)
        _check_range("pick.min_sigma", self.pick.min_sigma, 1e-9, 1e3)
        _check_range("pick.workspace_samples", self.pick.workspace_samples, 100, 1_000_000)
        _check_range("pick.limit_margin_deg", self.pick.limit_margin_deg, 0.0, 180.0)

    @property
    def cases_dir_path(self) -> Path:
        """算例 CSV 所在目录（已解析成绝对路径）。"""
        return resolve_path(self.paths.cases_dir)

    @property
    def outputs_dir_path(self) -> Path:
        """报告与图片的输出目录（已解析成绝对路径）。"""
        return resolve_path(self.paths.outputs_dir)

    def override(self, **kwargs: Any) -> Config:
        """把命令行传来的非 None 值写回配置（例如 max_iter=9 → config.ik.max_iter）。

        约定：ik_max_iter / ik_tol。
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
    print(f"警告: {message}", file=sys.stderr)
