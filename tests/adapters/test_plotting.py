"""@file test_plotting.py

@brief 绘图适配器：文件要真的生成；3D 不可用时要有回退。
"""

from __future__ import annotations

import numpy as np

from robotkinematics.adapters import plot_mpl as plotting
from robotkinematics.core import robots
from robotkinematics.usecases import pick as pick_uc


def test_chinese_font_is_configured_or_gracefully_skipped():
    """找不到中文字体时要安静跳过。"""
    name = plotting.configure_chinese_font()
    assert name is None or name in plotting.CJK_CANDIDATES


def test_panda_skeleton_is_written_even_without_3d(tmp_path, capsys):
    """没有 3D 后端时骨架图仍要出文件。"""
    robot = robots.panda(robots.TCP)
    path = plotting.plot_panda_skeleton(robot, np.zeros(7), tmp_path)
    assert path.exists()
    available, _ = plotting._axes3d_available()
    if not available:
        assert "3D 模块不可用" in capsys.readouterr().err


def test_reach_bound_is_the_sampled_maximum():
    """可达上界改成采样估计。"""
    robot = robots.panda(robots.FLANGE)
    bound = plotting._reach_bound(robot, samples=2000)
    assert 0.5 < bound < 1.5


def test_pick_figure_is_written(tmp_path):
    """抓取任务图要出 Panda 图。"""
    robot = robots.panda(robots.TCP)
    task = pick_uc.PickTask(observation=np.array([0.2, 0.0, 0.0]))
    params = pick_uc.PickParams(seeds=6, workspace_samples=1500)
    result = pick_uc.pick(robot, task, params)
    path = plotting.plot_pick_result(task, result, robot, tmp_path)
    assert path.exists() and path.name == "pick_panda.png"
    assert path.stat().st_size > 5000


def test_plotter_satisfies_the_protocol():
    """MatplotlibPlotter 要满足 Plotter 协议。"""
    from robotkinematics.contracts import Plotter

    assert isinstance(plotting.MatplotlibPlotter(), Plotter)
