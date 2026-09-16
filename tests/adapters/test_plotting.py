"""@file test_plotting.py
@brief 绘图适配器：文件要真的生成；3D 不可用时要有回退；内容与数据一致。
"""

from __future__ import annotations

import numpy as np

from robotkinematics.adapters import plot_mpl as plotting
from robotkinematics.core import robots
from robotkinematics.core.planar2r import Planar2R
from robotkinematics.usecases import pick as pick_uc


def test_chinese_font_is_configured_or_gracefully_skipped():
    name = plotting.configure_chinese_font()
    assert name is None or name in plotting.CJK_CANDIDATES


def test_two_link_plot_is_written(tmp_path):
    path = plotting.plot_two_link(Planar2R(), 45.0, 45.0, tmp_path)
    assert path.exists() and path.stat().st_size > 5000
    assert path.name == "fk_q1_45_q2_45.png"


def test_panda_skeleton_is_written_even_without_3d(tmp_path, capsys):
    robot = robots.panda(robots.TCP)
    path = plotting.plot_panda_skeleton(robot, np.zeros(7), tmp_path)
    assert path.exists()
    available, _ = plotting._axes3d_available()
    if not available:
        assert "3D 模块不可用" in capsys.readouterr().err


def test_animation_writes_a_gif(tmp_path):
    path = plotting.animate_two_link(Planar2R(), sweep="q2", frames=8, outputs_dir=tmp_path)
    assert path.exists() and path.suffix == ".gif" and path.stat().st_size > 1000


def test_reach_bound_is_the_sampled_maximum():
    """可达上界改成采样估计（库的 robot.reach 对本项目模型返回 0）。"""
    robot = robots.panda(robots.FLANGE)
    bound = plotting._reach_bound(robot, samples=2000)
    assert 0.5 < bound < 1.5  # 量级合理


def test_pick_figure_is_written_for_both_arms(tmp_path):
    """抓取任务图：两种机器人都要能出。"""
    for arm in ("panda", "planar"):
        if arm == "planar":
            robot, planar = robots.planar(), Planar2R()
        else:
            robot, planar = robots.panda(robots.TCP), None
        task = pick_uc.PickTask(observation=np.array([0.2, 0.0, 0.0]), arm=arm, planar=planar)
        params = pick_uc.PickParams(seeds=6, workspace_samples=1500)
        result = pick_uc.pick(robot, task, params)
        path = plotting.plot_pick_result(task, result, robot, tmp_path)
        assert path.exists() and path.name == f"pick_{arm}.png"
        assert path.stat().st_size > 5000


def test_plotter_satisfies_the_protocol():
    from robotkinematics.contracts import Plotter

    assert isinstance(plotting.MatplotlibPlotter(), Plotter)
