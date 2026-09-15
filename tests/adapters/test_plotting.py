"""@file test_plotting.py
@brief 绘图：文件要真的生成、3D 不可用时要有三视图回退、字体配置不能崩。
"""

from __future__ import annotations

import numpy as np
import pytest

from robotkinematics.adapters import plot_mpl as plotting
from robotkinematics.core import panda
from robotkinematics.core.planar2r import Planar2R


def test_chinese_font_is_configured_or_gracefully_skipped():
    """本机有 CJK 字体就该配上；没有也不能报错，只是图上中文缺字。"""
    name = plotting.configure_chinese_font()
    assert name is None or name in plotting.CJK_CANDIDATES


def test_two_link_plot_is_written(tmp_path):
    path = plotting.plot_two_link(Planar2R(), 45.0, 45.0, tmp_path)
    assert path.exists()
    assert path.stat().st_size > 5000
    assert path.name == "fk_q1_45_q2_45.png"


def test_two_link_plot_accepts_a_target_point(tmp_path):
    path = plotting.plot_two_link(Planar2R(), 0.0, 90.0, tmp_path, target=(1.0, 1.0))
    assert path.exists()


def test_panda_skeleton_is_written_even_without_3d(tmp_path, capsys):
    """3D 模块坏掉时回退三视图 —— 本机就是这种情况，所以这条测试实际在跑回退分支。"""
    chain = panda.panda_urdf_chain()
    path = plotting.plot_panda_skeleton(chain, panda.PANDA_Q_PPT_GOAL, tmp_path)
    assert path.exists()
    assert path.name == "panda_fk_pose.png"
    available, _ = plotting._axes3d_available()
    if not available:
        # 回退时必须说清楚原因，而不是悄悄换一种画法
        assert "3D 模块不可用" in capsys.readouterr().err


def test_animation_writes_a_gif(tmp_path):
    path = plotting.animate_two_link(Planar2R(), sweep="q2", frames=8, outputs_dir=tmp_path)
    assert path.exists()
    assert path.suffix == ".gif"
    assert path.stat().st_size > 1000


def test_reach_bound_matches_the_sum_of_fixed_translations():
    """工作空间上界 = 所有固定平移的长度之和（三角不等式），与 FK 的实测最大距离一致。"""
    chain = panda.panda_urdf_chain()
    bound = plotting._reach_bound(chain)
    rng = np.random.default_rng(0)
    for q in rng.uniform(-np.pi, np.pi, size=(100, 7)):
        assert np.linalg.norm(chain.fk(q).t) <= bound + 1e-9


@pytest.mark.parametrize("sweep", ["q1", "q2"])
def test_animation_supports_both_joints(tmp_path, sweep):
    path = plotting.animate_two_link(Planar2R(), sweep=sweep, frames=6, outputs_dir=tmp_path)
    assert path.name == f"two_link_sweep_{sweep}.gif"
