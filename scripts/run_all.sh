#!/usr/bin/env bash
# 一键跑完全部演示：自检 → 全部 CLI 用例 → 批量报告 → 出图 → 测试。
# 产物统一落到 outputs/，报告与图可以直接交作业。
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

# 优先用安装好的 rkin；没装就用 PYTHONPATH 直接跑模块
if command -v rkin >/dev/null 2>&1; then
    RKIN=(rkin)
else
    # 追加而不是覆盖：调用方已有的 PYTHONPATH 要保留（否则会屏蔽掉用户的环境设置）
    RKIN=(env "PYTHONPATH=${PYTHONPATH:+$PYTHONPATH:}$ROOT/src" python3 -m robotkinematics)
fi

step() { printf '\n\033[1;34m===== %s =====\033[0m\n' "$*"; }

step "0. 环境自检（含运动学库 spatialmath / roboticstoolbox）"
"${RKIN[@]}" check

step "1. 位姿表示与坐标变换"
"${RKIN[@]}" pose --xyz 0.4 0.2 0.3 --rpy 0 0 90 >/dev/null
echo "pose 完成（数值见 tests/test_applications.py::test_pose_report_contains_the_course_point_transform）"
"${RKIN[@]}" cam --all

step "2. 二连杆：FK / IK / 雅可比"
"${RKIN[@]}" fk --q1 0 --q2 90
"${RKIN[@]}" fk --q1 45 --q2 45 --plot
"${RKIN[@]}" ik --target 1 1
"${RKIN[@]}" jac --q1 0 --q2 90 --dx 0 0.1

step "3. 奇异点扫描"
"${RKIN[@]}" sing

step "4. Franka Panda"
"${RKIN[@]}" panda-fk --case ppt_goal --plot
"${RKIN[@]}" panda-ik --case ppt_goal
"${RKIN[@]}" panda-jac --case ik_seed
"${RKIN[@]}" panda-sing

step "4.5 抓取任务流水线（七个知识点串成一条）"
"${RKIN[@]}" pick --observe 0.2 0 0 --plot
"${RKIN[@]}" pick --observe 0.2 0 0 --arm planar --plot
echo "—— 失败也是结论：同一个目标、只换抓取姿态，结论就不同"
echo "   （自上而下抓）"
"${RKIN[@]}" pick --observe 0.35 -0.05 0.15 2>/dev/null | grep -E "^结论|^原因" || true
echo "   （换成侧面接近）"
"${RKIN[@]}" pick --observe 0.35 -0.05 0.15 --rpy 0 90 0 2>/dev/null | grep -E "^结论|^原因" || true

step "5. 出图与动画"
"${RKIN[@]}" anim --sweep q2 --frames 48
"${RKIN[@]}" anim --sweep q1 --frames 48

step "6. 批量算例报告"
"${RKIN[@]}" batch

step "7. 测试"
python3 -m pytest -q

step "完成：产物清单"
ls -1 outputs/
