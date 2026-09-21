#!/usr/bin/env bash
# 一键跑演示：Panda 抓取流水线（成功 / 失败对照）→ 出图 → 测试。
# 产物统一落到 outputs/。
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

step "1. 抓取流水线（可达目标 + 任务图）"
"${RKIN[@]}" --observe 0.2 0 0 --plot

step "2. 同一份结果输出 JSON"
"${RKIN[@]}" --observe 0.2 0 0 --format json >/dev/null
echo "json 完成"

step "3. 失败也是结论：同一个目标、只换抓取姿态"
echo "   （自上而下抓）"
"${RKIN[@]}" --observe 0.35 -0.05 0.15 2>/dev/null | grep -E "^结论|^原因" || true
echo "   （换成侧面接近）"
"${RKIN[@]}" --observe 0.35 -0.05 0.15 --rpy 0 90 0 2>/dev/null | grep -E "^结论|^原因" || true

step "4. 开关演示：--mode / --prefer-config / --seeds"
"${RKIN[@]}" --observe 0.5 0 0.4 --mode base >/dev/null
"${RKIN[@]}" --observe 0.2 0 0 --prefer-config 0 -0.3 0 -2.2 0 2.0 0.8 --seeds 16 >/dev/null
echo "开关演示完成"

step "5. 测试"
python3 -m pytest -q

step "完成：产物清单"
ls -1 outputs/
