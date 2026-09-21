# robot_kinematics_app —— 统一入口
#
# 用法：make help
# 约定：所有命令都在项目根目录运行（程序按相对路径找 configs/ 与 data/）
SHELL := /bin/bash
PYTHON ?= python3
export PYTHONPATH := src

.DEFAULT_GOAL := help
.PHONY: help test test-fast lint fmt type check-all run pick pick-fail arch clean

help:  ## 列出所有可用目标
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}'

test:  ## 跑全部测试（含覆盖率）
	$(PYTHON) -m pytest

test-fast:  ## 只跑测试，不算覆盖率（跳过 slow 标记）
	$(PYTHON) -m pytest -m "not slow" --no-cov -q

lint:  ## 静态检查（ruff）
	@command -v ruff >/dev/null 2>&1 || { echo "缺少 ruff：pip install -e ".[dev]""; exit 1; }
	ruff check src tests scripts

fmt:  ## 自动修复可修的静态问题
	@command -v ruff >/dev/null 2>&1 || { echo "缺少 ruff：pip install -e ".[dev]""; exit 1; }
	ruff check --fix src tests scripts
	ruff format src tests scripts

type:  ## 类型检查（mypy）
	@command -v mypy >/dev/null 2>&1 || { echo "缺少 mypy：pip install -e ".[dev]""; exit 1; }
	mypy

check-all: lint type test  ## 提交前跑这一条：静态检查 + 类型 + 测试

run:  ## 跑演示流程，产物落到 outputs/
	bash scripts/run_all.sh

pick:  ## 抓取流水线：相机观测 → 关节解（默认带一张任务图）
	$(PYTHON) -m robotkinematics --observe 0.2 0 0 --plot

pick-fail:  ## 抓取流水线：一个够不着 / 姿态不可达的例子（看失败结论怎么给）
	$(PYTHON) -m robotkinematics --observe 0.35 -0.05 0.15

arch:  ## 重新生成架构图（SVG + PNG）
	$(PYTHON) scripts/render_architecture.py

clean:  ## 清掉缓存与产物（不动源码）
	rm -rf .pytest_cache .mypy_cache .ruff_cache .coverage htmlcov build dist *.egg-info
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
	rm -f outputs/*.txt outputs/*.png outputs/*.gif
