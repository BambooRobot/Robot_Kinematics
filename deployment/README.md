# Deployment

## Local build

```bash
pip install -e .          # 依赖只声明在 pyproject.toml 一处
PYTHONPATH=src python -m robotkinematics --observe 0.2 0 0
# 或安装后：
rkin --observe 0.2 0 0
```

## Docker (environment only)

```bash
docker compose -f deployment/docker/compose.yaml up -d --build
docker compose -f deployment/docker/compose.yaml exec runtime bash

# 容器里
rkin --observe 0.2 0 0 --plot           # 任务图写到 /app/outputs → 宿主机 ./outputs
rkin --observe 0.2 0 0 --format json
```

镜像基于 `debian:bookworm-slim`，装了 python3、中文字体（`fonts-noto-cjk`，
出图上的中文标签需要它）以及本项目依赖；`MPLBACKEND=Agg` 保证无显示器环境也能出图。

`configs/`、`data/` 以只读方式挂载，`outputs/` 可写 —— 容器里跑出的报告和图
直接落在宿主机的 `outputs/` 里。

## 说明

本项目是教学与验证用的计算工程（没有摄像头、没有实时控制回路），
因此只提供 Docker 环境，不提供 systemd 常驻服务 —— 没有需要常驻的进程。
