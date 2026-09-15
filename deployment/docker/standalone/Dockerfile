# robot_kinematics_app 运行环境 —— 只提供环境，不自动跑任何东西。
# 进容器后自己敲 `rkin ...`，产物写回挂载出去的 outputs/。
FROM debian:bookworm-slim

ARG APT_MIRROR=mirrors.tuna.tsinghua.edu.cn
ARG PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple
ARG PIP_TRUSTED_HOST=pypi.tuna.tsinghua.edu.cn

ENV DEBIAN_FRONTEND=noninteractive
ENV PIP_INDEX_URL=${PIP_INDEX_URL}
ENV PIP_TRUSTED_HOST=${PIP_TRUSTED_HOST}

WORKDIR /app

COPY deployment/docker/scripts/configure-mirrors.sh /tmp/configure-mirrors.sh
RUN sh /tmp/configure-mirrors.sh && rm /tmp/configure-mirrors.sh

# python3 + 中文字体：出图时图上的中文标签需要 CJK 字体，否则是方块
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 python3-pip python3-venv \
        fonts-noto-cjk \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md Makefile ./
COPY src ./src
COPY scripts ./scripts
COPY configs ./configs
COPY data ./data
# 依赖只声明在 pyproject 里，装包时一并装上。
# 只装运行时依赖：容器是"跑起来"用的，不需要 ruff/mypy/pytest
RUN pip3 install --no-cache-dir --break-system-packages -e .

RUN mkdir -p outputs

# 环境里没有显示器：绘图走 Agg 后端（代码里已经设好），不会尝试开窗口
ENV MPLBACKEND=Agg
ENV PYTHONUNBUFFERED=1

CMD ["sleep", "infinity"]
