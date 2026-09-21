#!/bin/sh
# 把容器内的 apt / pip 源换成国内镜像（默认清华 TUNA），加速构建。
#
# 用法（构建时通过 --build-arg 传）：
#   docker build --build-arg APT_MIRROR=mirrors.tuna.tsinghua.edu.cn ...
#   docker build --build-arg PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple ...
set -eu

APT_MIRROR="${APT_MIRROR:-mirrors.tuna.tsinghua.edu.cn}"
PIP_INDEX_URL="${PIP_INDEX_URL:-https://pypi.tuna.tsinghua.edu.cn/simple}"
PIP_TRUSTED_HOST="${PIP_TRUSTED_HOST:-pypi.tuna.tsinghua.edu.cn}"

if [ -f /etc/apt/sources.list.d/debian.sources ]; then
    sed -i "s|deb.debian.org|${APT_MIRROR}|g; s|security.debian.org|${APT_MIRROR}|g" \
        /etc/apt/sources.list.d/debian.sources
elif [ -f /etc/apt/sources.list ]; then
    sed -i "s|deb.debian.org|${APT_MIRROR}|g; s|security.debian.org|${APT_MIRROR}|g" \
        /etc/apt/sources.list
fi

mkdir -p /etc/pip
cat > /etc/pip.conf <<EOF
[global]
index-url = ${PIP_INDEX_URL}
trusted-host = ${PIP_TRUSTED_HOST}
EOF

echo "已配置镜像: apt=${APT_MIRROR} pip=${PIP_INDEX_URL}"
