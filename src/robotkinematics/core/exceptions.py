"""@file exceptions.py

@brief 领域异常。

【异常策略】core / usecases 遇到不合法输入或"物理上做不到"就抛，不打印、不兜底 ——
那一层不知道错误该怎么呈现给用户。呈现是 cli 层的事。
所有异常都继承 KinematicsError，`main.py`（唯一入口）只捕获这一个基类。

（换库之后精简过两轮：库自己会校验的、以及改成"返回结论而不抛异常"的，都删了。
 现在只剩两种真正还会抛的情况：输入不合法、目标物理上够不着。）
"""

from __future__ import annotations


class KinematicsError(Exception):
    """本工程所有可预期错误的基类。"""


class UnreachableTargetError(KinematicsError):
    """目标超出机器人工作空间 —— 这是物理限制，不是程序 bug。"""
