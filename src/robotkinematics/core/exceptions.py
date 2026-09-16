"""@file exceptions.py
@brief 领域异常。

【异常策略】core / usecases 遇到不合法输入或"物理上做不到"就抛，不打印、不兜底 ——
那一层不知道错误该怎么呈现给用户。呈现是 cli 层的事。
所有异常都继承 KinematicsError，`cli/main.py` 只捕获这一个基类。

（换成调库之后，原来那些"旋转矩阵不合法""数值微分"之类的异常没有存在意义了 ——
 库自己会校验，所以只留下仍然属于本项目语义的这几种。）
"""

from __future__ import annotations


class KinematicsError(Exception):
    """本工程所有可预期错误的基类。"""


class UnreachableTargetError(KinematicsError):
    """目标超出机器人工作空间 —— 这是物理限制，不是程序 bug。"""


class NotConvergedError(KinematicsError):
    """数值 IK 在给定迭代次数内没收敛（初值太远，或目标在奇异点附近）。"""


class SingularPoseError(KinematicsError):
    """当前位姿奇异，解不出可用的关节微调量。"""
