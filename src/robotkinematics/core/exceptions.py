"""@file exceptions.py
@brief 领域异常。

【异常策略】domain 层遇到不合法输入或数学上无解就抛，不打印、不兜底 —— 这一层不知道
错误该怎么呈现给用户（终端表格？图？），呈现是 interfaces 层的事。
所有异常都继承 KinematicsError，顶层 main() 只捕获这一个基类。
"""

from __future__ import annotations


class KinematicsError(Exception):
    """本工程所有可预期错误的基类。"""


class InvalidRotationError(KinematicsError):
    """旋转矩阵不合法（非正交、det≠1）或旋转参数无法构造出合法旋转。"""


class UnreachableTargetError(KinematicsError):
    """目标点/位姿超出机器人工作空间 —— 这是物理限制，不是程序 bug。"""


class NotConvergedError(KinematicsError):
    """数值 IK 在给定迭代次数内没收敛（初值太远或目标在奇异点附近）。"""


class SingularPoseError(KinematicsError):
    """当前位姿奇异，Jacobian 不可逆，无法解出唯一的关节微调量。"""
