"""@file __init__.py
@brief robotkinematics：自研 SE3 与串联机器人运动学。

分层：interfaces -> application -> infrastructure -> domain，依赖只允许向下。
domain 层是纯数学（除 numpy 外零依赖、零 IO），也是本项目最该被读懂的部分。
"""

__version__ = "0.2.0"
