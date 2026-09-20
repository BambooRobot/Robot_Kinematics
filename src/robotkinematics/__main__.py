"""@file __main__.py

@brief 支持 `python -m robotkinematics ...`（等价于 `pip install -e .` 之后的 `rkin ...`）。
"""

from __future__ import annotations

import sys

from .main import main

if __name__ == "__main__":
    sys.exit(main())
