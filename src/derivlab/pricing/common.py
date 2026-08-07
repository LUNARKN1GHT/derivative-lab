"""实验中涉及到的"""

from enum import Enum


class OptionType(str, Enum):
    # 期权类型
    CALL = "call"  # 看涨期权
    PUT = "put"  # 看跌期权


class ExerciseStyle(str, Enum):
    # 处理的期权类型
    EUROPEAN = "european"  # 欧式期权
    AMERICAN = "american"  # 美式期权
