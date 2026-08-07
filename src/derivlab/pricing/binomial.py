from typing import Union

import numpy as np

from derivlab.pricing.common import OptionType, ExerciseStyle


def crr_binomial_price(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    n_steps: int,
    option_type: Union[OptionType, str] = OptionType.CALL,
    exercise: Union[ExerciseStyle, str] = ExerciseStyle.EUROPEAN,
    q: float = 0.0,
) -> float | int:
    """
    Cox-Ross-Rubinstein 二叉树定价，支持美式/欧式行权方式

    Args:
        S: 标的现价
        K: 行权价
        T: 到期时间（年）
        r: 无风险利率（连续复利）
        sigma: 波动率
        n_steps: 二叉树步数
        option_type: 期权类型，看涨/看跌
        exercise: 行权方式，美式/欧式
        q: 连续股息率

    Returns:
        基于二叉树得到的定价
    """
    option_type = OptionType(option_type)
    exercise = ExerciseStyle(exercise)

    dt = T / n_steps  # 单步时间间隔
    u = np.exp(sigma * np.sqrt(dt))  # 上涨比率
    d = 1 / u  # 下跌比率，在对数空间里假设二者只和为 0
    disc = np.exp(-r * dt)  # 折现率
    p = (np.exp((r - q) * dt) - d) / (u - d)  # 上涨概率

    # 终端股价（向量化）：第 j 层有 j 次上涨、(n_steps - j) 次下跌
    j = np.arange(n_steps + 1)
    terminal_prices = S * (u**j) * (d ** (n_steps - j))  # 终端价格，指数形式上涨与下跌

    # 边界条件
    if option_type is OptionType.CALL:
        values = np.maximum(terminal_prices - K, 0.0)
    else:
        values = np.maximum(K - terminal_prices, 0.0)

    # 反向归纳
    for step in range(n_steps - 1, -1, -1):
        values = disc * (p * values[1:] + (1 - p) * values[:-1])

        if exercise is ExerciseStyle.AMERICAN:
            # 美式股权另外计算，需要考虑中间步的股息
            j = np.arange(step + 1)
            prices_at_step = S * (u**j) * (d ** (step - j))
            if option_type is OptionType.CALL:
                intrinsic = np.maximum(prices_at_step - K, 0.0)
            else:
                intrinsic = np.maximum(K - prices_at_step, 0.0)
            values = np.maximum(values, intrinsic)

    return values[0]
