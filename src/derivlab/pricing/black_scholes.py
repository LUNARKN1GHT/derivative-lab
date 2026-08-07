from typing import Tuple, Union

import numpy as np

from derivlab.pricing.common import OptionType
from scipy.stats import norm


def bsm_d1_d2(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    q: float = 0.0,
) -> Tuple[float, float]:
    """计算BSM公式中的 d1，d2, 供定价及后续 Greeks 复用"""
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return d1, d2


def bsm_price(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    option_type: Union[OptionType, str] = OptionType.CALL,
    q: float = 0.0,
) -> float:
    """
    欧式期权 BSM 定价
    Args:
        S: 标的现价
        K: 行权价
        T: 到期时间（年）
        r: 无风险利率（连续复利）
        sigma: 波动率
        option_type: 期权类型，为看涨或看跌
        q: 连续股息率

    Returns:
        在欧式期权上利用 BSM 方程得到的定价
    """
    option_type = OptionType(option_type)
    d1, d2 = bsm_d1_d2(S, K, T, r, sigma, q)

    if option_type is OptionType.CALL:
        return S * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    else:
        return K * np.exp(-r * T) * norm.cdf(-d2) - S * np.exp(-q * T) * norm.cdf(-d1)
