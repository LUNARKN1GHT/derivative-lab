from typing import Union

import numpy as np
from scipy.stats import norm

from derivlab.pricing import OptionType, bsm_d1_d2


def bsm_delta(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    option_type: Union[OptionType, str] = OptionType.CALL,
    q: float = 0.0,
) -> float:
    """标的价格波动 1 单位，期权价格的变动量"""
    option_type = OptionType(option_type)
    d1, _ = bsm_d1_d2(S, K, T, r, sigma, q)
    if option_type is OptionType.CALL:
        return np.exp(-q * T) * norm.cdf(d1)
    else:
        return -np.exp(-q * T) * norm.cdf(-d1)


def bsm_gamma(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    q: float = 0.0,
) -> float:
    """delta 对标的价格变动率，call/put 通用同一公式"""
    d1, _ = bsm_d1_d2(S, K, T, r, sigma, q)
    return np.exp(-q * T) * norm.pdf(d1) / (S * sigma * np.sqrt(T))


def bsm_vega(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    q: float = 0.0,
) -> float:
    """
    波动率变动 1 单位(如 0.01 -> 0.02 对应变动量为 1),期权价格的变动量,call/put 通用。
    """
    d1, _ = bsm_d1_d2(S, K, T, r, sigma, q)
    return S * np.exp(-q * T) * norm.pdf(d1) * np.sqrt(T)


def bsm_theta(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    option_type: Union[OptionType, str] = OptionType.CALL,
    q: float = 0.0,
) -> float:
    """时间流逝一年,期权价格的变动量(年化,非按日)。"""
    option_type = OptionType(option_type)
    d1, d2 = bsm_d1_d2(S, K, T, r, sigma, q)
    term1 = -S * np.exp(-q * T) * norm.pdf(d1) * sigma / (2 * np.sqrt(T))

    if option_type is OptionType.CALL:
        term2 = q * S * np.exp(-q * T) * norm.cdf(d1)
        term3 = -r * K * np.exp(-r * T) * norm.cdf(d2)
    else:
        term2 = -q * S * np.exp(-q * T) * norm.cdf(-d1)
        term3 = r * K * np.exp(-r * T) * norm.cdf(-d2)

    return term1 + term2 + term3


def bsm_rho(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    option_type: Union[OptionType, str] = OptionType.CALL,
    q: float = 0.0,
) -> float:
    """无风险利率变动 1 单位(如 0.01 -> 0.02 对应变动量为 1),期权价格的变动量。"""
    option_type = OptionType(option_type)
    _, d2 = bsm_d1_d2(S, K, T, r, sigma, q)
    if option_type is OptionType.CALL:
        return K * T * np.exp(-r * T) * norm.cdf(d2)
    else:
        return -K * T * np.exp(-r * T) * norm.cdf(-d2)
