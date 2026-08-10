from typing import Union

from derivlab import OptionType, bsm_price, bsm_vega


class ImpliedVolatilityError(ValueError):
    """当给定的期权价格无法用 BSM 反解出合理的隐含波动率时抛出。"""


def implied_volatility(
    price: float,
    S: float,
    K: float,
    T: float,
    r: float,
    option_type: Union[OptionType, str] = OptionType.CALL,
    q: float = 0.0,
    tol: float = 1e-8,
    max_iter: int = 100,
) -> float:
    """给定市场期权价格,反解 BSM 隐含波动率。

    优先用 Newton-Raphson 迭代(用 vega 作导数,收敛快);
    如果某一步 vega 太小导致牛顿法不稳定,退化到二分法兜底,保证收敛。
    """
    option_type = OptionType(option_type)

    intrinsic = _intrinsic_value(S, K, T, r, option_type, q)
    if price < intrinsic - tol:
        raise ImpliedVolatilityError(
            f"price={price} 低于内在价值下界 {intrinsic},不存在能匹配的隐含波动率"
        )

    sigma = 0.3  # 初始猜测值
    for _ in range(max_iter):
        model_price = bsm_price(S, K, T, r, sigma, option_type, q)
        diff = model_price - price
        if abs(diff) < tol:
            return sigma

        vega = bsm_vega(S, K, T, r, sigma, q)
        if vega < 1e-8:
            break
        sigma = sigma - diff / vega
        if sigma <= 0:
            break

    return _bisection(price, S, K, T, r, option_type, q, tol, max_iter)


def _intrinsic_value(S, K, T, r, option_type, q) -> float:
    import numpy as np

    if option_type is OptionType.CALL:
        return max(S * np.exp(-q * T) - K * np.exp(-r * T), 0.0)
    else:
        return max(K * np.exp(-r * T) - S * np.exp(-q * T), 0.0)


def _bisection(price, S, K, T, r, option_type, q, tol, max_iter) -> float:
    lo, hi = 1e-6, 5.0
    for _ in range(max_iter):
        mid = (lo + hi) / 2
        model_price = bsm_price(S, K, T, r, mid, option_type, q)
        if abs(model_price - price) < tol:
            return mid
        if model_price > price:
            hi = mid
        else:
            lo = mid
    return (lo + hi) / 2
