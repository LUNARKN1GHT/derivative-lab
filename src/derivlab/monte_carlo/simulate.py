from typing import Optional, Tuple, Union

import numpy as np

from derivlab.pricing.common import OptionType


def mc_european_price(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    option_type: Union[OptionType, str] = OptionType.CALL,
    q: float = 0.0,
    n_paths: int = 100_000,
    antithetic: bool = True,
    seed: Optional[int] = None,
) -> Tuple[float, float]:
    """蒙特卡洛模拟定价欧式期权,直接对到期时刻终值采样(GBM 终值分布已知,
    无需逐步模拟路径)。

    返回 (price, std_error),std_error 用于构造置信区间,
    如 price ± 1.96 * std_error 为 95% 置信区间。
    """
    option_type = OptionType(option_type)
    rng = np.random.default_rng(seed)

    drift = (r - q - 0.5 * sigma**2) * T
    diffusion_coef = sigma * np.sqrt(T)

    if antithetic:
        half_n = n_paths // 2
        z = rng.standard_normal(half_n)
        z = np.concatenate([z, -z])
    else:
        z = rng.standard_normal(n_paths)

    S_T = S * np.exp(drift + diffusion_coef * z)

    if option_type is OptionType.CALL:
        payoffs = np.maximum(S_T - K, 0.0)
    else:
        payoffs = np.maximum(K - S_T, 0.0)

    discounted = np.exp(-r * T) * payoffs
    price = discounted.mean()
    std_error = discounted.std(ddof=1) / np.sqrt(len(discounted))

    return price, std_error
