import numpy as np


def forward_price(S: float, r: float, T: float, q: float = 0.0) -> float:
    """标的有连续股息率 q 时的远期/期货价格。q=0 时退化为无股息标的的基础公式。"""
    return S * np.exp((r - q) * T)


def forward_price_known_income(S: float, r: float, T: float, income_pv: float) -> float:
    """标的有已知现金收入(如已知股息)时的远期价格。income_pv 是收入的现值。"""
    return (S - income_pv) * np.exp(r * T)


def forward_price_cost_of_carry(S: float, c: float, T: float) -> float:
    """持有成本模型(常用于商品期货)。

    c 是净持有成本率(仓储成本 - 便利收益率等,调用者自行算好传入)。
    """
    return S * np.exp(c * T)


def forward_value(F0: float, K: float, r: float, T: float) -> float:
    """已存在的多头远期合约当前价值。

    F0 是当前市场远期价格,K 是合约约定交割价,T 是剩余期限。
    """
    return (F0 - K) * np.exp(-r * T)
