import numpy as np
import pytest

from derivlab import (
    forward_price,
    forward_price_cost_of_carry,
    forward_price_known_income,
    forward_value,
)


def test_forward_price_no_dividend():
    price = forward_price(S=100, r=0.05, T=1.0)
    assert price == pytest.approx(100 * np.exp(0.05), abs=1e-8)


def test_forward_price_increases_with_time_and_rate():
    base = forward_price(S=100, r=0.05, T=1.0)
    longer_T = forward_price(S=100, r=0.05, T=2.0)
    higher_r = forward_price(S=100, r=0.08, T=1.0)
    assert longer_T > base
    assert higher_r > base


def test_forward_price_matches_cost_of_carry_when_c_equals_r_minus_q():
    S, r, T, q = 100, 0.05, 1.0, 0.02
    price_dividend_form = forward_price(S, r, T, q)
    price_cost_of_carry_form = forward_price_cost_of_carry(S, c=r - q, T=T)
    assert price_dividend_form == pytest.approx(price_cost_of_carry_form, abs=1e-8)


def test_forward_price_known_income_lowers_price():
    S, r, T = 100, 0.05, 1.0
    price_no_income = forward_price_known_income(S, r, T, income_pv=0.0)
    price_with_income = forward_price_known_income(S, r, T, income_pv=5.0)
    assert price_with_income < price_no_income
    assert price_no_income == pytest.approx(forward_price(S, r, T), abs=1e-8)


def test_forward_value_zero_when_k_equals_current_forward_price():
    F0 = forward_price(S=100, r=0.05, T=1.0)
    value = forward_value(F0=F0, K=F0, r=0.05, T=1.0)
    assert value == pytest.approx(0.0, abs=1e-8)


def test_forward_value_positive_when_current_price_above_contract_price():
    F0 = forward_price(S=100, r=0.05, T=1.0)
    value = forward_value(F0=F0, K=F0 - 10, r=0.05, T=1.0)
    assert value > 0
