import numpy as np
import pytest

from derivlab import bsm_price


def test_bsm_call_known_value():
    price = bsm_price(S=42, K=40, T=0.5, r=0.10, sigma=0.20, option_type="call")
    assert price == pytest.approx(4.76, abs=0.01)


def test_bsm_put_known_value():
    price = bsm_price(S=42, K=40, T=0.5, r=0.10, sigma=0.20, option_type="put")
    assert price == pytest.approx(0.81, abs=0.01)


@pytest.mark.parametrize(
    "S,K,T,r,sigma,q",
    [
        (100, 100, 1.0, 0.05, 0.2, 0.0),
        (50, 60, 0.5, 0.03, 0.35, 0.01),
        (120, 100, 2.0, 0.01, 0.15, 0.02),
    ],
)
def test_put_call_parity(S, K, T, r, sigma, q):
    call = bsm_price(S, K, T, r, sigma, "call", q)
    put = bsm_price(S, K, T, r, sigma, "put", q)
    lhs = call - put

    rhs_value = S * np.exp(-q * T) - K * np.exp(-r * T)
    assert lhs == pytest.approx(rhs_value, abs=1e-8)


def test_call_converges_to_intrinsic_as_t_approaches_zero():
    price = bsm_price(S=50, K=40, T=1e-6, r=0.05, sigma=0.2, option_type="call")
    assert price == pytest.approx(10.0, abs=0.01)
