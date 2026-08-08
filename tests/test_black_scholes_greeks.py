import numpy as np
import pytest

from derivlab import (
    OptionType,
    bsm_delta,
    bsm_gamma,
    bsm_price,
    bsm_rho,
    bsm_theta,
    bsm_vega,
)

# Hull 教材经典参数：S=42, K=40, r=10%, sigma=20%, T=0.5, q=0
S, K, T, R, SIGMA, Q = 42.0, 40.0, 0.5, 0.10, 0.20, 0.0


def test_delta_call_known_value():
    assert bsm_delta(S, K, T, R, SIGMA, "call") == pytest.approx(0.7791, abs=1e-3)


def test_delta_put_known_value():
    assert bsm_delta(S, K, T, R, SIGMA, "put") == pytest.approx(-0.2209, abs=1e-3)


def test_gamma_known_value():
    assert bsm_gamma(S, K, T, R, SIGMA) == pytest.approx(0.04996267, abs=1e-8)


def test_vega_known_value():
    assert bsm_vega(S, K, T, R, SIGMA) == pytest.approx(8.81341506, abs=1e-8)


def test_theta_known_value():
    # 按 Hull 惯例 theta 是时间流逝（年化）对期权价值的影响，通常为负
    assert bsm_theta(S, K, T, R, SIGMA, "call") == pytest.approx(-4.559092, abs=1e-4)
    assert bsm_theta(S, K, T, R, SIGMA, "put") == pytest.approx(-0.754174, abs=1e-4)


def test_rho_known_value():
    assert bsm_rho(S, K, T, R, SIGMA, "call") == pytest.approx(13.982046, abs=1e-4)
    assert bsm_rho(S, K, T, R, SIGMA, "put") == pytest.approx(-5.042543, abs=1e-4)


def test_option_type_accepts_enum_and_string():
    assert bsm_delta(S, K, T, R, SIGMA, OptionType.CALL) == pytest.approx(
        bsm_delta(S, K, T, R, SIGMA, "call"), abs=1e-12
    )


PARAMS = [
    (100, 100, 1.0, 0.05, 0.2, 0.0),
    (50, 60, 0.5, 0.03, 0.35, 0.01),
    (120, 100, 2.0, 0.01, 0.15, 0.02),
]


@pytest.mark.parametrize("S,K,T,r,sigma,q", PARAMS)
def test_delta_put_call_parity(S, K, T, r, sigma, q):
    call = bsm_delta(S, K, T, r, sigma, "call", q)
    put = bsm_delta(S, K, T, r, sigma, "put", q)
    assert call - put == pytest.approx(np.exp(-q * T), abs=1e-10)


@pytest.mark.parametrize("S,K,T,r,sigma,q", PARAMS)
def test_rho_put_call_parity(S, K, T, r, sigma, q):
    call = bsm_rho(S, K, T, r, sigma, "call", q)
    put = bsm_rho(S, K, T, r, sigma, "put", q)
    assert call - put == pytest.approx(K * T * np.exp(-r * T), abs=1e-8)


@pytest.mark.parametrize("S,K,T,r,sigma,q", PARAMS)
def test_theta_put_call_parity(S, K, T, r, sigma, q):
    call = bsm_theta(S, K, T, r, sigma, "call", q)
    put = bsm_theta(S, K, T, r, sigma, "put", q)
    expected = q * S * np.exp(-q * T) - r * K * np.exp(-r * T)
    assert call - put == pytest.approx(expected, abs=1e-8)


def test_delta_bounds_and_monotonicity():
    strikes = np.linspace(80, 120, 5)
    call_deltas = [bsm_delta(s, 100, 1.0, 0.05, 0.2, "call") for s in strikes]
    put_deltas = [bsm_delta(s, 100, 1.0, 0.05, 0.2, "put") for s in strikes]
    assert all(0 < d < 1 for d in call_deltas)
    assert all(-1 < d < 0 for d in put_deltas)
    assert call_deltas == sorted(call_deltas)
    assert put_deltas == sorted(put_deltas)


def test_delta_extremes():
    assert bsm_delta(1e6, 1.0, 1.0, 0.05, 0.2, "call") == pytest.approx(1.0, abs=1e-6)
    assert bsm_delta(1e-6, 1.0, 1.0, 0.05, 0.2, "call") == pytest.approx(0.0, abs=1e-6)
    assert bsm_delta(1e-6, 1.0, 1.0, 0.05, 0.2, "put") == pytest.approx(-1.0, abs=1e-6)


def test_gamma_and_vega_are_positive():
    for s in (80.0, 100.0, 120.0):
        assert bsm_gamma(s, 100, 1.0, 0.05, 0.2) > 0
        assert bsm_vega(s, 100, 1.0, 0.05, 0.2) > 0


def test_rho_signs():
    assert bsm_rho(S, K, T, R, SIGMA, "call") > 0
    assert bsm_rho(S, K, T, R, SIGMA, "put") < 0


def test_theta_negative_for_plain_options():
    assert bsm_theta(100, 100, 1.0, 0.05, 0.2, "call") < 0
    assert bsm_theta(100, 100, 1.0, 0.05, 0.2, "put") < 0


def test_delta_matches_price_finite_difference():
    h = 1e-4
    fd = (
        bsm_price(S + h, K, T, R, SIGMA, "call")
        - bsm_price(S - h, K, T, R, SIGMA, "call")
    ) / (2 * h)
    assert bsm_delta(S, K, T, R, SIGMA, "call") == pytest.approx(fd, rel=1e-5)


def test_gamma_matches_delta_finite_difference():
    h = 1e-3
    fd = (
        bsm_delta(S + h, K, T, R, SIGMA, "call")
        - bsm_delta(S - h, K, T, R, SIGMA, "call")
    ) / (2 * h)
    assert bsm_gamma(S, K, T, R, SIGMA) == pytest.approx(fd, rel=1e-4)


def test_vega_matches_price_finite_difference():
    h = 1e-4
    fd = (
        bsm_price(S, K, T, R, SIGMA + h, "call")
        - bsm_price(S, K, T, R, SIGMA - h, "call")
    ) / (2 * h)
    assert bsm_vega(S, K, T, R, SIGMA) == pytest.approx(fd, rel=1e-5)


def test_rho_matches_price_finite_difference():
    h = 1e-4
    fd = (
        bsm_price(S, K, T, R + h, SIGMA, "call")
        - bsm_price(S, K, T, R - h, SIGMA, "call")
    ) / (2 * h)
    assert bsm_rho(S, K, T, R, SIGMA, "call") == pytest.approx(fd, rel=1e-5)


def test_theta_matches_price_finite_difference():
    # theta 是 ∂V/∂t（时间流逝），而价格是到期时间的函数，∂V/∂t = -∂V/∂T
    h = 1e-5
    fd = -(
        bsm_price(S, K, T + h, R, SIGMA, "call")
        - bsm_price(S, K, T - h, R, SIGMA, "call")
    ) / (2 * h)
    assert bsm_theta(S, K, T, R, SIGMA, "call") == pytest.approx(fd, rel=1e-4)
