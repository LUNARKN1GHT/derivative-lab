import pytest

from derivlab import (
    bsm_delta,
    bsm_gamma,
    bsm_price,
    bsm_rho,
    bsm_theta,
    bsm_vega,
)

S, K, T, r, sigma = 42, 40, 0.5, 0.10, 0.20
H = 1e-4


def test_delta_matches_finite_difference():
    fd = (
        bsm_price(S + H, K, T, r, sigma, "call")
        - bsm_price(S - H, K, T, r, sigma, "call")
    ) / (2 * H)
    assert bsm_delta(S, K, T, r, sigma, "call") == pytest.approx(fd, abs=1e-4)


def test_gamma_matches_finite_difference():
    fd = (
        bsm_price(S + H, K, T, r, sigma, "call")
        - 2 * bsm_price(S, K, T, r, sigma, "call")
        + bsm_price(S - H, K, T, r, sigma, "call")
    ) / (H**2)
    assert bsm_gamma(S, K, T, r, sigma) == pytest.approx(fd, abs=1e-2)


def test_vega_matches_finite_difference():
    fd = (
        bsm_price(S, K, T, r, sigma + H, "call")
        - bsm_price(S, K, T, r, sigma - H, "call")
    ) / (2 * H)
    assert bsm_vega(S, K, T, r, sigma) == pytest.approx(fd, abs=1e-4)


def test_rho_matches_finite_difference():
    fd = (
        bsm_price(S, K, T, r + H, sigma, "call")
        - bsm_price(S, K, T, r - H, sigma, "call")
    ) / (2 * H)
    assert bsm_rho(S, K, T, r, sigma, "call") == pytest.approx(fd, abs=1e-4)


def test_theta_matches_negative_finite_difference():
    fd_dT = (
        bsm_price(S, K, T + H, r, sigma, "call")
        - bsm_price(S, K, T - H, r, sigma, "call")
    ) / (2 * H)
    assert bsm_theta(S, K, T, r, sigma, "call") == pytest.approx(-fd_dT, abs=1e-4)


def test_gamma_and_vega_same_for_call_and_put():
    assert bsm_gamma(S, K, T, r, sigma) == bsm_gamma(
        S, K, T, r, sigma
    )  # gamma has no option_type param
    assert bsm_vega(S, K, T, r, sigma) == bsm_vega(
        S, K, T, r, sigma
    )  # vega has no option_type param


def test_delta_put_call_parity():
    import numpy as np

    q = 0.0
    delta_call = bsm_delta(S, K, T, r, sigma, "call", q)
    delta_put = bsm_delta(S, K, T, r, sigma, "put", q)
    assert delta_call - delta_put == pytest.approx(np.exp(-q * T), abs=1e-8)
