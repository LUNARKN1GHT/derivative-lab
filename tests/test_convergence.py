import pytest

from derivlab import bsm_price, crr_binomial_price


@pytest.mark.parametrize("option_type", ["call", "put"])
@pytest.mark.parametrize("q", [0.0, 0.02])
def test_binomial_converges_to_bsm_as_steps_increase(option_type, q):
    S, K, T, r, sigma = 55, 50, 0.75, 0.04, 0.25

    bsm = bsm_price(S, K, T, r, sigma, option_type, q)

    error_small_steps = abs(
        crr_binomial_price(S, K, T, r, sigma, n_steps=10, option_type=option_type, q=q)
        - bsm
    )
    error_large_steps = abs(
        crr_binomial_price(
            S, K, T, r, sigma, n_steps=1000, option_type=option_type, q=q
        )
        - bsm
    )

    assert error_large_steps < error_small_steps
    assert error_large_steps < 1e-2
