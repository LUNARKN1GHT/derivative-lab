import pytest

from derivlab import crr_binomial_price, bsm_price


def test_european_binomial_converges_to_bsm():
    price = crr_binomial_price(
        S=42,
        K=40,
        T=0.5,
        r=0.10,
        sigma=0.20,
        n_steps=200,
        option_type="call",
        exercise="european",
    )
    bsm = bsm_price(S=42, K=40, T=0.5, r=0.10, sigma=0.20, option_type="call")
    assert price == pytest.approx(bsm, abs=0.05)


def test_american_call_equals_european_call_no_dividend():
    kwargs = dict(S=50, K=45, T=1.0, r=0.05, sigma=0.3, n_steps=200, option_type="call")
    american = crr_binomial_price(**kwargs, exercise="american")
    european = crr_binomial_price(**kwargs, exercise="european")
    assert american == pytest.approx(european, abs=1e-6)


def test_american_put_at_least_as_valuable_as_european_put():
    kwargs = dict(S=50, K=52, T=2.0, r=0.05, sigma=0.3, n_steps=200, option_type="put")
    american = crr_binomial_price(**kwargs, exercise="american")
    european = crr_binomial_price(**kwargs, exercise="european")
    assert american >= european - 1e-9
