import pytest

from derivlab import bsm_price, mc_european_price


def test_mc_price_close_to_bsm():
    S, K, T, r, sigma = 42, 40, 0.5, 0.10, 0.20
    bsm = bsm_price(S, K, T, r, sigma, "call")

    price, std_error = mc_european_price(
        S, K, T, r, sigma, "call", n_paths=200_000, seed=42
    )
    # 用 3 倍标准误差作为置信区间容差,对应约 99.7% 置信水平
    assert price == pytest.approx(bsm, abs=3 * std_error)


@pytest.mark.parametrize("option_type", ["call", "put"])
def test_mc_price_close_to_bsm_put_and_call(option_type):
    S, K, T, r, sigma = 100, 95, 1.0, 0.03, 0.25
    bsm = bsm_price(S, K, T, r, sigma, option_type)

    price, std_error = mc_european_price(
        S, K, T, r, sigma, option_type, n_paths=200_000, seed=1
    )
    assert price == pytest.approx(bsm, abs=3 * std_error)


def test_seed_gives_reproducible_result():
    kwargs = dict(
        S=42,
        K=40,
        T=0.5,
        r=0.10,
        sigma=0.20,
        option_type="call",
        n_paths=10_000,
        seed=123,
    )
    price1, se1 = mc_european_price(**kwargs)
    price2, se2 = mc_european_price(**kwargs)
    assert price1 == price2
    assert se1 == se2
