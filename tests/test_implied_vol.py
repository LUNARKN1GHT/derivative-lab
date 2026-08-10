import pytest

from derivlab import ImpliedVolatilityError, bsm_price, implied_volatility


def test_round_trip_recovers_known_sigma():
    S, K, T, r, true_sigma = 100, 100, 1.0, 0.05, 0.25
    price = bsm_price(S, K, T, r, true_sigma, "call")

    recovered = implied_volatility(price, S, K, T, r, "call")
    assert recovered == pytest.approx(true_sigma, abs=1e-6)


@pytest.mark.parametrize("true_sigma", [0.05, 0.15, 0.5, 1.2])
def test_round_trip_across_vol_range(true_sigma):
    S, K, T, r = 100, 90, 0.5, 0.03
    price = bsm_price(S, K, T, r, true_sigma, "put")

    recovered = implied_volatility(price, S, K, T, r, "put")
    assert recovered == pytest.approx(true_sigma, abs=1e-4)


def test_deep_otm_falls_back_to_bisection():
    # 深度虚值 + 临近到期,vega 接近 0,牛顿法容易失效,验证兜底的二分法仍能收敛
    S, K, T, r, true_sigma = 100, 150, 0.05, 0.05, 0.3
    price = bsm_price(S, K, T, r, true_sigma, "call")

    recovered = implied_volatility(price, S, K, T, r, "call")
    assert recovered == pytest.approx(true_sigma, abs=1e-3)


def test_price_below_intrinsic_raises():
    S, K, T, r = 100, 80, 1.0, 0.05
    # call 的内在价值下界远高于这个价格,不存在能匹配的隐含波动率
    with pytest.raises(ImpliedVolatilityError):
        implied_volatility(price=1.0, S=S, K=K, T=T, r=r, option_type="call")
