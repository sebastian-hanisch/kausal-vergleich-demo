"""Die sechs Strategien: Handwerte, Gegenrechnung mit lstsq, Grenzwerte, Kreuzungsgröße, Vortest."""

import math

import numpy as np
import pytest

import kvl_constants as K
import kvl_strategies as S
import kvl_world as W
from kausal_generator import SplitMix64

Z, WW, I, U, X, M, Y, C = range(8)
DEFAULT = K.knobs()
S1 = K.knobs(a=0, b=0)


def _cov(k, n=1500, seed=5):
    return np.cov(W.simulate(k, n, SplitMix64(seed)), rowvar=False), n


def test_ols_by_hand():
    x = np.array([1.0, 2, 3, 4])
    y = np.array([2.0, 1, 4, 3])
    S_ = np.cov(np.column_stack([x, y]), rowvar=False)
    beta, se = S.ols(S_, 1, [0], 4)
    assert beta[0] == pytest.approx(0.6) and se[0] == pytest.approx(math.sqrt(1.6 / 5))              # Sxy = 3, Sxx = 5, SSR = 3,2, σ² = 3,2 / 2


def test_regression_strategies_equal_lstsq_with_an_intercept():
    data = W.simulate(DEFAULT, 1500, SplitMix64(5))
    cov = np.cov(data, rowvar=False)
    est = S.estimates(cov, 1500)
    for name, cols in (("naiv", [X]), ("Backdoor {Z}", [X, Z]), ("Backdoor {Z, W}", [X, Z, WW]), ("alles anpassen", [X, Z, WW, I, M, C])):
        z = np.column_stack([np.ones(1500)] + [data[:, c] for c in cols])
        beta, *_ = np.linalg.lstsq(z, data[:, Y], rcond=None)
        res = data[:, Y] - z @ beta
        se = math.sqrt((res @ res / (1500 - z.shape[1])) * np.linalg.inv(z.T @ z)[1, 1])
        assert est[name][0] == pytest.approx(beta[1], abs=1e-9) and est[name][1] == pytest.approx(se, rel=1e-9), name


def test_frontdoor_is_the_product_of_two_regressions():
    data = W.simulate(DEFAULT, 1500, SplitMix64(7))
    est = S.estimates(np.cov(data, rowvar=False), 1500)
    c = np.linalg.lstsq(np.column_stack([np.ones(1500), data[:, X]]), data[:, M], rcond=None)[0][1]
    d = np.linalg.lstsq(np.column_stack([np.ones(1500), data[:, M], data[:, X]]), data[:, Y], rcond=None)[0][1]
    assert est["Frontdoor"][0] == pytest.approx(c * d, abs=1e-9)


def test_2sls_equals_the_two_stage_regression_with_covariates():
    data = W.simulate(DEFAULT, 1500, SplitMix64(9))
    est = S.estimates(np.cov(data, rowvar=False), 1500)
    first = np.column_stack([np.ones(1500), data[:, I], data[:, Z], data[:, WW]])
    xhat = first @ np.linalg.lstsq(first, data[:, X], rcond=None)[0]
    second = np.column_stack([np.ones(1500), xhat, data[:, Z], data[:, WW]])
    assert est["2SLS"][0] == pytest.approx(np.linalg.lstsq(second, data[:, Y], rcond=None)[0][1], abs=1e-9)


def test_estimates_never_read_the_hidden_cause():
    cov, n = _cov(DEFAULT)
    blind = cov.copy()
    blind[U, :] = np.nan
    blind[:, U] = np.nan
    a, b = S.estimates(cov, n), S.estimates(blind, n)
    assert all(a[m] == b[m] for m in K.STRATEGIES) and S.first_stage_f(cov, n) == S.first_stage_f(blind, n)


def test_limits_hand_values():
    lim = S.limits(S1)
    assert lim["naiv"][0] == pytest.approx(0.6 * 0.8 / 1.89, abs=1e-9) and lim["naiv"][0] == pytest.approx(0.254, abs=0.0006)           # Var(X) = 0,64 + 0,25 + 1
    assert lim["alles anpassen"][0] == pytest.approx(-0.6, abs=1e-9) and lim["Frontdoor"][0] == pytest.approx(0, abs=1e-9) and lim["2SLS"][0] == pytest.approx(0, abs=1e-9)
    assert lim["Backdoor {Z, W}"][1] == pytest.approx(math.sqrt(1.36 / 1.25), abs=1e-6) and lim["2SLS"][1] == pytest.approx(math.sqrt(1.36) / 0.5, abs=1e-6)  # σ² = 0,36 + 1; Var(X | Z, W) = π² + 1; Kov(I, X) = π
    hid = S.limits(DEFAULT)
    assert hid["Backdoor {Z}"][0] == pytest.approx(0.6 * 0.6 / (0.36 + 0.25 + 1), abs=1e-9) and hid["Backdoor {Z}"][0] == pytest.approx(0.2236, abs=0.00006)       # b Kov(X, U | Z) / Var(X | Z)
    assert S.limits(K.knobs(e=0.3, c=0.5))["Frontdoor"][0] == pytest.approx(-0.3, abs=1e-9) and S.limits(K.knobs(g=0.3))["2SLS"][0] == pytest.approx(0.6, abs=1e-9)
    assert math.isnan(S.limits(K.knobs(pi=0))["2SLS"][0]) and all(isinstance(v, float) for pair in S.limits(DEFAULT).values() for v in pair)


def test_crossing_size_is_where_the_two_asymptotic_rmse_meet():
    for s in (0.2, 0.3, 0.4, 0.6):
        lim = S.limits(K.knobs(a=s, b=s))
        bias, sd_bd = lim["Backdoor {Z, W}"]
        sd_iv = lim["2SLS"][1]
        n_star = S.crossing_n(bias, sd_bd, sd_iv)
        assert S.asymptotic_rmse(bias, sd_bd, n_star) == pytest.approx(S.asymptotic_rmse(0.0, sd_iv, n_star), rel=1e-9)
        assert S.asymptotic_rmse(bias, sd_bd, n_star / 2) < S.asymptotic_rmse(0.0, sd_iv, n_star / 2) and S.asymptotic_rmse(bias, sd_bd, n_star * 2) > S.asymptotic_rmse(0.0, sd_iv, n_star * 2)
    assert S.crossing_n(0.0, 1.0, 2.0) == math.inf and S.crossing_n(0.1, 2.0, 1.0) == 0.0 and S.crossing_n(0.1, 1.0, math.nan) == math.inf
    assert S.crossing_n(0.1, 1.0, 2.0) == pytest.approx(300.0)
    assert S.asymptotic_rmse(0.3, 2.0, 100) == pytest.approx(math.sqrt(0.09 + 0.04))


def test_hausman_statistic_by_hand():
    est = {"Backdoor {Z, W}": (0.5, 0.02), "2SLS": (0.7, 0.05)}
    assert S.hausman_z(est) == pytest.approx(0.2 / math.sqrt(0.0025 - 0.0004))
    assert S.hausman_z({"Backdoor {Z, W}": (0.5, 0.05), "2SLS": (0.7, 0.02)}) == pytest.approx(0.2 / math.sqrt(1e-12))      # Varianzdifferenz negativ: Untergrenze


def test_first_stage_f_is_the_squared_t_of_the_instrument():
    cov, n = _cov(DEFAULT)
    beta, se = S.ols(cov, X, [I, Z, WW], n)
    assert S.first_stage_f(cov, n) == pytest.approx((beta[0] / se[0]) ** 2) and S.first_stage_f(cov, n) > 100
    cov0, n0 = _cov(K.knobs(pi=0))
    assert S.first_stage_f(cov0, n0) < 10


def test_monte_carlo_structure_and_determinism():
    a = S.monte_carlo(DEFAULT, 400, 12, 77)
    b = S.monte_carlo(DEFAULT, 400, 12, 77)
    assert a == b and set(a) == set(K.STRATEGIES)
    for m, r in a.items():
        assert set(r) == {"bias", "sd", "rmse", "median_error", "cover", "width", "median", "q25", "q75"} and r["rmse"] >= abs(r["bias"]) - 1e-12 and r["q25"] <= r["median"] <= r["q75"] and 0 <= r["cover"] <= 1
    assert S.monte_carlo(DEFAULT, 400, 12, 78) != a


def test_2sls_standard_error_by_the_two_stage_formula():
    n = 1200
    data = W.simulate(DEFAULT, n, SplitMix64(21))
    est = S.estimates(np.cov(data, rowvar=False), n)
    ctrl = np.column_stack([np.ones(n), data[:, Z], data[:, WW]])

    def resid(v):
        return v - ctrl @ np.linalg.lstsq(ctrl, v, rcond=None)[0]

    xt, yt, it = resid(data[:, X]), resid(data[:, Y]), resid(data[:, I])
    beta = (it @ yt) / (it @ xt)
    xhat = it * (it @ xt) / (it @ it)
    u = yt - beta * xt
    se = math.sqrt((u @ u / (n - 1)) / (xhat @ xhat))
    assert est["2SLS"][0] == pytest.approx(beta, abs=1e-9) and est["2SLS"][1] == pytest.approx(se, rel=1e-9)


def test_frontdoor_criterion_finds_the_mediator_in_the_worlds_diagram():
    import frd_criterion as F
    import bdr_backdoor as B
    import ins_criterion as N

    parents = W.parents_of(DEFAULT)
    assert F.frontdoor_sets(parents, X, Y, K.OBSERVED) == [(M,), (WW, M)] and F.frontdoor_sets(W.parents_of(K.knobs(e=0.3, c=0.5)), X, Y, K.OBSERVED) == []
    assert N.instruments(parents, X, Y, K.OBSERVED) == [I] and N.instruments(W.parents_of(K.knobs(g=0.3)), X, Y, K.OBSERVED) == [] and N.frontdoor_sets(parents, X, Y, K.OBSERVED) == [(M,), (WW, M)]
    assert B.valid_sets(parents, X, Y, (Z, WW), "backdoor") == [] and B.valid_sets(W.parents_of(K.knobs(a=0, b=0)), X, Y, (Z, WW), "backdoor") == [(Z,), (Z, WW)]


def test_monte_carlo_uses_every_dataset():
    assert S.monte_carlo(DEFAULT, 300, 2, 5)["Frontdoor"]["sd"] > 0
    one, two = S.monte_carlo(DEFAULT, 300, 1, 5), S.monte_carlo(DEFAULT, 300, 2, 5)
    assert one["Frontdoor"]["sd"] == 0 and one["Frontdoor"]["median"] != two["Frontdoor"]["median"]
