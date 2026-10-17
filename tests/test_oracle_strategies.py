"""Dauerhafte Orakel mit unabhängiger zweiter Rechnung: (1) der exakte Grenzwert jeder Strategie stimmt mit einer großen Simulation überein; (2) die Gültigkeit am Diagramm (Backdoor-, Frontdoor-, Instrument-Kriterium) stimmt über ein Gitter von
64 Welten mit dem exakten Grenzwert-Bias null überein (zwei verschiedene Wege zur selben Aussage); (3) die exakte Streuung je √n stimmt mit der Streuung aus 300 festen Läufen überein; (4) der Vortest hat die Größe 5 % bei erfüllter Nullhypothese;
(5) die Zeilen der Analyse und der Experimente sind dieselben Zahlen."""

import itertools
import math

import numpy as np
import pytest

import kvl_constants as K
import kvl_evaluation as ev
import kvl_strategies as S
import kvl_validity as V
import kvl_world as W
from kausal_generator import SplitMix64


def _grid():
    for a, pi, e, f, g, h, c in itertools.product((0.0, 0.4), (0.0, 0.5), (0.0, 0.3), (0.0, 0.4), (0.0, 0.3), (0.0, 0.3), (1.0,)):
        yield K.knobs(a=a, b=a, pi=pi, e=e, f=f, g=g, h=h, c=c)


def test_limits_agree_with_a_large_simulation():
    for name in ("Verborgene Ursache, Anstoß da (Voreinstellung)", "Alles beobachtet", "Anstoß wirkt direkt", "Teilweise vermittelt", "Ursache wirkt auf den Mediator", "Anstoß hängt an der Ursache"):
        k = K.PRESETS[name]["knobs"]
        lim = S.limits(k)
        est = S.estimates(np.cov(W.simulate(k, 400000, SplitMix64(91)), rowvar=False), 400000)
        for m in K.STRATEGIES:
            assert est[m][0] - W.truth(k) == pytest.approx(lim[m][0], abs=0.015), (name, m)


def test_validity_at_the_diagram_iff_the_exact_bias_is_zero_over_a_grid_of_worlds():
    count = 0
    for k in _grid():
        lim, val = S.limits(k), V.validity(k)
        for m in K.STRATEGIES:
            assert val[m] == (not math.isnan(lim[m][0]) and abs(lim[m][0]) < 1e-9), (k, m)
        count += 1
    assert count == 64


def test_the_exact_streuung_agrees_with_the_monte_carlo_standard_deviation():
    for k in (K.knobs(), K.knobs(a=0, b=0)):
        mc = S.monte_carlo(k, 1000, 300, 123)
        lim = S.limits(k)
        for m in ("Backdoor {Z}", "Backdoor {Z, W}", "Frontdoor", "2SLS"):
            assert mc[m]["sd"] == pytest.approx(lim[m][1] / math.sqrt(1000), rel=0.15), m
        assert all(mc[m]["width"] == pytest.approx(2 * 1.96 * lim[m][1] / math.sqrt(1000), rel=0.1) for m in ("Backdoor {Z, W}", "Frontdoor"))


def test_the_hausman_test_has_size_five_percent_when_the_instrument_is_valid_and_nothing_is_hidden():
    k = K.knobs(a=0, b=0)
    rejected = []
    for r in range(600):
        est = S.estimates(np.cov(W.simulate(k, 1000, SplitMix64(7000 + r)), rowvar=False), 1000)
        rejected.append(S.hausman_z(est) > 1.96)
    assert 0.025 <= np.mean(rejected) <= 0.085


def test_the_analysis_rows_are_the_exact_limits_and_the_sample_estimates():
    s = ev.Settings(n=800, seed=3)
    a = ev.analyse(s)
    cov = np.cov(W.simulate(s.knobs, 800, SplitMix64(3)), rowvar=False)
    est = S.estimates(cov, 800)
    lim = S.limits(s.knobs)
    for r in a.rows:
        assert r.estimate == pytest.approx(est[r.strategy][0]) and r.limit_bias == pytest.approx(lim[r.strategy][0]) and r.sd1 == pytest.approx(lim[r.strategy][1])
    res = ev.limits_experiment()
    assert res["Alles beobachtet"]["bias"]["naiv"] == pytest.approx(S.limits(K.knobs(a=0, b=0))["naiv"][0])
