"""Die Welt, die Gültigkeit am Diagramm und die Daten aus festen Strömen: Handwerte."""

import numpy as np
import pytest

import kvl_constants as K
import kvl_validity as V
import kvl_world as W
from kausal_generator import SplitMix64

Z, WW, I, U, X, M, Y, C = range(8)
DEFAULT = K.knobs()


def test_edges_and_truth_of_the_default_world():
    es = W.edge_weights(DEFAULT)
    assert es == {(Z, X): 0.8, (Z, Y): 0.6, (WW, Y): 0.6, (Y, C): 0.8, (U, X): 0.6, (U, Y): 0.6, (I, X): 0.5, (X, M): 1.0, (M, Y): 0.6}
    assert W.truth(DEFAULT) == pytest.approx(0.6) and W.truth(K.knobs(c=0.5, e=0.3)) == pytest.approx(0.6) and W.truth(K.knobs(e=0.3)) == pytest.approx(0.9)
    assert (U, M) not in es and (I, Y) not in es and (U, I) not in es and (X, Y) not in es


def test_zero_strengths_drop_the_arrow_and_the_parents_follow():
    es = W.edge_weights(K.knobs(a=0, b=0, pi=0))
    assert (U, X) not in es and (U, Y) not in es and (I, X) not in es
    p = W.parents_of(K.knobs(f=0.4, g=0.3, h=0.3, e=0.3))
    assert p == ((), (), (U,), (), (Z, I, U), (U, X), (Z, WW, I, U, X, M), (Y,))


def test_population_covariance_hand_values():
    cov = W.covariance(W.coefficient_matrix(K.knobs(a=0, b=0)))
    var_x = 0.8 ** 2 + 0.5 ** 2 + 1                                           # Z, I und das Rauschen
    assert cov[X, X] == pytest.approx(var_x) and cov[X, Z] == pytest.approx(0.8) and cov[I, X] == pytest.approx(0.5) and cov[M, M] == pytest.approx(var_x + 1)
    assert np.allclose(cov, cov.T) and np.all(np.linalg.eigvalsh(W.covariance(W.coefficient_matrix(DEFAULT))) > 0)


def test_simulate_is_deterministic_and_has_the_population_covariance():
    d1, d2 = W.simulate(DEFAULT, 2000, SplitMix64(5)), W.simulate(DEFAULT, 2000, SplitMix64(5))
    assert d1.shape == (2000, 8) and np.array_equal(d1, d2) and not np.array_equal(d1, W.simulate(DEFAULT, 2000, SplitMix64(6)))
    big = W.simulate(DEFAULT, 300000, SplitMix64(11))
    assert np.allclose(np.cov(big, rowvar=False), W.covariance(W.coefficient_matrix(DEFAULT)), atol=0.04)


def test_validity_of_each_strategy_in_the_nine_scenarios():
    expect = {                                   # naiv, Backdoor {Z}, {Z, W}, alles, Frontdoor, 2SLS
        "Verborgene Ursache, Anstoß da (Voreinstellung)": (0, 0, 0, 0, 1, 1), "Alles beobachtet": (0, 1, 1, 0, 1, 1), "Kein Anstoß": (0, 0, 0, 0, 1, 0), "Nichts brauchbar": (0, 0, 0, 0, 0, 0),
        "Schwacher Anstoß": (0, 0, 0, 0, 1, 1), "Anstoß wirkt direkt": (0, 0, 0, 0, 1, 0), "Anstoß hängt an der Ursache": (0, 0, 0, 0, 1, 0), "Teilweise vermittelt": (0, 0, 0, 0, 0, 1), "Ursache wirkt auf den Mediator": (0, 0, 0, 0, 0, 1),
    }
    assert set(expect) == set(K.PRESETS)
    for name, flags in expect.items():
        assert tuple(int(V.validity(K.PRESETS[name]["knobs"])[s]) for s in K.STRATEGIES) == flags, name


def test_reasons_name_the_violated_assumption():
    k = DEFAULT
    assert V.reason(k, "Frontdoor") == "" and V.reason(k, "2SLS") == ""
    assert V.reason(k, "Backdoor {Z, W}") == "offener Rückwärtspfad X ← U → Y" == V.reason(k, "Backdoor {Z}") and V.reason(k, "naiv") == "offener Rückwärtspfad X ← Z → Y"
    assert V.reason(K.knobs(a=0, b=0), "alles anpassen") == "enthält Nachkommen des Faktors (M, C)" and V.reason(K.knobs(a=0, b=0), "Backdoor {Z}") == ""
    assert V.reason(K.knobs(e=0.3, c=0.5), "Frontdoor") == "der Bestand fängt nicht alle Wege ab (direkter Pfeil)"
    assert V.reason(K.knobs(f=0.4), "Frontdoor") == "offener Rückweg vom Faktor zum Bestand (verborgene Ursache wirkt auf ihn)"
    assert V.reason(K.knobs(pi=0), "2SLS") == "kein Anstoß (nicht relevant)"
    assert V.reason(K.knobs(g=0.3), "2SLS") == "der Anstoß wirkt am Faktor vorbei oder hängt an einer gemeinsamen Ursache" == V.reason(K.knobs(h=0.3), "2SLS")
    with pytest.raises(ValueError):
        V.strategy_valid(DEFAULT, "Zufall")
