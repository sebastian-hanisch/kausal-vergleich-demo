"""Presets, Reglergrenzen, Permalink-Raster und Konstanten."""

import pytest

import kvl_constants as K
import kvl_evaluation as ev
import kvl_presets as P
import kvl_strategies as S
import kvl_validity as V
import kvl_world as W

FIRST = "Verborgene Ursache, Anstoß da (Voreinstellung)"


def test_every_preset_has_help_and_all_keys():
    assert set(K.PRESETS) == set(K.PRESET_HELP) == set(K.SCENARIOS) and len(K.PRESETS) == 9
    for name, p in K.PRESETS.items():
        assert set(p) == {"knobs", "n", "seed"} and set(p["knobs"]) == set(K.KNOBS) and K.PRESET_HELP[name]


def test_preset_values_are_valid_and_on_the_slider_grid():
    for name, p in K.PRESETS.items():
        for k, v in p["knobs"].items():
            spec = P.SETTING_SPECS[f"{k}_slider"]
            assert spec.lo <= v <= spec.hi and P.snap(v, spec) == pytest.approx(v), (name, k)
        assert P.snap(p["n"], P.SETTING_SPECS["n_slider"]) == p["n"] and 0 <= p["seed"] <= K.SEED_MAX


def test_first_preset_is_the_default_setting():
    assert next(iter(K.PRESETS)) == FIRST == K.DEFAULT_SCENARIO and ev.settings_from(K.PRESETS[FIRST]) == ev.Settings()
    assert all(P.SETTING_SPECS[f"{k}_slider"].default == v[0] for k, v in K.KNOBS.items()) and P.SETTING_SPECS["n_slider"].default == K.DEFAULT_N and P.SETTING_SPECS["seed_input"].default == K.DEFAULT_SEED


def test_presets_differ_and_each_changes_at_most_three_knobs():
    base = K.PRESETS[FIRST]["knobs"]
    assert len({tuple(sorted(p["knobs"].items())) + (p["seed"],) for p in K.PRESETS.values()}) == 9
    for name, p in K.PRESETS.items():
        assert len([k for k in base if p["knobs"][k] != base[k]]) <= 3, name
    assert {n: p["seed"] for n, p in K.PRESETS.items() if p["seed"] != K.DEFAULT_SEED} == {"Schwacher Anstoß": 2}


def test_the_presets_show_what_their_help_says():
    a = {name: ev.analyse(ev.settings_from(p)) for name, p in K.PRESETS.items()}
    lim = {name: S.limits(p["knobs"]) for name, p in K.PRESETS.items()}
    d = lim[FIRST]
    assert d["Backdoor {Z, W}"][0] == pytest.approx(0.22, abs=0.005) and a[FIRST].valid == ["Frontdoor", "2SLS"] and d["2SLS"][1] / d["Frontdoor"][1] == pytest.approx(1.9, abs=0.05)
    o = lim["Alles beobachtet"]
    assert a["Alles beobachtet"].valid == ["Backdoor {Z}", "Backdoor {Z, W}", "Frontdoor", "2SLS"] and o["2SLS"][1] / o["Backdoor {Z, W}"][1] == pytest.approx(2.2, abs=0.05) and o["alles anpassen"][0] == pytest.approx(-0.60, abs=0.005)
    assert a["Kein Anstoß"].valid == ["Frontdoor"] and a["Kein Anstoß"].rows[5].covers and a["Kein Anstoß"].rows[5].hi - a["Kein Anstoß"].rows[5].lo > 1
    assert a["Nichts brauchbar"].valid == [] and not any(r.covers for r in a["Nichts brauchbar"].rows[:5])
    weak = a["Schwacher Anstoß"]
    assert weak.first_stage_f == pytest.approx(6.0, abs=0.1) and "2SLS" in weak.valid and weak.rows[5].hi - weak.rows[5].lo == pytest.approx(1.7, abs=0.2)
    assert lim["Anstoß wirkt direkt"]["2SLS"][0] == pytest.approx(0.60, abs=0.005) and lim["Anstoß hängt an der Ursache"]["2SLS"][0] == pytest.approx(0.25, abs=0.005) == pytest.approx(lim["Anstoß hängt an der Ursache"]["Backdoor {Z, W}"][0], abs=0.005)
    assert lim["Teilweise vermittelt"]["Frontdoor"][0] == pytest.approx(-0.30, abs=0.005) and lim["Ursache wirkt auf den Mediator"]["Frontdoor"][0] == pytest.approx(0.21, abs=0.005)
    assert a["Teilweise vermittelt"].valid == ["2SLS"] == a["Ursache wirkt auf den Mediator"].valid


def test_snap_clamps_and_rounds():
    a = P.SETTING_SPECS["a_slider"]
    assert P.snap(9.0, a) == 1.0 and P.snap(-1.0, a) == 0.0 and P.snap(0.34, a) == pytest.approx(0.3) and P.snap(0.36, a) == pytest.approx(0.4)
    pi = P.SETTING_SPECS["pi_slider"]
    assert P.snap(0.07, pi) == pytest.approx(0.05) and P.snap(0.08, pi) == pytest.approx(0.10)
    c = P.SETTING_SPECS["c_slider"]
    assert P.snap(0.0, c) == pytest.approx(0.2) and P.snap(0.57, c) == pytest.approx(0.6) and P.snap(0.53, c) == pytest.approx(0.5)
    n = P.SETTING_SPECS["n_slider"]
    assert P.snap(50, n) == K.N_MIN and P.snap(10 ** 6, n) == K.N_MAX and P.snap(1049, n) == 1000 and P.snap(1051, n) == 1100 and isinstance(P.snap(1051, n), int)


def test_bounds_defaults_and_params():
    assert P.bounds("n_slider") == (K.N_MIN, K.N_MAX) and P.bounds("seed_input") == (0, K.SEED_MAX) and P.bounds("pi_slider") == (0.0, 1.0) and P.bounds("c_slider") == (0.2, 1.0)
    params = [s.url_param for s in P.SETTING_SPECS.values()]
    assert sorted(params) == sorted(["a", "b", "pi", "c", "e", "f", "g", "h", "n", "seed"]) and len(set(params)) == 10
    assert K.DEFAULT_N in range(K.N_MIN, K.N_MAX + 1, K.N_STEP) and P.PRESET_KEYS["seed"] == "seed_input" and P.PRESET_KEYS["a"] == "a_slider"


def test_knob_tables_are_complete_and_consistent():
    assert set(K.KNOB_LABELS) == set(K.KNOB_HELP) == set(K.KNOBS)
    for k, (default, lo, hi, step) in K.KNOBS.items():
        assert lo <= default <= hi and step > 0 and round((default - lo) / step, 6) == round((default - lo) / step)
    assert K.STRATEGIES == tuple(K.STRATEGY_HELP) == tuple(K.STRATEGY_COLORS) and set(K.MEASURE_HELP) == set(K.MEASURES)
    assert {key for key, _, _ in K.ASSUMPTIONS} == {"backdoor", "frontdoor", "instrument"} and all(set(ms) <= set(K.STRATEGIES) for _, _, ms in K.ASSUMPTIONS)
    assert K.VALID_TOL == 0.02 and set(K.VORTEST_SCENARIOS) <= set(K.PRESETS) and K.EXP_CROSS_S == tuple(sorted(K.EXP_CROSS_S)) and K.EXP_CURVE_NS == tuple(sorted(K.EXP_CURVE_NS))


def test_experiment_constants_are_consistent():
    assert K.EXP_MC_REPS >= 200 and K.EXP_VORTEST_REPS >= 300 and K.EXP_VORTEST_NS == (200, 1000) and K.FIRST_STAGE_F == 10.0 and K.SEED_BASE > 10 ** 6
    assert set(W.edge_weights(K.knobs())) | {(K.U, K.M), (K.X, K.Y), (K.I, K.Y), (K.U, K.I)} == set(W.edge_weights(K.knobs(f=0.1, e=0.1, g=0.1, h=0.1)))
    assert V.validity(K.knobs())["Frontdoor"]
