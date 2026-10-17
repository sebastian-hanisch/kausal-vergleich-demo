"""Auswertung: Formatierung, Analyse einer Stichprobe, Checkliste, die fünf Experimente (Struktur und Invarianten)."""

import math

import numpy as np
import pytest

import kvl_constants as K
import kvl_evaluation as ev
import kvl_strategies as S

DEFAULT = ev.Settings()


def test_format_helpers():
    assert ev.de(1.5) == "1,50" and ev.de(-0.234, 3) == "−0,234" and ev.signed(0.384) == "+0,38" and ev.signed(-0.2341, 3) == "−0,234" and ev.signed(-1e-12, 3) == "+0,000"
    assert ev.signed(-4e-7, 3) == "+0,000" and ev.signed(-6e-7, 3) == "−0,000" and ev.signed(-(0.5 * 10 ** -3 * 1e-3), 3) == "−0,000"
    assert ev.pct(0.914) == "91 %" and ev.pct(0.0625, 1) == "6,2 %" and ev._n(1047.3) == "1 047" and ev._n(69739) == "69 739" and ev._n(math.inf) == "—" and ev._n(math.nan) == "—"
    assert ev.names_of(["a", "b"]) == "a, b" and ev.names_of([]) == "keine"


def test_settings_default_hashable_and_from_the_first_preset():
    assert DEFAULT.knobs == K.knobs() and DEFAULT.n == 1000 and DEFAULT.seed == K.DEFAULT_SEED and hash(DEFAULT) == hash(ev.Settings())
    assert ev.settings_from(K.PRESETS[K.DEFAULT_SCENARIO]) == DEFAULT and ev.settings_from(K.PRESETS["Kein Anstoß"]).pi == 0.0 and ev.settings_from(K.PRESETS["Kein Anstoß"]).knobs["a"] == 0.6


def test_analysis_of_the_default_sample():
    a = ev.analyse(DEFAULT)
    l = S.limits(K.knobs())
    assert a.truth == pytest.approx(0.6) and [r.strategy for r in a.rows] == list(K.STRATEGIES) and a.valid == ["Frontdoor", "2SLS"] and a.best == "Frontdoor"
    assert a.sd_ratio["Frontdoor"] == 1.0 and a.sd_ratio["2SLS"] == pytest.approx(l["2SLS"][1] / l["Frontdoor"][1], abs=1e-9) and a.sd_ratio["2SLS"] == pytest.approx(1.89, abs=0.01) and set(a.sd_ratio) == {"Frontdoor", "2SLS"}
    assert a.holds == {"backdoor": False, "frontdoor": True, "instrument": True} and a.first_stage_f > 100
    by = {r.strategy: r for r in a.rows}
    assert by["Frontdoor"].valid and not by["naiv"].valid and by["Backdoor {Z, W}"].limit_bias == pytest.approx(0.2236, abs=0.0006) and by["naiv"].reason == "offener Rückwärtspfad X ← Z → Y"
    assert all(r.lo < r.estimate < r.hi and r.hi - r.lo == pytest.approx(2 * 1.96 * r.se) and r.error == pytest.approx(r.estimate - 0.6) for r in a.rows)
    assert all(r.covers == (abs(r.error) <= 1.96 * r.se) for r in a.rows) and [r.covers for r in a.rows if not r.valid] == [False] * 4
    assert a.hausman_rejected == (a.hausman_z > 1.96)


def test_analysis_is_deterministic_and_seeded():
    assert ev.analyse(ev.Settings(seed=3)).rows[4].estimate == ev.analyse(ev.Settings(seed=3)).rows[4].estimate != ev.analyse(ev.Settings(seed=4)).rows[4].estimate
    assert ev.analyse(ev.Settings(n=4000)).rows[4].se < ev.analyse(ev.Settings(n=200)).rows[4].se


def test_analysis_without_any_valid_strategy():
    a = ev.analyse(ev.settings_from(K.PRESETS["Nichts brauchbar"]))
    assert a.valid == [] and a.best is None and a.sd_ratio == {} and a.holds == {"backdoor": False, "frontdoor": False, "instrument": False} and a.first_stage_f < 10
    assert math.isnan(a.rows[5].limit_bias) and math.isnan(a.rows[5].sd1)


def test_admissible_follows_the_order_of_the_assumptions():
    assert ev.admissible(()) == [] and ev.admissible(("backdoor",)) == ["Backdoor {Z}", "Backdoor {Z, W}"] and ev.admissible(("instrument", "backdoor")) == ["Backdoor {Z}", "Backdoor {Z, W}", "2SLS"]
    assert ev.admissible(("frontdoor",)) == ["Frontdoor"] and "naiv" not in ev.admissible(("backdoor", "frontdoor", "instrument")) and "alles anpassen" not in ev.admissible(("backdoor", "frontdoor", "instrument"))


def test_recommendation_names_the_smallest_streuung_and_the_false_assumptions():
    a = ev.analyse(DEFAULT)
    r = ev.recommend(a, ("backdoor",))
    assert r == {"admissible": ["Backdoor {Z}", "Backdoor {Z, W}"], "pick": "Backdoor {Z, W}", "false": ["backdoor"]}
    r = ev.recommend(a, ("frontdoor", "instrument"))
    assert r["pick"] == "Frontdoor" and r["false"] == [] and ev.recommend(a, ())["pick"] is None
    none = ev.recommend(ev.analyse(ev.settings_from(K.PRESETS["Kein Anstoß"])), ("instrument",))
    assert none["pick"] is None and none["false"] == ["instrument"]


def test_limits_experiment_structure(limits):
    assert list(limits) == list(K.PRESETS)
    for name, e in limits.items():
        assert set(e) == {"truth", "bias", "sd1", "valid"} and set(e["bias"]) == set(K.STRATEGIES) and e["truth"] == pytest.approx(0.6)
        assert all(e["valid"][m] == (not math.isnan(e["bias"][m]) and abs(e["bias"][m]) < 1e-9) for m in K.STRATEGIES)
    assert not any(limits["Nichts brauchbar"]["valid"].values()) and math.isnan(limits["Kein Anstoß"]["bias"]["2SLS"])


def test_rmse_curves():
    res = ev.rmse_curves(K.knobs())
    assert res["ns"] == K.EXP_CURVE_NS and set(res["curves"]) == set(K.STRATEGIES) and all(len(v) == len(res["ns"]) for v in res["curves"].values())
    fd = res["curves"]["Frontdoor"]
    assert fd == sorted(fd, reverse=True) and fd[-1] == pytest.approx(res["limits"]["Frontdoor"][1] / math.sqrt(50000), rel=1e-6) and res["limits"]["Frontdoor"][1] == pytest.approx(1.388, abs=0.002)
    assert res["curves"]["Backdoor {Z, W}"][-1] == pytest.approx(0.2236, abs=0.003) and res["limits"]["Frontdoor"][0] == pytest.approx(0, abs=1e-9)
    assert all(math.isnan(x) for x in ev.rmse_curves(K.knobs(pi=0))["curves"]["2SLS"])


def test_crossing_table():
    rows = ev.crossing_table(K.knobs())
    assert [r["s"] for r in rows] == list(K.EXP_CROSS_S)
    r3 = next(r for r in rows if r["s"] == 0.3)
    assert r3["n_iv"] == pytest.approx(1047, abs=1) and r3["bias"] == pytest.approx(0.0672, abs=0.0001) and r3["sd_bd"] == pytest.approx(1.038, abs=0.001) and r3["sd_iv"] == pytest.approx(2.408, abs=0.001)
    assert [r["n_iv"] for r in rows] == sorted((r["n_iv"] for r in rows), reverse=True) and rows[0]["n_iv"] == pytest.approx(69739, abs=2) and rows[-1]["n_iv"] == pytest.approx(117, abs=1)
    assert all(r["n_fd"] < r["n_iv"] for r in rows) and not any(math.isfinite(r["n_iv"]) for r in ev.crossing_table(K.knobs(pi=0)))


def test_mc_experiment_structure(small_mc):
    assert small_mc["n"] == 300 and small_mc["reps"] == 6 and list(small_mc["scenarios"]) == list(K.PRESETS)
    for e in small_mc["scenarios"].values():
        assert set(e) == {"mc", "valid", "truth"} and set(e["mc"]) == set(K.STRATEGIES)


def test_winners_by_measure_by_hand():
    def cell(bias, rmse, med, cover, width):
        return {"bias": bias, "sd": 0.0, "rmse": rmse, "median_error": med, "cover": cover, "width": width}

    entry = {"valid": {m: m in ("Frontdoor", "2SLS") for m in K.STRATEGIES}, "truth": 0.6,
             "mc": {"naiv": cell(0.3, 0.3, 0.3, 0.0, 0.1), "Backdoor {Z}": cell(-0.2, 0.2, 0.2, 0.5, 0.2), "Backdoor {Z, W}": cell(0.2, 0.25, 0.25, 0.93, 0.3), "alles anpassen": cell(0.5, 0.5, 0.5, 0.0, 0.2),
                    "Frontdoor": cell(0.01, 0.05, 0.04, 0.92, 0.4), "2SLS": cell(-0.02, 0.07, 0.05, 0.96, 1.0)}}
    assert ev.winners(entry, "Bias (Betrag)") == ["Frontdoor"] and ev.winners(entry, "RMSE") == ["Frontdoor"] and ev.winners(entry, "Median-Fehler") == ["Frontdoor"] and ev.winners(entry, "Breite") == ["naiv"]
    assert ev.winners(entry, "Deckung") == ["Backdoor {Z, W}", "2SLS"]                                                             # 93 % zählt, 92 % nicht
    assert ev.winners(entry, "Deckung", True) == ["2SLS"] and ev.winners(entry, "Breite", True) == ["Frontdoor"] and ev.winners(entry, "Bias (Betrag)", True) == ["Frontdoor"]
    none = {**entry, "valid": {m: False for m in K.STRATEGIES}}
    assert ev.winners(none, "RMSE", True) == [] and ev.winners(none, "RMSE") == ["Frontdoor"]


def test_vortest_experiment_structure(small_vortest):
    res = small_vortest
    assert res["ns"] == (200,) and res["reps"] == 20 and set(res["cells"]) == {(nm, 200) for nm in K.VORTEST_SCENARIOS}
    for c in res["cells"].values():
        assert set(c) == {"reject", "f", "rmse", "cover", "rmse_bd", "median_error_iv"} and 0 <= c["reject"] <= 1 and 0 <= c["cover"] <= 1 and c["f"] > 0


def test_vortest_experiment_is_deterministic():
    assert ev.vortest_experiment(ns=(200,), reps=5) == ev.vortest_experiment(ns=(200,), reps=5)


def test_regret_by_hand():
    res = ev.regret_experiment(("Alles beobachtet",), 1000)
    rm_bd, rm_iv = 1.0431 / math.sqrt(1000), 2.3323 / math.sqrt(1000)
    t = res["table"]["Alles beobachtet"]
    assert t["Backdoor {Z, W}"] == pytest.approx(0, abs=1e-12) and t["2SLS"] == pytest.approx(rm_iv - rm_bd, abs=2e-4) and t["Frontdoor"] == pytest.approx(1.3369 / math.sqrt(1000) - rm_bd, abs=2e-4)
    assert t["naiv"] == pytest.approx(math.sqrt(0.254 ** 2 + 1.018 ** 2 / 1000) - rm_bd, abs=1e-3) and t["alles anpassen"] == pytest.approx(math.sqrt(0.36 + 1.104 ** 2 / 1000) - rm_bd, abs=1e-3)
    assert res["pick"] == "Backdoor {Z, W}" and res["skipped"] == [] and res["rows"]["Backdoor {Z, W}"]["max"] == pytest.approx(0, abs=1e-12)


def test_regret_skips_scenarios_without_a_valid_strategy_and_counts_infinite_regret():
    res = ev.regret_experiment(("Nichts brauchbar", "Kein Anstoß", "Alles beobachtet"), 1000)
    assert res["skipped"] == ["Nichts brauchbar"] and list(res["table"]) == ["Kein Anstoß", "Alles beobachtet"] and math.isinf(res["table"]["Kein Anstoß"]["2SLS"])
    assert math.isinf(res["rows"]["2SLS"]["max"]) and res["rows"]["2SLS"]["count"] == 2 and res["rows"]["Frontdoor"]["below"] == 2 and res["pick"] == "Frontdoor"
    empty = ev.regret_experiment(("Nichts brauchbar",), 1000)
    assert empty["pick"] is None and empty["table"] == {} and math.isnan(empty["rows"]["naiv"]["max"])
    assert ev.regret_experiment((), 1000)["pick"] is None


def test_regret_depends_on_the_chosen_scenarios():
    all9 = ev.regret_experiment(tuple(K.PRESETS), 1000)
    only_hidden = ev.regret_experiment(("Verborgene Ursache, Anstoß da (Voreinstellung)", "Teilweise vermittelt"), 1000)
    assert all9["pick"] == "Frontdoor" and only_hidden["pick"] == "2SLS"


def test_crossing_table_ignores_the_violations_but_not_the_instrument():
    clean = ev.crossing_table(K.knobs())
    assert ev.crossing_table(K.knobs(e=0.3, f=0.4, g=0.3, h=0.3)) == clean
    assert ev.crossing_table(K.knobs(pi=0.25))[2]["n_iv"] > clean[2]["n_iv"] and ev.crossing_table(K.knobs(c=0.5))[2]["sd_fd"] != clean[2]["sd_fd"]
