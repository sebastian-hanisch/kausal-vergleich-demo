"""Jede Zahl aus README.md und den App-Texten wird hier über dieselben Auswertungsfunktionen nachgerechnet (DEMO-PLAYBOOK §4).

Die Experimente 1, 2 und 5 sind exakt (Bevölkerungskovarianz, Asymptotik): Dort gelten Toleranzen der Rundung. Experimente 3 und 4 laufen über SplitMix64-Ströme (ganzzahlig, plattformstabil), die Schätzungen rechnen aber in Gleitkomma;
wo ein einzelner Datensatz die Deckung kippen kann (Anteile bei 300 oder 400 Läufen) gelten Toleranzen von zwei Läufen. Was Identität oder Konstruktion ist, steht ausdrücklich dabei: „Kriterium am Diagramm erfüllt ⇔ Grenzwert-Bias null“
(Identität der Welt, kein Befund), die Kreuzungsgröße n* (Rechnung aus Bias und Streuung, keine Beobachtung), die Reue (ein Werkzeug über die gewählten Szenarien, kein Befund über Verfahren: Frontdoor ist in sechs der neun Szenarien nach Konstruktion gültig).
"""

import math

import pytest

import kvl_constants as K
import kvl_evaluation as ev
import kvl_strategies as S

DEFAULT = K.PRESETS[K.DEFAULT_SCENARIO]["knobs"]
ALL = K.PRESETS["Alles beobachtet"]["knobs"]


def test_claim_hand_instances():
    assert S.limits(ALL)["naiv"][0] == pytest.approx(0.6 * 0.8 / 1.89, abs=1e-9) and S.limits(ALL)["naiv"][0] == pytest.approx(0.254, abs=0.0006)
    assert S.limits(ALL)["alles anpassen"][0] == pytest.approx(-0.600, abs=1e-9)
    assert S.limits(DEFAULT)["Backdoor {Z, W}"][0] == pytest.approx(0.36 / 1.61, abs=1e-9) and S.limits(DEFAULT)["Backdoor {Z, W}"][0] == pytest.approx(0.2236, abs=0.00006)
    assert S.limits(K.PRESETS["Nichts brauchbar"]["knobs"])["Frontdoor"][0] == pytest.approx(-0.300, abs=1e-9)
    assert ev.crossing_table(K.knobs())[2]["n_iv"] == pytest.approx(1047, abs=1)


def test_claim_the_criterion_at_the_diagram_equals_the_exact_bias_is_an_identity_of_this_world(limits):
    for name, e in limits.items():
        for m in K.STRATEGIES:
            assert e["valid"][m] == (not math.isnan(e["bias"][m]) and abs(e["bias"][m]) < 1e-9), (name, m)


def test_claim_who_is_valid_where(limits):
    count = {m: sum(e["valid"][m] for e in limits.values()) for m in K.STRATEGIES}
    assert count == {"naiv": 0, "Backdoor {Z}": 1, "Backdoor {Z, W}": 1, "alles anpassen": 0, "Frontdoor": 6, "2SLS": 5}
    assert [n for n, e in limits.items() if not any(e["valid"].values())] == ["Nichts brauchbar"]
    assert [n for n, e in limits.items() if e["valid"]["Backdoor {Z, W}"]] == ["Alles beobachtet"]
    assert [n for n, e in limits.items() if not e["valid"]["Frontdoor"]] == ["Nichts brauchbar", "Teilweise vermittelt", "Ursache wirkt auf den Mediator"]


def test_claim_exact_limits_table(limits):
    d, o = limits[K.DEFAULT_SCENARIO]["bias"], limits["Alles beobachtet"]["bias"]
    assert [round(d[m], 3) for m in K.STRATEGIES] == [0.373, 0.224, 0.224, -0.454, 0.0, 0.0] and [round(o[m], 3) for m in K.STRATEGIES] == [0.254, 0.0, 0.0, -0.6, 0.0, 0.0]
    assert limits["Anstoß wirkt direkt"]["bias"]["2SLS"] == pytest.approx(0.6, abs=1e-9) and limits["Anstoß hängt an der Ursache"]["bias"]["2SLS"] == pytest.approx(0.248, abs=0.0006)
    assert limits["Anstoß hängt an der Ursache"]["bias"]["2SLS"] == pytest.approx(limits["Anstoß hängt an der Ursache"]["bias"]["Backdoor {Z, W}"], abs=0.0006)
    assert limits["Teilweise vermittelt"]["bias"]["Frontdoor"] == pytest.approx(-0.3, abs=1e-9) and limits["Ursache wirkt auf den Mediator"]["bias"]["Frontdoor"] == pytest.approx(0.211, abs=0.0006)
    assert math.isnan(limits["Kein Anstoß"]["bias"]["2SLS"]) and math.isnan(limits["Nichts brauchbar"]["bias"]["2SLS"])


def test_claim_price_of_validity_is_exact():
    sd = {m: S.limits(ALL)[m][1] for m in K.STRATEGIES}
    assert sd["Backdoor {Z, W}"] == pytest.approx(1.043, abs=0.0006) and sd["Backdoor {Z}"] == pytest.approx(1.173, abs=0.0006) and sd["Frontdoor"] == pytest.approx(1.337, abs=0.0006) and sd["2SLS"] == pytest.approx(2.332, abs=0.0006)
    assert sd["2SLS"] / sd["Backdoor {Z, W}"] == pytest.approx(2.2, abs=0.05) and sd["Frontdoor"] / sd["Backdoor {Z, W}"] == pytest.approx(1.3, abs=0.02)
    assert sd["Backdoor {Z, W}"] < sd["Backdoor {Z}"] < sd["Frontdoor"] < sd["2SLS"]


def test_claim_the_narrowest_interval_never_belongs_to_a_valid_strategy(limits):
    for name, e in limits.items():
        if not any(e["valid"].values()):
            continue
        sds = {m: e["sd1"][m] for m in K.STRATEGIES if math.isfinite(e["sd1"][m])}
        assert not e["valid"][min(sds, key=sds.get)], name


def test_claim_crossing_sizes_are_a_calculation():
    rows = {r["s"]: r for r in ev.crossing_table(K.knobs())}
    assert [round(rows[s]["n_iv"]) for s in K.EXP_CROSS_S] == pytest.approx([69739, 4697, 1047, 389, 117], abs=2)
    assert [round(rows[s]["bias"], 4) for s in K.EXP_CROSS_S] == [0.0079, 0.031, 0.0672, 0.1135, 0.2236] and rows[0.3]["sd_bd"] == pytest.approx(1.038, abs=0.001) and rows[0.3]["sd_iv"] == pytest.approx(2.408, abs=0.001)
    assert rows[0.3]["n_fd"] < rows[0.3]["n_iv"] and rows[0.3]["n_fd"] == pytest.approx(S.crossing_n(rows[0.3]["bias"], rows[0.3]["sd_bd"], rows[0.3]["sd_fd"]))


def test_claim_the_crossing_agrees_with_the_simulation():
    """Unterhalb von n* schlägt das verzerrte Backdoor das 2SLS nach RMSE, oberhalb nicht (a = b = 0,3, n* = 1 047)."""
    k = K.knobs(a=0.3, b=0.3)
    small = S.monte_carlo(k, 300, 300, 4001)
    large = S.monte_carlo(k, 5000, 300, 4002)
    assert small["Backdoor {Z, W}"]["rmse"] < small["2SLS"]["rmse"] and large["Backdoor {Z, W}"]["rmse"] > large["2SLS"]["rmse"]
    assert small["Backdoor {Z, W}"]["cover"] < 0.95 and large["Backdoor {Z, W}"]["cover"] < 0.1


def test_claim_among_the_valid_rmse_follows_the_assumptions_and_bias_is_noise(full_mc):
    e = full_mc["scenarios"]["Alles beobachtet"]
    mc = e["mc"]
    assert all(abs(mc[m]["bias"]) < 0.015 for m in ("Backdoor {Z}", "Backdoor {Z, W}", "Frontdoor", "2SLS")) and all(e["valid"][m] for m in ("Backdoor {Z}", "Backdoor {Z, W}", "Frontdoor", "2SLS"))
    assert mc["Backdoor {Z, W}"]["rmse"] < mc["Frontdoor"]["rmse"] < mc["2SLS"]["rmse"] and mc["2SLS"]["rmse"] / mc["Backdoor {Z, W}"]["rmse"] == pytest.approx(2.2, abs=0.35)
    assert ev.winners(e, "RMSE", True) == ["Backdoor {Z, W}"] and ev.winners(e, "Median-Fehler", True) == ["Backdoor {Z, W}"]
    assert mc["naiv"]["bias"] == pytest.approx(0.254, abs=0.015) and mc["alles anpassen"]["bias"] == pytest.approx(-0.6, abs=0.015) and mc["naiv"]["cover"] == 0 and mc["alles anpassen"]["cover"] == 0


def test_claim_invalid_strategies_miss_with_narrow_intervals_and_the_empty_instrument_covers_with_a_huge_one(full_mc):
    for name, e in full_mc["scenarios"].items():
        for m in K.STRATEGIES:
            x = e["mc"][m]
            if e["valid"][m]:
                assert x["cover"] >= 0.88, (name, m)
            elif m == "2SLS" and name in ("Kein Anstoß", "Nichts brauchbar"):
                assert x["cover"] >= 0.95 and x["width"] > 5, (name, m)
            else:
                assert x["cover"] <= 0.05 and x["width"] < 0.5, (name, m)


def test_claim_nothing_is_valid_in_one_scenario_and_five_strategies_miss_confidently(full_mc):
    e = full_mc["scenarios"]["Nichts brauchbar"]
    assert not any(e["valid"].values()) and [m for m in K.STRATEGIES if e["mc"][m]["cover"] <= 0.05] == ["naiv", "Backdoor {Z}", "Backdoor {Z, W}", "alles anpassen", "Frontdoor"]
    assert {m: round(e["mc"][m]["bias"], 2) for m in K.STRATEGIES if m != "2SLS"} == {"naiv": 0.42, "Backdoor {Z}": 0.26, "Backdoor {Z, W}": 0.26, "alles anpassen": -0.29, "Frontdoor": -0.3}


def test_claim_coverage_without_width_is_worth_nothing(full_mc):
    for name in ("Kein Anstoß", "Nichts brauchbar"):
        e = full_mc["scenarios"][name]
        assert "2SLS" in ev.winners(e, "Deckung") and e["mc"]["2SLS"]["width"] > 5 and e["mc"]["2SLS"]["width"] > 30 * e["mc"]["Frontdoor"]["width"]


def test_claim_weak_instrument_needs_median_and_width():
    weak = S.monte_carlo(K.knobs(pi=0.05), 1000, 300, 4003)["2SLS"]
    strong = S.monte_carlo(K.knobs(pi=0.5), 1000, 300, 4004)["2SLS"]
    assert weak["rmse"] > 1 and weak["q75"] - weak["q25"] > 0.8 and abs(weak["median"] - 0.6) < 0.25 and weak["cover"] >= 0.95 and weak["width"] > 2
    assert strong["rmse"] < 0.1 and strong["q75"] - strong["q25"] < 0.16 and strong["width"] < 0.4 and weak["median_error"] < weak["rmse"] / 5


def test_claim_the_winner_depends_on_the_measure(full_mc):
    d = full_mc["scenarios"][K.DEFAULT_SCENARIO]
    assert ev.winners(d, "RMSE") == ["Frontdoor"] and ev.winners(d, "Breite") == ["naiv"] and set(ev.winners(d, "Deckung")) == {"Frontdoor", "2SLS"}
    narrowest_invalid = [n for n, e in full_mc["scenarios"].items() if any(e["valid"].values()) and not e["valid"][ev.winners(e, "Breite")[0]]]
    assert len(narrowest_invalid) == 8
    assert ev.winners(full_mc["scenarios"]["Teilweise vermittelt"], "RMSE") == ["2SLS"] and ev.winners(full_mc["scenarios"]["Ursache wirkt auf den Mediator"], "RMSE") == ["2SLS"]
    assert ev.winners(full_mc["scenarios"]["Anstoß wirkt direkt"], "RMSE") == ["Frontdoor"] and ev.winners(full_mc["scenarios"]["Nichts brauchbar"], "RMSE", True) == []


def test_claim_vortest_shows_disagreement_not_validity(full_vortest):
    c = full_vortest["cells"]
    ok, hid, weak, direct, coupled, med = (c[(nm, 1000)] for nm in K.VORTEST_SCENARIOS)
    assert 0.02 <= ok["reject"] <= 0.09 and 0.80 <= hid["reject"] <= 0.93 and weak["reject"] <= 0.06 and weak["f"] == pytest.approx(8.3, abs=1.0)
    assert direct["reject"] >= 0.95 and direct["cover"] <= 0.05 and direct["rmse"] >= 0.5 and direct["rmse"] > direct["rmse_bd"]
    assert coupled["reject"] <= 0.09 and coupled["rmse"] == pytest.approx(0.25, abs=0.03) and coupled["cover"] <= 0.05
    assert [round(x["rmse_bd"], 2) for x in (ok, hid, direct, coupled, med)] == pytest.approx([0.03, 0.23, 0.32, 0.25, 0.32], abs=0.02) and direct["median_error_iv"] == pytest.approx(0.60, abs=0.03) and coupled["median_error_iv"] == pytest.approx(0.25, abs=0.03)
    assert med["reject"] >= 0.93 and med["cover"] >= 0.88 and med["rmse"] < med["rmse_bd"] / 2 and hid["rmse"] < hid["rmse_bd"] / 1.5 and hid["cover"] >= 0.75


def test_claim_vortest_small_sample(full_vortest):
    c = full_vortest["cells"]
    ok200, hid200 = c[(K.VORTEST_SCENARIOS[0], 200)], c[(K.VORTEST_SCENARIOS[1], 200)]
    assert 0.1 <= hid200["reject"] <= 0.35 and ok200["rmse"] > ok200["rmse_bd"] * 1.3 and hid200["rmse"] == pytest.approx(hid200["rmse_bd"], rel=0.15)


def test_claim_regret_is_a_tool_over_the_chosen_scenarios():
    res = ev.regret_experiment(tuple(K.PRESETS), 1000)
    r = res["rows"]
    assert res["skipped"] == ["Nichts brauchbar"] and res["pick"] == "Frontdoor" and r["Frontdoor"]["max"] == pytest.approx(0.218, abs=0.005) and r["Frontdoor"]["mean"] == pytest.approx(0.044, abs=0.004) and r["Frontdoor"]["below"] == 6
    assert math.isinf(r["2SLS"]["max"]) and r["alles anpassen"]["max"] == pytest.approx(0.568, abs=0.005) and r["naiv"]["max"] == pytest.approx(0.398, abs=0.005) and r["Backdoor {Z, W}"]["max"] == pytest.approx(0.275, abs=0.005)
    assert ev.regret_experiment(("Verborgene Ursache, Anstoß da (Voreinstellung)", "Teilweise vermittelt"), 1000)["pick"] == "2SLS"
