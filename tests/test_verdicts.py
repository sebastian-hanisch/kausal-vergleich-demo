"""Verdikte: Art und Text je Lage; jede Zahl im Text stammt aus dem Befund, nicht aus einer Konstanten."""

import dataclasses

import pytest

import kvl_constants as K
import kvl_evaluation as ev


def _a(name=K.DEFAULT_SCENARIO):
    return ev.analyse(ev.settings_from(K.PRESETS[name]))


def test_analysis_verdict_default_names_the_valid_strategies_and_the_confident_wrong_ones():
    kind, text = ev.verdict_analysis(_a())
    assert kind == "success" and text.startswith("Wahrheit 0,60; Schätzungen aus n = 1 000: naiv 1,02, Backdoor {Z} 0,86") and "Gültig: Frontdoor, 2SLS; die kleinste Streuung hat Frontdoor (2SLS 1,9-fach)." in text
    assert text.endswith("Ungültig und trotzdem mit engem Intervall daneben: naiv, Backdoor {Z}, Backdoor {Z, W}, alles anpassen.")


def test_analysis_verdict_without_a_valid_strategy_is_an_error_that_points_to_sensitivity():
    kind, text = ev.verdict_analysis(_a("Nichts brauchbar"))
    assert kind == "error" and "Keine der sechs Strategien ist gültig: 5 verfehlen die Wahrheit mit einem Intervall, das sie ausschließt (Breiten 0,10 bis 0,14)" in text and text.endswith("Hier hilft keine Wahl der Strategie, nur die Sensitivität.")


def test_analysis_verdict_all_valid_lists_the_ratios():
    kind, text = ev.verdict_analysis(_a("Alles beobachtet"))
    assert kind == "success" and "Gültig: Backdoor {Z}, Backdoor {Z, W}, Frontdoor, 2SLS; die kleinste Streuung hat Backdoor {Z, W} (Backdoor {Z} 1,1-fach, Frontdoor 1,3-fach, 2SLS 2,2-fach)." in text
    assert text.endswith("Ungültig und trotzdem mit engem Intervall daneben: naiv, alles anpassen.")


def test_analysis_verdict_weak_instrument_is_a_warning():
    kind, text = ev.verdict_analysis(_a("Schwacher Anstoß"))
    assert kind == "warning" and "(2SLS 9,5-fach)" in text and text.endswith("Der Anstoß ist schwach (F = 6,0): 2SLS ist gültig, aber seine Schätzung streut stark und der Mittelwert ist unbrauchbar.")
    strong = ev.analyse(dataclasses.replace(ev.settings_from(K.PRESETS["Schwacher Anstoß"]), seed=35))
    assert strong.first_stage_f > 10 and ev.verdict_analysis(strong)[0] == "success"


def test_analysis_verdict_a_valid_strategy_that_misses_the_truth_is_not_listed_as_wrong():
    a = _a("Alles beobachtet")
    a.rows[1].covers = False                                                    # Backdoor {Z} gültig, aber diesmal daneben
    assert "Backdoor {Z}" not in ev.verdict_analysis(a)[1].split("Ungültig und trotzdem")[1]


def test_recommendation_verdicts():
    a = ev.analyse(ev.Settings())
    kind, text = ev.verdict_recommend(a, ())
    assert kind == "warning" and text.startswith("Keine Annahme angekreuzt: Dann ist keine der sechs Strategien zulässig.") and "Sensitivität (Stück 11)" in text
    kind, text = ev.verdict_recommend(a, ("backdoor",))
    assert kind == "error" and text.startswith("Zulässig wären Backdoor {Z}, Backdoor {Z, W}; die kleinste Streuung hat Backdoor {Z, W}.") and "Annahme „backdoor“ aber **nicht**" in text and "+0,22 daneben" in text and text.endswith("in der Wirklichkeit nie.")
    kind, text = ev.verdict_recommend(a, ("frontdoor", "instrument"))
    assert kind == "success" and text == "Zulässig: Frontdoor, 2SLS; die kleinste Streuung hat Frontdoor. Die angekreuzten Annahmen gelten in dieser Welt."
    assert "backdoor, instrument" in ev.verdict_recommend(ev.analyse(ev.settings_from(K.PRESETS["Kein Anstoß"])), ("backdoor", "instrument"))[1]


def test_limits_verdict(limits):
    kind, text = ev.verdict_limits(limits)
    assert kind == "info" and text.startswith("In 9 Szenarien ist Frontdoor am häufigsten gültig (6), 2SLS in 5, Backdoor {Z, W} in 1; naiv und „alles anpassen“ in keinem (0 und 0).") and "In 1 Szenario (Nichts brauchbar) ist keine der sechs gültig." in text


def test_crossing_verdict():
    kind, text = ev.verdict_crossing(ev.crossing_table(K.knobs()))
    assert kind == "info" and "Stärke 0,3 (a = b) hat das Backdoor den Bias 0,067 und streut 1,04 je √n, das 2SLS 2,41" in text and "Erst ab n = 1 047 schlägt das gültige 2SLS" in text and "Bei s = 0,1 liegt die Kreuzung bei 69 739, bei s = 0,6 bei 117." in text
    kind, text = ev.verdict_crossing(ev.crossing_table(K.knobs(pi=0)))
    assert kind == "info" and "keine Kreuzung mit ihm (kein Anstoß oder kein Bias)" in text and "gegen Frontdoor liegt sie bei n = " in text


def test_mc_verdict(small_mc):
    kind, text = ev.verdict_mc(small_mc)
    assert kind == "info" and text.startswith("Bei „Alles beobachtet“ sind vier Strategien gültig: nach RMSE gewinnt ") and "(n = 300)" in text and "6 feste Läufe je Zelle." in text and "nach Deckung (≥ 93 %)" in text


def test_vortest_verdict(small_vortest):
    kind, text = ev.verdict_vortest(small_vortest)
    assert kind == "info" and text.startswith("Bei n = 200: ohne verborgene Ursache verwirft der Vortest in ") and "Er zeigt Uneinigkeit, nicht Gültigkeit." in text and "hängt der Anstoß an der Ursache, verwirft er nur in" in text


def test_regret_verdict():
    kind, text = ev.verdict_regret(ev.regret_experiment(tuple(K.PRESETS), 1000))
    assert kind == "info" and text.startswith("Über die 8 gewählten Szenarien mit gültiger Strategie (n = 1 000) hat Frontdoor die kleinste größte Reue (0,22); unter 6 von 8 Szenarien liegt sie unter 0,05.") and text.endswith("übersprungen: Nichts brauchbar.")
    kind, text = ev.verdict_regret(ev.regret_experiment(("Nichts brauchbar",), 1000))
    assert kind == "warning" and text == "Keines der gewählten Szenarien hat eine gültige Strategie: Reue ist nicht definiert."
    assert "übersprungen" not in ev.verdict_regret(ev.regret_experiment(("Alles beobachtet",), 1000))[1]


def test_vortest_verdict_names_the_largest_sample_size():
    res = ev.vortest_experiment(ns=(150, 250), reps=6)
    assert ev.verdict_vortest(res)[1].startswith("Bei n = 250: ohne verborgene Ursache verwirft der Vortest in ")
