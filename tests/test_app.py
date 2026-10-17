"""End-to-end Smoke-Test via Streamlits AppTest: lädt app.py, klickt jeden Preset-Button, fährt die Regler an die Grenzen und prüft, dass kein Python-Fehler auftritt."""

import os

from streamlit.testing.v1 import AppTest

import kvl_constants as K
import kvl_presets as P

APP_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "app.py")
TIMEOUT = 300
SLIDERS = tuple(f"{k}_slider" for k in K.KNOBS) + ("n_slider",)


def _fresh_app() -> AppTest:
    at = AppTest.from_file(APP_PATH, default_timeout=TIMEOUT)
    at.run(timeout=TIMEOUT)
    assert not at.exception, [str(e) for e in at.exception]
    return at


def _metrics(at):
    return {m.label: m.value for m in at.metric}


def test_app_loads_without_exception_and_shows_the_numbers():
    at = _fresh_app()
    m = _metrics(at)
    assert m["Wahrheit (do)"] == "0,60" and m["Gültige Strategien"] == "2 von 6" and m["Kleinste Streuung unter den gültigen"] == "Frontdoor" and float(m["Erste Stufe des Anstoßes (F)"].replace(",", ".")) > 100
    assert all(at.slider(key=f"{k}_slider").value == v[0] for k, v in K.KNOBS.items()) and at.slider(key="n_slider").value == K.DEFAULT_N and at.number_input(key="seed_input").value == K.DEFAULT_SEED
    assert any("Gültig: Frontdoor, 2SLS" in s.value for s in at.success) and any(w.value.startswith("Keine Annahme angekreuzt") for w in at.warning)


def test_every_button_click_does_not_raise():
    labels = [b.label for b in _fresh_app().button]
    assert labels, "keine Buttons gefunden"
    for index, label in enumerate(labels):
        if label.startswith("▶"):
            continue                                                         # die Experimente laufen in eigenen Tests
        at = _fresh_app()
        at.button[index].click().run(timeout=TIMEOUT)
        assert not at.exception, (label, [str(e) for e in at.exception])


def test_the_sliders_at_min_and_max_do_not_raise():
    for key in SLIDERS:
        for edge in ("min", "max"):
            at = _fresh_app()
            s = at.slider(key=key)
            s.set_value(s.min if edge == "min" else s.max).run(timeout=TIMEOUT)
            assert not at.exception, (key, edge, [str(e) for e in at.exception])


def test_every_preset_is_applied_to_the_widgets():
    labels = {b.label: i for i, b in enumerate(_fresh_app().button)}
    for name, preset in K.PRESETS.items():
        at = _fresh_app()
        at.button[labels[name]].click().run(timeout=TIMEOUT)
        assert not at.exception, (name, [str(e) for e in at.exception])
        assert all(at.slider(key=f"{k}_slider").value == preset["knobs"][k] for k in K.KNOBS) and at.slider(key="n_slider").value == preset["n"] and at.number_input(key="seed_input").value == preset["seed"], name


def test_no_instrument_preset_shows_the_numbers():
    at = _fresh_app()
    at.button[[b.label for b in at.button].index("Kein Anstoß")].click().run(timeout=TIMEOUT)
    m = _metrics(at)
    assert not at.exception and m["Gültige Strategien"] == "1 von 6" and m["Kleinste Streuung unter den gültigen"] == "Frontdoor" and float(m["Erste Stufe des Anstoßes (F)"].replace(",", ".")) < 5


def test_nothing_usable_preset_is_an_error_and_has_no_best_strategy():
    at = _fresh_app()
    at.button[[b.label for b in at.button].index("Nichts brauchbar")].click().run(timeout=TIMEOUT)
    m = _metrics(at)
    assert not at.exception and m["Gültige Strategien"] == "0 von 6" and m["Kleinste Streuung unter den gültigen"] == "—" and any("Keine der sechs Strategien ist gültig" in e.value for e in at.error)


def test_the_checklist_changes_the_recommendation():
    at = _fresh_app()
    at.checkbox(key="assume_backdoor").set_value(True).run(timeout=TIMEOUT)
    assert not at.exception and any("aber **nicht**" in e.value for e in at.error)
    at.checkbox(key="assume_backdoor").set_value(False).run(timeout=TIMEOUT)
    at.checkbox(key="assume_frontdoor").set_value(True).run(timeout=TIMEOUT)
    assert not at.exception and any(s.value.startswith("Zulässig: Frontdoor;") for s in at.success)


def test_the_plausible_scenarios_change_the_regret():
    at = _fresh_app()
    assert any("hat Frontdoor die kleinste größte Reue" in i.value for i in at.info)
    at.multiselect(key="plausible_select").set_value(["Verborgene Ursache, Anstoß da (Voreinstellung)", "Teilweise vermittelt"]).run(timeout=TIMEOUT)
    assert not at.exception and any("hat 2SLS die kleinste größte Reue" in i.value for i in at.info)
    at.multiselect(key="plausible_select").set_value([]).run(timeout=TIMEOUT)
    assert not at.exception and any("Keines der gewählten Szenarien" in w.value for w in at.warning)


def test_every_experiment_runs_on_click():
    at = _fresh_app()
    assert "exp3_ready" not in at.session_state
    assert any("Gültigkeit am Diagramm und exakter Grenzwert stimmen" in i.value for i in at.info) and any("Erst ab n = 1 047" in i.value for i in at.info)          # Experimente 1, 2 und 5 sind exakt und stehen gleich da
    for key, text in (("exp3_run", "Bei „Alles beobachtet“ sind vier Strategien gültig"), ("exp4_run", "Er zeigt Uneinigkeit, nicht Gültigkeit")):
        at.button(key=key).click().run(timeout=TIMEOUT)
        assert not at.exception, (key, [str(e) for e in at.exception])
        assert any(text in i.value for i in at.info), key
    at.radio(key="measure_radio").set_value("Deckung").run(timeout=TIMEOUT)
    at.selectbox(key="scenario_select").set_value("Nichts brauchbar").run(timeout=TIMEOUT)
    assert not at.exception, [str(e) for e in at.exception]


def test_pdf_is_created_only_after_the_click():
    at = _fresh_app()
    assert "pdf_made" not in at.session_state
    at.button(key="pdf_make").click().run(timeout=TIMEOUT)
    assert not at.exception, [str(e) for e in at.exception]
    settings, data = at.session_state["pdf_made"]
    assert data[:5] == b"%PDF-" and len(data) > 1000


def test_permalink_values_are_taken_and_clamped():
    at = AppTest.from_file(APP_PATH, default_timeout=TIMEOUT)
    at.query_params["n"] = "50"
    at.query_params["a"] = "9"
    at.query_params["pi"] = "0.07"
    at.query_params["c"] = "0"
    at.query_params["seed"] = "abc"
    at.run()
    assert not at.exception, [str(e) for e in at.exception]
    assert at.slider(key="n_slider").value == K.N_MIN and at.slider(key="a_slider").value == 1.0 and abs(at.slider(key="pi_slider").value - 0.05) < 1e-9 and at.slider(key="c_slider").value == 0.2 and at.number_input(key="seed_input").value == K.DEFAULT_SEED


def test_the_address_bar_mirrors_the_settings():
    at = _fresh_app()
    assert at.query_params["n"] in ("1000", ["1000"]) and at.query_params["pi"] in ("0.5", ["0.5"])
    at.slider(key="g_slider").set_value(0.3).run(timeout=TIMEOUT)
    assert at.query_params["g"] in ("0.3", ["0.3"])
    at.button[[b.label for b in at.button].index("Alles beobachtet")].click().run(timeout=TIMEOUT)
    assert at.query_params["a"] in ("0.0", ["0.0"]) and at.query_params["g"] in ("0.0", ["0.0"])
