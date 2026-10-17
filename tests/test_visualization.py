"""Figuren und Tabellen: bauen ohne Fehler, und die gezeichneten Zahlen sind die der Auswertung."""

import re

import pytest

import kvl_constants as K
import kvl_evaluation as ev
import kvl_visualization as V
import kvl_world as W


def _a(name=K.DEFAULT_SCENARIO):
    return ev.analyse(ev.settings_from(K.PRESETS[name]))


def _arrows(fig):
    return [ann for ann in fig.layout.annotations if ann.showarrow]


def test_world_diagram_draws_every_arrow_and_colours_the_hidden_and_violating_ones():
    fig = V.build_world_diagram(K.knobs())
    assert len(_arrows(fig)) == len(W.edge_weights(K.knobs())) == 9
    colours = [ann.arrowcolor for ann in _arrows(fig)]
    assert colours.count(K.COLOR_TRUTH) == 2 and colours.count("#6b6b6b") == 7 and K.COLOR_BAD not in colours
    nodes = {t.text[0]: t for t in fig.data if t.mode == "markers+text"}
    assert set(nodes) == set(K.NAMES) and nodes["X"].marker.line.color == K.COLOR_GOOD and nodes["U"].marker.line.color == K.COLOR_TRUTH
    bad = V.build_world_diagram(K.knobs(e=0.3, f=0.4, g=0.3, h=0.3))
    assert len(_arrows(bad)) == 13 and [ann.arrowcolor for ann in _arrows(bad)].count(K.COLOR_BAD) == 4
    assert len(_arrows(V.build_world_diagram(K.knobs(a=0, b=0, pi=0)))) == 6


def test_estimate_chart_has_one_point_per_strategy_with_the_validity_colour():
    a = _a()
    fig = V.build_estimate_chart(a)
    d = fig.data[0]
    assert list(d.y) == list(reversed(K.STRATEGIES)) and list(d.marker.color) == [K.COLOR_GOOD, K.COLOR_GOOD, K.COLOR_BAD, K.COLOR_BAD, K.COLOR_BAD, K.COLOR_BAD]
    rows = list(reversed(a.rows))
    assert list(d.x) == pytest.approx([r.estimate for r in rows]) and list(d.error_x.array) == pytest.approx([r.hi - r.estimate for r in rows]) and list(d.error_x.arrayminus) == pytest.approx([r.estimate - r.lo for r in rows])
    assert fig.layout.shapes[0].x0 == pytest.approx(0.6) and "Wahrheit 0,60" in fig.layout.annotations[0].text


def test_strategy_table_and_summary():
    df = V.build_strategy_table(_a())
    assert list(df["Strategie"]) == list(K.STRATEGIES) and list(df["Annahme gilt"]) == ["nein", "nein", "nein", "nein", "ja", "ja"] and list(df["deckt die Wahrheit"])[:4] == ["nein"] * 4
    assert df["Grenzwert-Bias (exakt)"][2] == "+0,224" and df["Streuung je √n (exakt)"][4] == "1,39" and df["Grund, wenn die Annahme nicht gilt"][4] == "" and "X ← U → Y" in df["Grund, wenn die Annahme nicht gilt"][2]
    assert df["Schätzung"][4].count(",") == 1 and " bis " in df["95-%-Intervall"][0]
    assert V.style_strategy_table(df).to_html().count("rgba(192,57,43,0.15)") > 0
    nodata = V.build_strategy_table(_a("Kein Anstoß"))
    assert nodata["Grenzwert-Bias (exakt)"][5] == "—" and nodata["Streuung je √n (exakt)"][5] == "—"
    s = V.build_summary_table(_a())
    assert s["Wert"][0] == "0,600" and s["Wert"][1] == "Frontdoor, 2SLS" and s["Wert"][2] == "Frontdoor" and s["Wert"][5] == "1000, 35" and s["Wert"][4] == f"{ev.de(_a().hausman_z, 2)} ({'verworfen' if _a().hausman_rejected else 'nicht verworfen'})"
    assert V.build_summary_table(_a("Nichts brauchbar"))["Wert"][2] == "—"


def test_limits_table_and_chart(limits):
    df = V.build_limits_table(limits)
    assert len(df) == 9 and list(df.columns) == ["Szenario", "Wahrheit", *K.STRATEGIES] and df["Wahrheit"][0] == "0,60"
    assert df["Backdoor {Z, W}"][0] == "✗ +0,224" and df["Frontdoor"][0].startswith("✓ ") and df["2SLS"][2] == "✗ —" and df["alles anpassen"][1] == "✗ −0,600"
    html = V.style_limits_table(df).to_html().split("</style>")[0]
    assert len(re.findall(r"_row\d+_col\d+", html)) == 9 * 6 and "rgba(47,107,101,0.20)" in html and "rgba(192,57,43,0.15)" in html


def test_curves_chart_and_crossing_table():
    res = ev.rmse_curves(K.knobs())
    fig = V.build_curves_chart(res)
    assert [t.name for t in fig.data] == list(K.STRATEGIES) and fig.layout.xaxis.type == "log" and list(fig.data[0].x) == list(K.EXP_CURVE_NS)
    assert [t.name for t in V.build_curves_chart(ev.rmse_curves(K.knobs(pi=0))).data] == [m for m in K.STRATEGIES if m != "2SLS"]
    df = V.build_crossing_table(ev.crossing_table(K.knobs()))
    assert list(df["Stärke s (a = b)"]) == ["0,1", "0,2", "0,3", "0,4", "0,6"] and df["Kreuzung n* gegen 2SLS"][2] == "1 047" and df["Bias Backdoor {Z, W}"][2] == "+0,067" and df["2SLS"][2] == "2,41"
    assert set(V.build_crossing_table(ev.crossing_table(K.knobs(pi=0)))["Kreuzung n* gegen 2SLS"]) == {"—"} and set(V.build_crossing_table(ev.crossing_table(K.knobs(pi=0)))["2SLS"]) == {"—"}


def test_mc_figures(small_mc):
    name = "Alles beobachtet"
    for measure in K.MEASURES:
        fig = V.build_mc_chart(small_mc, name, measure)
        assert list(fig.data[0].x) == list(K.STRATEGIES) and list(fig.data[0].marker.color) == [K.COLOR_BAD, K.COLOR_GOOD, K.COLOR_GOOD, K.COLOR_BAD, K.COLOR_GOOD, K.COLOR_GOOD], measure
    assert V.build_mc_chart(small_mc, name, "Deckung").layout.shapes[0].y0 == pytest.approx(0.93) and V.build_mc_chart(small_mc, name, "RMSE").layout.yaxis.type == "log"
    e = small_mc["scenarios"][name]
    assert list(V.build_mc_chart(small_mc, name, "RMSE").data[0].y) == pytest.approx([e["mc"][m]["rmse"] for m in K.STRATEGIES]) and list(V.build_mc_chart(small_mc, name, "Bias (Betrag)").data[0].y) == pytest.approx([abs(e["mc"][m]["bias"]) for m in K.STRATEGIES])
    df = V.build_mc_table(small_mc, name)
    assert list(df["Strategie"]) == list(K.STRATEGIES) and list(df["gültig"]) == ["nein", "ja", "ja", "nein", "ja", "ja"] and list(df.columns)[-1] == "Breite (Median)"
    w = V.build_winner_table(small_mc, "RMSE")
    assert len(w) == 9 and list(w.columns) == ["Szenario", "gültig", "Sieger (alle sechs)", "Sieger unter den gültigen"] and w["gültig"][3] == "keine" and w["Sieger unter den gültigen"][3] == "keine"


def test_vortest_figures(small_vortest):
    fig = V.build_vortest_chart(small_vortest)
    assert [t.name for t in fig.data] == ["Vortest verwirft (%)", "Deckung der Vortest-Strategie (%)"] and list(fig.data[0].x) == list(K.VORTEST_SCENARIOS)
    df = V.build_vortest_table(small_vortest)
    assert len(df) == len(K.VORTEST_SCENARIOS) and set(df["n"]) == {200} and list(df.columns)[2] == "Vortest verwirft"


def test_regret_tables():
    rows, detail = V.build_regret_tables(ev.regret_experiment(("Kein Anstoß", "Alles beobachtet"), 1000))
    assert list(rows["Strategie"]) == list(K.STRATEGIES) and rows["größte Reue"][5] == "unendlich" and rows["Szenarien mit Reue < 0,05"][4] == "2 von 2"
    assert list(detail["Szenario"]) == ["Kein Anstoß", "Alles beobachtet"] and detail["2SLS"][0] == "∞" and detail["Backdoor {Z, W}"][1] == "0,000"
