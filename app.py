"""Welches Verfahren, gemessen woran? - interaktive Konzept-Demo
Sebastian Hanisch - Operations Research und Machine Learning

Siebzehntes und letztes Stück der Linie Kausale Diagramme und Adjustierung. Die sechs Schätzstrategien der Linie (naiv, Backdoor, alles anpassen, Frontdoor, 2SLS) stehen in einer Welt mit verborgener Ursache,
Anstoß und Mediator nebeneinander vor der bekannten do-Wahrheit: gemessen an Bias, Streuung, Deckung und Breite, mit der Gültigkeit am Diagramm und dem Preis jeder Annahme.
Kern: kvl_world.py, kvl_strategies.py, kvl_validity.py, kvl_evaluation.py; Kriterien: bdr_backdoor.py, frd_criterion.py, ins_criterion.py, dsep_graph.py (wortgleich aus Stück 2, 3, 8, 9).

Lauffähig mit: streamlit run app.py
"""

import streamlit as st

import kvl_constants as K
from kvl_evaluation import (
    Settings,
    analyse,
    crossing_table,
    de,
    limits_experiment,
    mc_experiment,
    names_of,
    regret_experiment,
    rmse_curves,
    verdict_analysis,
    verdict_crossing,
    verdict_limits,
    verdict_mc,
    verdict_recommend,
    verdict_regret,
    verdict_vortest,
    vortest_experiment,
)
from kvl_pdf_export import build_pdf
from kvl_presets import apply_preset, bounds, init_session_state_defaults, load_permalink_settings, randomize_seed, sync_query_params
from kvl_visualization import (
    build_crossing_table,
    build_curves_chart,
    build_estimate_chart,
    build_limits_table,
    build_mc_chart,
    build_mc_table,
    build_regret_tables,
    build_strategy_table,
    build_summary_table,
    build_vortest_chart,
    build_vortest_table,
    build_winner_table,
    build_world_diagram,
    style_limits_table,
    style_strategy_table,
)

st.set_page_config(page_title="Welches Verfahren, gemessen woran? – Sebastian Hanisch", layout="wide")

SPINNER = "Experiment rechnet über feste Zufallsströme …"


@st.cache_data(show_spinner=False)
def _analyse(settings):
    return analyse(settings)


@st.cache_data(show_spinner=False)
def _exp1():
    return limits_experiment()


@st.cache_data(show_spinner=SPINNER)
def _exp3(n):
    return mc_experiment(n)


@st.cache_data(show_spinner=SPINNER)
def _exp4():
    return vortest_experiment()


def verdict(kind_text):
    kind, text = kind_text
    getattr(st, kind)(text)


def experiment_button(label, key):
    if st.button(label, key=f"{key}_run"):
        st.session_state[f"{key}_ready"] = True
    return st.session_state.get(f"{key}_ready", False)


st.title("⚖️ Welches Verfahren, gemessen woran?")
st.markdown(
    """
**Es gibt kein bestes Verfahren, nur eines, dessen Annahme gilt, und das Maß entscheidet, welches „das beste“ heißt.** Dies ist das siebzehnte und letzte Stück der Linie **Kausale Diagramme und Adjustierung**. Die Stücke 3 bis 9 haben je eine Strategie gebaut; hier stehen **sechs nebeneinander**
in einer Welt mit **verborgener Ursache, Anstoß und Mediator** vor der bekannten **do-Wahrheit**: naiv, Backdoor (zwei Mengen), alles anpassen, Frontdoor, 2SLS. Gemessen werden **Bias, Streuung, Deckung und Breite**, dazu ob die Annahme der Strategie **im Diagramm gilt**.
Die Wahrheit kennt in der Wirklichkeit niemand; deshalb ist das Orakel nötig, und deshalb zeigt die Demo auch, **was ein Vergleich der Schätzungen untereinander nicht verrät**.

Weiter unten: die **Checkliste der Annahmen**, fünf Experimente (**🔬**), **🚧 Wo die Annahmen enden** und der Expander **📐 Mathematische Formulierung**.
"""
)

with st.expander("So liest man das Ergebnis", expanded=True):
    st.markdown(
        """
1. **Die Welt** (links): Nachfrage Z und Wetter W wirken auf den Servicegrad Y, Z auch auf den Faktor X; ein **Anstoß** I verschiebt X; eine **verborgene Standortqualität** U (orange) treibt X, Y und (wenn gewählt) den Bestand M und den Anstoß; X wirkt über den **Bestand** M auf Y (und, wenn gewählt, direkt). **Rote Pfeile** verletzen die Annahme einer Strategie.
2. **Sechs Strategien** (rechts): Schätzung mit 95-%-Intervall gegen die **Wahrheit** (orange gestrichelt). **Grün**: Die Annahme gilt im Diagramm der Welt. **Rot**: Sie gilt nicht, und das Intervall ist trotzdem eng.
3. **Gültig** heißt: Backdoor, Frontdoor bzw. Instrument-Kriterium am Diagramm erfüllt; das stimmt mit dem exakten Grenzwert-Bias überein (Bias null).
4. **Preis der Gültigkeit**: Unter den gültigen streut die mit den meisten Annahmen (Backdoor) am wenigsten, die mit den wenigsten (2SLS) am meisten.
5. **Checkliste**: Welche Annahmen Sie vertreten, bestimmt, welche Strategien zulässig sind. Ob sie in Ihrer Welt gelten, zeigt nur die Demo.
        """
    )

st.caption("🎯 Schnellstart – ein Szenario laden:")
names = list(K.PRESETS)
for chunk in (names[:5], names[5:]):
    cols = st.columns(5)
    for col, name in zip(cols, chunk):
        with col:
            st.button(name, width="stretch", on_click=apply_preset, args=(name,), help=K.PRESET_HELP.get(name, ""), key=f"preset_{name}")

st.caption("🔗 Die Adresszeile oben spiegelt Ihre aktuelle Konfiguration wider – einfach kopieren, um ein Szenario zu teilen.")

load_permalink_settings()
init_session_state_defaults()

GROUPS = (("Verborgene Ursache", ("a", "b")), ("Anstoß", ("pi", "g", "h")), ("Weg über den Bestand", ("c", "e", "f")))
with st.sidebar:
    st.header("⚙️ Einstellungen")
    values = {}
    for title, keys in GROUPS:
        st.markdown(f"**{title}**")
        for k in keys:
            lo, hi = bounds(f"{k}_slider")
            values[k] = st.slider(K.KNOB_LABELS[k], lo, hi, step=K.KNOBS[k][3], key=f"{k}_slider", help=K.KNOB_HELP[k])
    st.markdown("**Daten**")
    n = st.slider("Beobachtungen", *bounds("n_slider"), step=K.N_STEP, key="n_slider", help="Zeilen der Stichprobe; der Vergleich der Experimente 3 und 5 gilt für dieses n.")
    seed = st.number_input("Zufalls-Seed der Stichprobe", *bounds("seed_input"), key="seed_input", step=1)
    st.button("🎲 Neue Stichprobe würfeln", width="stretch", on_click=randomize_seed)

sync_query_params({**{f"{k}_slider": round(float(v), 2) for k, v in values.items()}, "n_slider": int(n), "seed_input": int(seed)})

settings = Settings(**{k: round(float(v), 2) for k, v in values.items()}, n=int(n), seed=int(seed))
a = _analyse(settings)

# --- Ergebnis -----------------------------------------------------------------------------------------------------------------------------

st.markdown("## 🎯 Sechs Strategien vor derselben Wahrheit")
(c1, c2), (c3, c4) = st.columns(2), st.columns(2)
c1.metric("Wahrheit (do)", de(a.truth, 2), help="Gesamtwirkung des Faktors auf den Servicegrad bei Eingriff: c · 0,6 + e.")
c2.metric("Gültige Strategien", f"{len(a.valid)} von 6", delta=names_of(a.valid), delta_color="off", help="Strategien, deren Annahme im Diagramm der Welt erfüllt ist (Kriterium des Diagramms, dem exakten Grenzwert-Bias gleich).")
c3.metric("Kleinste Streuung unter den gültigen", a.best or "—", delta=f"{de(a.rows[[r.strategy for r in a.rows].index(a.best)].sd1, 2)} je √n" if a.best else "keine gültige", delta_color="off", help="Exakte Streuung der Schätzung je Wurzel aus n.")
c4.metric("Erste Stufe des Anstoßes (F)", de(a.first_stage_f, 1), delta="schwach" if a.first_stage_f < K.FIRST_STAGE_F else "ausreichend", delta_color="off", help="F-Statistik des Anstoßes in der Regression des Faktors; unter 10 gilt der Anstoß als schwach.")

left, right = st.columns(2)
with left:
    st.plotly_chart(build_world_diagram(settings.knobs), width="stretch", key="world_chart")
    st.caption("Die Welt. X (grün umrandet) ist der Faktor, Y der Servicegrad, U die verborgene Standortqualität (orange, gestrichelt gedacht: nie beobachtet), C die Beschwerden. Rote Pfeile verletzen eine Annahme.")
with right:
    st.plotly_chart(build_estimate_chart(a), width="stretch", key="estimate_chart")
    st.caption("Schätzung und 95-%-Intervall jeder Strategie aus derselben Stichprobe, gegen die Wahrheit.")
verdict(verdict_analysis(a))
st.dataframe(style_strategy_table(build_strategy_table(a)), hide_index=True, width="stretch")
st.dataframe(build_summary_table(a), hide_index=True, width="stretch")

st.markdown("### ✅ Welche Annahmen vertreten Sie?")
ticked = []
for key, text, ms in K.ASSUMPTIONS:
    if st.checkbox(text, key=f"assume_{key}", help="Zulässig: " + ", ".join(ms)):
        ticked.append(key)
verdict(verdict_recommend(a, tuple(ticked)))
st.caption("„Naiv“ und „alles anpassen“ erlaubt keine Annahme: naiv nur ohne jede gemeinsame Ursache, „alles anpassen“ nie in dieser Welt (der Bestand und die Beschwerden sind Wirkungen des Faktors).")

if st.button("📄 PDF der aktuellen Auswertung erstellen", key="pdf_make"):
    st.session_state["pdf_made"] = (settings, build_pdf(a))
made = st.session_state.get("pdf_made")
if made and made[0] == settings:
    st.download_button("⬇️ PDF herunterladen", data=made[1], file_name="verfahrensvergleich.pdf", mime="application/pdf", key="pdf_download")

st.markdown("---")

# --- Experimente ---------------------------------------------------------------------------------------------------------------------------

st.subheader("🔬 Experiment 1: Wer ist wo gültig? (exakt)")
st.caption("Alle sechs Strategien in den neun Szenarien: exakter Grenzwert-Bias gegen die Wahrheit (kein Zufall), dazu ✓ oder ✗, ob das Kriterium am Diagramm erfüllt ist. Beides stimmt in jeder Zelle überein.")
res1 = _exp1()
verdict(verdict_limits(res1))
st.dataframe(style_limits_table(build_limits_table(res1)), hide_index=True, width="stretch")

st.subheader("🔬 Experiment 2: Der Preis der Gültigkeit und die Kreuzungsgröße (exakt)")
st.caption("Asymptotischer RMSE jeder Strategie über n in der Welt der Seitenleiste, und die Stichprobengröße n*, ab der das gültige (aber streuende) Verfahren das verzerrte (aber ruhige) Backdoor nach RMSE schlägt: n* = (σ²_gültig − σ²_Backdoor) / Bias². Bei schwachem Anstoß ist die Asymptotik des 2SLS unbrauchbar. Die Tabelle rechnet ohne die Verletzungen e, f, g und h (sie vergleicht mit gültigen Strategien).")
curves = rmse_curves(settings.knobs)
st.plotly_chart(build_curves_chart(curves), width="stretch", key="curves_chart")
cross = crossing_table(settings.knobs)
verdict(verdict_crossing(cross))
st.dataframe(build_crossing_table(cross), hide_index=True, width="stretch")

st.subheader("🔬 Experiment 3: Wer gewinnt nach welchem Maß?")
st.caption(f"Alle neun Szenarien, alle sechs Strategien, {K.EXP_MC_REPS} feste Datensätze der Größe n aus der Seitenleiste. Das Maß wählen Sie nach der Rechnung; der Sieger wechselt mit ihm.")
if experiment_button("▶ Experiment 3 rechnen (einige Sekunden)", "exp3"):
    res3 = _exp3(settings.n)
    verdict(verdict_mc(res3))
    measure = st.radio("Maß", K.MEASURES, horizontal=True, key="measure_radio", help=" ".join(f"{m}: {t}" for m, t in K.MEASURE_HELP.items()))
    scenario = st.selectbox("Szenario", list(K.PRESETS), key="scenario_select")
    st.plotly_chart(build_mc_chart(res3, scenario, measure), width="stretch", key="mc_chart")
    st.dataframe(build_mc_table(res3, scenario), hide_index=True, width="stretch")
    st.dataframe(build_winner_table(res3, measure), hide_index=True, width="stretch")

st.subheader("🔬 Experiment 4: Einigkeit ist keine Gültigkeit")
st.caption(f"Hausman-Vortest zwischen Backdoor {{Z, W}} und 2SLS in sechs Szenarien, {K.EXP_VORTEST_REPS} feste Datensätze bei n = {K.EXP_VORTEST_NS[0]} und {K.EXP_VORTEST_NS[1]}; die Strategie „2SLS, wenn verworfen und F ≥ 10, sonst Backdoor“.")
if experiment_button("▶ Experiment 4 rechnen (einige Sekunden)", "exp4"):
    res4 = _exp4()
    st.plotly_chart(build_vortest_chart(res4), width="stretch", key="vortest_chart")
    verdict(verdict_vortest(res4))
    st.dataframe(build_vortest_table(res4), hide_index=True, width="stretch")

st.subheader("🔬 Experiment 5: Reue über die plausiblen Szenarien (exakt)")
st.caption("Welche Welten halten Sie für möglich? Die Reue einer Strategie in einem Szenario ist ihr asymptotischer RMSE minus der der besten gültigen Strategie (n aus der Seitenleiste); gewählt wird die Strategie mit der kleinsten größten Reue. Szenarien ohne gültige Strategie werden übersprungen.")
plausible = st.multiselect("Plausible Szenarien", list(K.PRESETS), default=list(K.PRESETS), key="plausible_select")
res5 = regret_experiment(tuple(plausible), settings.n)
verdict(verdict_regret(res5))
if res5["table"]:
    for table in build_regret_tables(res5):
        st.dataframe(table, hide_index=True, width="stretch")

st.markdown("---")

# --- Grenzen -------------------------------------------------------------------------------------------------------------------------------

st.subheader("🚧 Wo die Annahmen enden")
st.markdown(
    """
| Annahme | Was passiert, wenn sie verletzt ist | Wer setzt an |
|---|---|---|
| **Linear und Gauß'sch, ein Faktor, ein Ergebnis** | Alle Zahlen gelten für diese Welt; Schätzvarianten (Gewichten, Matching, doppelt robust) und nichtlineare Wirkungen sind nicht gebaut (Stück 6 und 7). | nicht gebaut |
| **Das Diagramm der Welt ist bekannt** | Die Gültigkeit wird am wahren Diagramm gerechnet; ein falsch gezeichnetes Diagramm ändert, welche Strategie man für gültig hält (Stück 16). | nicht gebaut |
| **Die Wahrheit ist bekannt** | Nur im Orakel. In der Praxis lässt sich nicht prüfen, ob eine einzelne Strategie gilt; der Vortest zeigt Uneinigkeit, nicht den Fehler (Experiment 4). | nicht gebaut |
| **Die Szenarien sind meine Wahl** | Reue und „am häufigsten gültig“ hängen an den neun Szenarien (Frontdoor ist in sechs nach Konstruktion gültig); sie sind ein Werkzeug für die Welten, die Sie für möglich halten, kein Befund über Verfahren. | nicht gebaut |
| **Ein Anstoß, ein Mediator, eine verborgene Ursache** | Mehrere Anstöße (Überidentifikation, J-Test), mehrere Mediatoren und mehrere verborgene Ursachen sind nicht gebaut; gelernte Diagramme (Stück 13 bis 15) und Übertragbarkeit (12) sind keine Strategien dieses Vergleichs. | nicht gebaut |
| **Wenn nichts gilt** | Das Szenario „Nichts brauchbar“ zeigt, dass keine der sechs hilft und kein Intervall warnt; was dann bleibt, ist die Sensitivität (Stück 11). | nicht gebaut |
"""
)

st.markdown("---")

with st.expander("📐 Mathematische Formulierung"):
    st.markdown(
        r"""
**Modell.** Lineares Gauß-Modell $x = Bx + e$ mit Einheitsrauschen; $Z \to X, Y$; $W \to Y$; $I \to X$ ($\pi$); $U \to X, Y, M, I$ ($a, b, f, h$); $X \to M$ ($c$); $M \to Y$ ($0{,}6$); $X \to Y$ ($e$); $I \to Y$ ($g$); $Y \to C$.
Wahrheit (Gesamtwirkung bei Eingriff): $\tau = 0{,}6\,c + e$.

**Strategien.** Backdoor: Koeffizient von $X$ in der Regression von $Y$ auf $X$ und die Menge; Frontdoor: $\hat c\,\hat d$ mit $\hat c$ aus $M \sim X$ und $\hat d$ aus $Y \sim M + X$ (Delta-Methode für den Standardfehler);
2SLS mit Kovariaten $Z, W$: $\hat\beta = \widetilde{\mathrm{Cov}}(I, Y) / \widetilde{\mathrm{Cov}}(I, X)$ (Teilkovarianzen, Kovariaten herausgerechnet), $\mathrm{Var}(\hat\beta) = \sigma^2\,\widetilde{\mathrm{Var}}(I)\,/\,(n\,\widetilde{\mathrm{Cov}}(I, X)^2)$.

**Gültigkeit.** Backdoor: kein Nachkomme von $X$ in der Menge, $X \perp Y$ gegeben die Menge im Diagramm ohne die Pfeile aus $X$; Frontdoor: der Bestand fängt alle gerichteten Pfade ab, kein offener Rückweg von $X$ zu ihm, alle Rückwege von ihm zu $Y$ durch $X$ blockiert;
Instrument: $I$ ist nicht von $X$ getrennt und im Diagramm ohne die Pfeile in $X$ von $Y$ getrennt. In der Welt gilt: **Kriterium erfüllt $\Leftrightarrow$ Grenzwert-Bias null**.

**Preis und Kreuzung.** Asymptotisch $\mathrm{RMSE}(n)^2 = \mathrm{Bias}^2 + \sigma^2/n$ ($\sigma$ die Streuung je $\sqrt n$). Das verzerrte Backdoor schlägt das gültige 2SLS, solange $\mathrm{Bias}^2 < (\sigma_{2SLS}^2 - \sigma_{BD}^2)/n$, also unter $n^* = (\sigma_{2SLS}^2 - \sigma_{BD}^2)/\mathrm{Bias}^2$.
Beim 2SLS mit exakter Identifikation fehlen die Momente; bei schwachem Anstoß zählen Median und Breite.

**Vortest.** Hausman: $z = |\hat\beta_{BD} - \hat\beta_{IV}| / \sqrt{\mathrm{Var}(\hat\beta_{IV}) - \mathrm{Var}(\hat\beta_{BD})}$, verworfen bei $z > 1{,}96$ (unter der Nullhypothese ist die Regression effizient). Er sieht nur die Differenz der beiden Schätzungen.

**Reue.** $\mathrm{Reue}(m, s) = \mathrm{RMSE}_{as}(m, s) - \min_{m'\ \mathrm{gültig\ in}\ s} \mathrm{RMSE}_{as}(m', s)$; gewählt wird $\arg\min_m \max_s \mathrm{Reue}(m, s)$ (Minimax-Reue).

**Literatur.** J. Pearl, *Causality*, 2009 (Backdoor, Frontdoor); J. Hausman, *Specification tests in econometrics*, Econometrica 1978; D. Staiger, J. Stock, *Instrumental variables regression with weak instruments*, Econometrica 1997;
L. Savage, *The theory of statistical decision*, JASA 1951 (alle aus dem Gedächtnis, nicht geprüft).
        """
    )

st.markdown("---")
st.caption(
    "Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – "
    "Operations Research und Machine Learning ([Über mich](https://sebastianhanisch.net/ueber-mich.html)). "
    "Mehr zu den Demos: [Interaktive Demos](https://sebastianhanisch.net/demos.html)."
)
