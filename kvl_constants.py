"""Konstanten: die Welt mit acht Größen, Regler, Szenarien, Strategien, feste Experimente, Farben und Presets."""

SEED_MAX, DEFAULT_SEED = 999999, 35

# Welt (Linie Kausale Diagramme und Adjustierung): Z (Nachfrage) → X, Y; W (Wetter) → Y; I (Anstoß) → X; U (verborgen) → X, Y, M; X → M (Bestand) → Y; X → Y direkt; Y → C (Beschwerden); Einheitsrauschen
Z, W, I, U, X, M, Y, C = range(8)
NAMES = ("Z", "W", "I", "U", "X", "M", "Y", "C")
LONG = ("Nachfrage", "Wetter", "Anstoß", "Standortqualität (verborgen)", "Faktor", "Bestand (Mediator)", "Servicegrad", "Beschwerden")
OBSERVED = (Z, W, I, X, M, Y, C)
FIXED = {(Z, X): 0.8, (Z, Y): 0.6, (W, Y): 0.6, (Y, C): 0.8}
D_EFFECT = 0.6                                                           # M → Y

# Regler: Name -> (Voreinstellung, kleinster Wert, größter Wert, Schritt)
KNOBS = {
    "a": (0.6, 0.0, 1.0, 0.1),        # verborgene Ursache → Faktor
    "b": (0.6, 0.0, 1.0, 0.1),        # verborgene Ursache → Servicegrad
    "pi": (0.5, 0.0, 1.0, 0.05),      # Anstoß → Faktor
    "c": (1.0, 0.2, 1.0, 0.1),        # Faktor → Bestand
    "e": (0.0, 0.0, 0.6, 0.1),        # Faktor → Servicegrad direkt
    "f": (0.0, 0.0, 0.6, 0.1),        # verborgene Ursache → Bestand
    "g": (0.0, 0.0, 0.6, 0.1),        # Anstoß → Servicegrad (verletzt den Ausschluss)
    "h": (0.0, 0.0, 0.6, 0.1),        # verborgene Ursache → Anstoß (verletzt die Unabhängigkeit)
}
KNOB_LABELS = {"a": "Verborgene Ursache → Faktor (a)", "b": "Verborgene Ursache → Servicegrad (b)", "pi": "Anstoß → Faktor (π)", "c": "Faktor → Bestand (c)", "e": "Faktor → Servicegrad direkt (e)",
               "f": "Verborgene Ursache → Bestand (f)", "g": "Anstoß → Servicegrad direkt (g)", "h": "Verborgene Ursache → Anstoß (h)"}
KNOB_HELP = {
    "a": "Wie stark die verborgene Standortqualität die Regelwahl treibt. 0: keine verborgene Ursache auf den Faktor.",
    "b": "Wie stark die verborgene Standortqualität den Servicegrad treibt. Bias der Backdoor-Regression nach Anpassung an Z, W: a b / Var(X | Z, W).",
    "pi": "Stärke des Anstoßes auf den Faktor (die erste Stufe des 2SLS). 0: kein Anstoß.",
    "c": "Wirkung des Faktors auf den Bestand. Die Gesamtwirkung ist c · 0,6 + e.",
    "e": "Direkter Pfeil vom Faktor zum Servicegrad am Bestand vorbei: verletzt die Frontdoor-Annahme (der Bestand fängt nicht alles ab).",
    "f": "Die verborgene Ursache wirkt auch auf den Bestand: verletzt die Frontdoor-Annahme (offener Rückweg vom Faktor zum Bestand).",
    "g": "Der Anstoß wirkt direkt auf den Servicegrad: verletzt die Ausschlussrestriktion des 2SLS.",
    "h": "Der Anstoß hängt an der verborgenen Ursache: verletzt die Unabhängigkeit des 2SLS.",
}
N_MIN, N_MAX, N_STEP, DEFAULT_N = 100, 5000, 100, 1000

STRATEGIES = ("naiv", "Backdoor {Z}", "Backdoor {Z, W}", "alles anpassen", "Frontdoor", "2SLS")
STRATEGY_HELP = {
    "naiv": "Regression des Servicegrads auf den Faktor allein (Stück 1).",
    "Backdoor {Z}": "Regression auf Faktor und Nachfrage (Stück 3): gültig, wenn die Nachfrage alle Rückwärtspfade blockiert.",
    "Backdoor {Z, W}": "Zusätzlich das Wetter, eine Ursache des Servicegrads (gute Kontrolle, Stück 4 und 5): dieselbe Annahme, kleinere Streuung.",
    "alles anpassen": "Alle übrigen beobachteten Größen angepasst (Nachfrage, Wetter, Anstoß, Bestand, Beschwerden): schlechte Kontrollen (Stück 4).",
    "Frontdoor": "Wirkung des Faktors auf den Bestand mal Wirkung des Bestands auf den Servicegrad (Stück 8): braucht keine Kontrolle der verborgenen Ursache.",
    "2SLS": "Zweistufige Kleinste Quadrate mit dem Anstoß als Instrument, Kovariaten Nachfrage und Wetter (Stück 9).",
}
ASSUMPTIONS = (
    ("backdoor", "Alle gemeinsamen Ursachen von Faktor und Servicegrad sind beobachtet (hier: Nachfrage und Wetter).", ("Backdoor {Z}", "Backdoor {Z, W}")),
    ("frontdoor", "Der Bestand fängt die ganze Wirkung des Faktors ab (kein direkter Pfeil), und keine verborgene Ursache wirkt auf ihn.", ("Frontdoor",)),
    ("instrument", "Der Anstoß wirkt nur über den Faktor, ist von der verborgenen Ursache unabhängig und nicht schwach.", ("2SLS",)),
)
MEASURES = ("Bias (Betrag)", "RMSE", "Median-Fehler", "Deckung", "Breite")
MEASURE_HELP = {
    "Bias (Betrag)": "Betrag der mittleren Abweichung von der Wahrheit: zwischen gültigen Strategien entscheidet hier das Rauschen.",
    "RMSE": "Wurzel des mittleren quadratischen Fehlers; beim 2SLS mit schwachem Anstoß unbrauchbar (die Momente fehlen).",
    "Median-Fehler": "Median des Fehlerbetrags: auch bei schwachem Anstoß stabil.",
    "Deckung": "Anteil der 95-%-Intervalle, die die Wahrheit enthalten: ohne die Breite nichts wert (ein riesiges Intervall deckt immer).",
    "Breite": "Median der Intervallbreite: Was die Annahme spart oder kostet.",
}
VALID_TOL = 0.02                                                         # Grenzwert-Bias unter diesem Betrag: Strategie gilt als gültig (exakt null in der Welt)
FIRST_STAGE_F = 10.0                                                     # Faustregel für den schwachen Anstoß

# Experimente: feste Ströme (Seeds aus SEED_BASE), nie ein einzelner Seed als Befund.
SEED_BASE = 1070000
EXP_MC_REPS = 300
EXP_VORTEST_REPS, EXP_VORTEST_NS = 400, (200, 1000)
EXP_CROSS_S = (0.1, 0.2, 0.3, 0.4, 0.6)
EXP_CURVE_NS = (50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000, 50000)
VORTEST_SCENARIOS = ("Alles beobachtet", "Verborgene Ursache, Anstoß da (Voreinstellung)", "Schwacher Anstoß", "Anstoß wirkt direkt", "Anstoß hängt an der Ursache", "Ursache wirkt auf den Mediator")

COLOR_TRUTH = "#D68A2E"
COLOR_GOOD = "#2F6B65"
COLOR_BAD = "#c0392b"
COLOR_NEUTRAL = "#8a8a8a"
COLOR_CONTROL = "#4c78a8"
STRATEGY_COLORS = {"naiv": COLOR_NEUTRAL, "Backdoor {Z}": COLOR_CONTROL, "Backdoor {Z, W}": COLOR_GOOD, "alles anpassen": COLOR_BAD, "Frontdoor": COLOR_TRUTH, "2SLS": "#7b5ea7"}


def knobs(**kw) -> dict:
    base = {k: v[0] for k, v in KNOBS.items()}
    base.update(kw)
    return base


def preset(**kw) -> dict:
    base = dict(knobs=knobs(**kw.pop("knobs", {})), n=DEFAULT_N, seed=DEFAULT_SEED)
    base.update(kw)
    return base


SCENARIO_SEEDS = {"Schwacher Anstoß": 2}                                 # eine Stichprobe, deren erste Stufe (F = 6,0) den schwachen Anstoß zeigt; der Seed 35 gäbe F = 12,9


SCENARIOS = {
    "Verborgene Ursache, Anstoß da (Voreinstellung)": {},
    "Alles beobachtet": dict(a=0.0, b=0.0),
    "Kein Anstoß": dict(pi=0.0),
    "Nichts brauchbar": dict(pi=0.0, e=0.3, c=0.5),
    "Schwacher Anstoß": dict(pi=0.1),
    "Anstoß wirkt direkt": dict(g=0.3),
    "Anstoß hängt an der Ursache": dict(h=0.3),
    "Teilweise vermittelt": dict(e=0.3, c=0.5),
    "Ursache wirkt auf den Mediator": dict(f=0.4),
}
PRESETS = {name: preset(knobs=kw, seed=SCENARIO_SEEDS.get(name, DEFAULT_SEED)) for name, kw in SCENARIOS.items()}
PRESET_HELP = {
    "Verborgene Ursache, Anstoß da (Voreinstellung)": "Die verborgene Standortqualität treibt Faktor und Servicegrad (je 0,6): Backdoor bleibt um +0,22 verzerrt, Frontdoor und 2SLS sind gültig, 2SLS streut 1,9-mal so stark wie Frontdoor.",
    "Alles beobachtet": "Keine verborgene Ursache: Backdoor, Frontdoor und 2SLS sind alle gültig; Backdoor {Z, W} streut am wenigsten, 2SLS am meisten (2,2-fach). „Alles anpassen“ löscht die ganze Wirkung (−0,60).",
    "Kein Anstoß": "Der Anstoß ist null: nur Frontdoor ist gültig. 2SLS hat kein Instrument, sein Intervall ist riesig und deckt trotzdem (Deckung ohne Breite).",
    "Nichts brauchbar": "Kein Anstoß und ein direkter Pfeil am Bestand vorbei: keine der sechs Strategien ist gültig, alle geben enge Intervalle um falsche Zahlen. Hier hilft nur die Sensitivität (Stück 11).",
    "Schwacher Anstoß": "Der Anstoß ist schwach (π = 0,1, F in dieser Stichprobe 6,0, im Mittel etwa 8): 2SLS ist gültig, aber Mittel und RMSE explodieren; Median und Breite zeigen es (Breite 1,7). Frontdoor ist gültig und ruhig.",
    "Anstoß wirkt direkt": "Der Anstoß wirkt auch direkt auf den Servicegrad: 2SLS ist um +0,60 verzerrt, ein Vortest zwischen Backdoor und 2SLS verwirft und wählt das falsche.",
    "Anstoß hängt an der Ursache": "Der Anstoß hängt an der verborgenen Ursache: 2SLS und Backdoor sind gleich verzerrt (+0,25), ein Vortest sieht keine Abweichung: Einigkeit ist keine Gültigkeit.",
    "Teilweise vermittelt": "Ein Teil der Wirkung geht direkt am Bestand vorbei (e = 0,3): Frontdoor misst nur den Weg über den Bestand (−0,30), 2SLS ist gültig.",
    "Ursache wirkt auf den Mediator": "Die verborgene Ursache wirkt auch auf den Bestand: Frontdoor ist um +0,21 verzerrt, 2SLS ist gültig.",
}
DEFAULT_SCENARIO = next(iter(SCENARIOS))
