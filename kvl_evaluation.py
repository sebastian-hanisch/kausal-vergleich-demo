"""Auswertung: eine Stichprobe der Welt mit allen sechs Strategien, Gültigkeit am Diagramm, Checkliste der Annahmen, fünf Experimente (exakte Grenzwerte, Preis der Gültigkeit und Kreuzungsgröße, Sieger nach Maß, Vortest, Reue) und die Verdikte.
Nur numpy und die Module der Linie, kein Streamlit."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

import kvl_constants as K
import kvl_strategies as S
import kvl_validity as V
import kvl_world as W
from kausal_generator import SplitMix64

COVER_OK = 0.93                                                          # „deckt“: mindestens 93 % der Intervalle (95 % minus Zufall bei 300 Läufen)


def de(x: float, digits: int = 2) -> str:
    """Zahl mit deutschem Dezimalkomma und typografischem Minuszeichen."""
    return f"{x:.{digits}f}".replace(".", ",").replace("-", "−")


def signed(x: float, digits: int = 2) -> str:
    x = 0.0 if abs(x) < 0.5 * 10 ** -digits * 1e-3 else x
    return ("+" if x >= 0 else "−") + de(abs(x), digits)


def pct(x: float, digits: int = 0) -> str:
    return de(100 * x, digits) + " %"


def _n(n: float) -> str:
    return f"{int(round(n)):,}".replace(",", " ") if math.isfinite(n) else "—"


def names_of(strategies) -> str:
    return ", ".join(strategies) if strategies else "keine"


@dataclass(frozen=True)
class Settings:
    a: float = K.KNOBS["a"][0]
    b: float = K.KNOBS["b"][0]
    pi: float = K.KNOBS["pi"][0]
    c: float = K.KNOBS["c"][0]
    e: float = K.KNOBS["e"][0]
    f: float = K.KNOBS["f"][0]
    g: float = K.KNOBS["g"][0]
    h: float = K.KNOBS["h"][0]
    n: int = K.DEFAULT_N
    seed: int = K.DEFAULT_SEED

    @property
    def knobs(self) -> dict:
        return {k: getattr(self, k) for k in K.KNOBS}


def settings_from(preset: dict) -> Settings:
    return Settings(**preset["knobs"], n=preset["n"], seed=preset["seed"])


# ----------------------------------------------------------------------------- Analyse einer Stichprobe

@dataclass
class Row:
    strategy: str
    estimate: float
    se: float
    lo: float
    hi: float
    error: float                 # Schätzung minus Wahrheit
    covers: bool
    valid: bool                  # Annahme gilt im Diagramm der Welt
    limit_bias: float            # exakt, nan wo der Grenzwert nicht existiert
    sd1: float                   # Streuung je √n (exakt)
    reason: str


@dataclass
class Analysis:
    settings: Settings
    truth: float
    rows: list
    valid: list                  # Namen der gültigen Strategien
    best: str | None             # gültig mit der kleinsten Streuung
    sd_ratio: dict               # Streuung gültiger Strategien gegen die kleinste gültige
    first_stage_f: float
    hausman_z: float
    hausman_rejected: bool
    holds: dict                  # gilt die Annahme „backdoor“, „frontdoor“, „instrument“ in dieser Welt


def assumption_holds(valid: dict) -> dict:
    return {"backdoor": valid["Backdoor {Z, W}"], "frontdoor": valid["Frontdoor"], "instrument": valid["2SLS"]}


def analyse(s: Settings) -> Analysis:
    k = s.knobs
    tau = W.truth(k)
    cov = np.cov(W.simulate(k, s.n, SplitMix64(s.seed)), rowvar=False)
    est = S.estimates(cov, s.n)
    lim = S.limits(k)
    val = V.validity(k)
    rows = []
    for m in K.STRATEGIES:
        e, se = est[m]
        rows.append(Row(m, float(e), float(se), float(e - 1.96 * se), float(e + 1.96 * se), float(e - tau), bool(abs(e - tau) <= 1.96 * se), val[m], lim[m][0], lim[m][1], V.reason(k, m)))
    valid = [m for m in K.STRATEGIES if val[m]]
    finite = [m for m in valid if math.isfinite(lim[m][1])]
    best = min(finite, key=lambda m: lim[m][1]) if finite else None
    ratio = {m: lim[m][1] / lim[best][1] for m in finite} if best else {}
    f = S.first_stage_f(cov, s.n)
    z = S.hausman_z(est)
    return Analysis(s, tau, rows, valid, best, ratio, f, z, bool(z > 1.96), assumption_holds(val))


def admissible(ticked) -> list:
    """Strategien, die die angekreuzten Annahmen erlauben (nur diese; naiv und „alles anpassen“ erlaubt keine Annahme)."""
    return [m for key, _, ms in K.ASSUMPTIONS if key in ticked for m in ms]


def recommend(a: Analysis, ticked) -> dict:
    """Aus den angekreuzten Annahmen: zulässige Strategien, die mit der kleinsten Streuung in dieser Welt, und ob die Annahmen in der Welt auch gelten (das weiß nur die Demo)."""
    adm = admissible(ticked)
    sd = {r.strategy: r.sd1 for r in a.rows}
    finite = [m for m in adm if math.isfinite(sd[m])]
    pick = min(finite, key=lambda m: sd[m]) if finite else None
    false = [key for key in ticked if not a.holds[key]]
    return {"admissible": adm, "pick": pick, "false": false}


# ----------------------------------------------------------------------------- Experiment 1: exakte Grenzwerte aller Strategien in den neun Szenarien

def limits_experiment() -> dict:
    """Je Szenario: Wahrheit, je Strategie exakter Grenzwert-Bias, Streuung je √n und Gültigkeit am Diagramm."""
    out = {}
    for name, p in K.PRESETS.items():
        k = p["knobs"]
        lim, val = S.limits(k), V.validity(k)
        out[name] = {"truth": W.truth(k), "bias": {m: lim[m][0] for m in K.STRATEGIES}, "sd1": {m: lim[m][1] for m in K.STRATEGIES}, "valid": val}
    return out


# ----------------------------------------------------------------------------- Experiment 2: Preis der Gültigkeit und Kreuzungsgröße

def rmse_curves(k: dict, ns=K.EXP_CURVE_NS) -> dict:
    """Asymptotischer RMSE je Strategie über n (Bias² + Streuung² / n); nan beim 2SLS ohne Anstoß."""
    lim = S.limits(k)
    return {"ns": tuple(ns), "curves": {m: [S.asymptotic_rmse(lim[m][0], lim[m][1], n) for n in ns] for m in K.STRATEGIES}, "limits": lim}


def crossing_table(k: dict, ss=K.EXP_CROSS_S) -> list:
    """Verborgene Ursache mit a = b = s (Anstoß π und c wie k; die Verletzungen e, f, g, h auf null, denn die Kreuzung vergleicht das Backdoor mit gültigen Strategien): Bias des Backdoor {Z, W}, Streuungen und die Kreuzungsgröße gegen 2SLS und gegen Frontdoor."""
    rows = []
    for s in ss:
        lim = S.limits({**k, "a": s, "b": s, "e": 0.0, "f": 0.0, "g": 0.0, "h": 0.0})
        bias, sd_bd = lim["Backdoor {Z, W}"]
        rows.append({"s": s, "bias": bias, "sd_bd": sd_bd, "sd_iv": lim["2SLS"][1], "sd_fd": lim["Frontdoor"][1], "n_iv": S.crossing_n(bias, sd_bd, lim["2SLS"][1]), "n_fd": S.crossing_n(bias, sd_bd, lim["Frontdoor"][1])})
    return rows


# ----------------------------------------------------------------------------- Experiment 3: wer gewinnt nach welchem Maß

def mc_experiment(n: int = K.DEFAULT_N, reps: int = K.EXP_MC_REPS) -> dict:
    """Alle neun Szenarien, alle sechs Strategien, `reps` feste Datensätze der Größe n."""
    out = {}
    for i, (name, p) in enumerate(K.PRESETS.items()):
        k = p["knobs"]
        out[name] = {"mc": S.monte_carlo(k, n, reps, K.SEED_BASE + 100000 * (i + 1)), "valid": V.validity(k), "truth": W.truth(k)}
    return {"n": n, "reps": reps, "scenarios": out}


MEASURE_KEYS = {"Bias (Betrag)": "bias", "RMSE": "rmse", "Median-Fehler": "median_error", "Deckung": "cover", "Breite": "width"}


def winners(entry: dict, measure: str, only_valid: bool = False) -> list:
    """Sieger eines Szenarios nach einem Maß: die Strategie mit dem kleinsten Wert (bei der Deckung: alle mit Deckung ≥ 93 %); nur unter den gültigen, wenn verlangt."""
    pool = [m for m in K.STRATEGIES if entry["valid"][m]] if only_valid else list(K.STRATEGIES)
    if not pool:
        return []
    key = MEASURE_KEYS[measure]
    if measure == "Deckung":
        return [m for m in pool if entry["mc"][m]["cover"] >= COVER_OK]
    val = {m: abs(entry["mc"][m]["bias"]) if key == "bias" else entry["mc"][m][key] for m in pool}
    return [min(pool, key=lambda m: val[m])]


# ----------------------------------------------------------------------------- Experiment 4: Einigkeit ist keine Gültigkeit

def vortest_experiment(ns=K.EXP_VORTEST_NS, reps: int = K.EXP_VORTEST_REPS) -> dict:
    """Hausman-Vortest zwischen Backdoor {Z, W} und 2SLS; Strategie „2SLS, wenn verworfen und F ≥ 10, sonst Backdoor“."""
    out = {}
    for ni, n in enumerate(ns):
        for si, name in enumerate(K.VORTEST_SCENARIOS):
            k = K.PRESETS[name]["knobs"]
            tau = W.truth(k)
            rej, fs, pick, bd, iv = [], [], [], [], []
            for r in range(reps):
                cov = np.cov(W.simulate(k, n, SplitMix64(K.SEED_BASE + 5000000 + 100000 * ni + 10000 * si + r)), rowvar=False)
                est = S.estimates(cov, n)
                z, f = S.hausman_z(est), S.first_stage_f(cov, n)
                rej.append(z > 1.96)
                fs.append(f)
                chosen = est["2SLS"] if (z > 1.96 and f >= K.FIRST_STAGE_F) else est["Backdoor {Z, W}"]
                pick.append(chosen)
                bd.append(est["Backdoor {Z, W}"][0])
                iv.append(est["2SLS"][0])
            pick = np.array(pick)
            out[(name, n)] = {"reject": float(np.mean(rej)), "f": float(np.mean(fs)), "rmse": float(np.sqrt(np.mean((pick[:, 0] - tau) ** 2))), "cover": float(np.mean(np.abs(pick[:, 0] - tau) <= 1.96 * pick[:, 1])),
                              "rmse_bd": float(np.sqrt(np.mean((np.array(bd) - tau) ** 2))), "median_error_iv": float(np.median(np.abs(np.array(iv) - tau)))}
    return {"ns": tuple(ns), "reps": reps, "cells": out}


# ----------------------------------------------------------------------------- Experiment 5: Reue über die plausiblen Szenarien

def regret_experiment(scenarios, n: int) -> dict:
    """Reue je Strategie: asymptotischer RMSE minus der der besten gültigen Strategie, je gewähltem Szenario (n fest); Szenarien ohne gültige Strategie werden übersprungen."""
    table, skipped = {}, []
    for name in scenarios:
        k = K.PRESETS[name]["knobs"]
        lim, val = S.limits(k), V.validity(k)
        rm = {m: (S.asymptotic_rmse(lim[m][0], lim[m][1], n) if not math.isnan(lim[m][0]) else math.inf) for m in K.STRATEGIES}
        valid = [m for m in K.STRATEGIES if val[m]]
        if not valid:
            skipped.append(name)
            continue
        best = min(rm[m] for m in valid)
        table[name] = {m: rm[m] - best for m in K.STRATEGIES}
    rows = {}
    for m in K.STRATEGIES:
        reg = [table[nm][m] for nm in table]
        rows[m] = {"max": max(reg) if reg else math.nan, "mean": float(np.mean(reg)) if reg else math.nan, "below": sum(r < 0.05 for r in reg), "count": len(reg)}
    pick = min(K.STRATEGIES, key=lambda m: rows[m]["max"]) if table else None
    return {"n": n, "table": table, "rows": rows, "pick": pick, "skipped": skipped}


# ----------------------------------------------------------------------------- Verdikte

def verdict_analysis(a: Analysis) -> tuple:
    s = a.settings
    wrong = [r for r in a.rows if not r.valid and not r.covers]
    head = f"Wahrheit {de(a.truth)}; Schätzungen aus n = {_n(s.n)}: " + ", ".join(f"{r.strategy} {de(r.estimate)}" for r in a.rows) + ". "
    if not a.valid:
        return "error", head + f"Keine der sechs Strategien ist gültig: {len(wrong)} verfehlen die Wahrheit mit einem Intervall, das sie ausschließt (Breiten {de(min(r.hi - r.lo for r in wrong), 2)} bis {de(max(r.hi - r.lo for r in wrong), 2)}), und keine Zahl warnt davor. Hier hilft keine Wahl der Strategie, nur die Sensitivität."
    text = head + f"Gültig: {names_of(a.valid)}; die kleinste Streuung hat {a.best}"
    if a.best is not None:
        others = [f"{m} {de(v, 1)}-fach" for m, v in a.sd_ratio.items() if m != a.best]
        text += (" (" + ", ".join(others) + ")") if others else ""
    text += ". "
    if wrong:
        text += f"Ungültig und trotzdem mit engem Intervall daneben: {names_of(r.strategy for r in wrong)}. "
    if "2SLS" in a.valid and a.first_stage_f < K.FIRST_STAGE_F:
        return "warning", text + f"Der Anstoß ist schwach (F = {de(a.first_stage_f, 1)}): 2SLS ist gültig, aber seine Schätzung streut stark und der Mittelwert ist unbrauchbar."
    return "success", text.strip()


def verdict_recommend(a: Analysis, ticked) -> tuple:
    r = recommend(a, ticked)
    if not ticked:
        return "warning", "Keine Annahme angekreuzt: Dann ist keine der sechs Strategien zulässig. Die Sensitivität (Stück 11) zeigt, wie stark eine verborgene Ursache sein dürfte; eine Zahl zu nennen, ohne eine Annahme zu vertreten, geht nicht."
    names = {key: text for key, text, _ in K.ASSUMPTIONS}
    if r["false"]:
        who = ", ".join(r["false"])
        return "error", f"Zulässig wären {names_of(r['admissible'])}; die kleinste Streuung hat {r['pick']}. In dieser Welt gilt die Annahme „{who}“ aber **nicht**: Die Strategie liegt dann um {signed(next(x.limit_bias for x in a.rows if x.strategy == r['pick']))} daneben. Das weiß nur die Demo, in der Wirklichkeit nie."
    return "success", f"Zulässig: {names_of(r['admissible'])}; die kleinste Streuung hat {r['pick']}. Die angekreuzten Annahmen gelten in dieser Welt."


def verdict_limits(res: dict) -> tuple:
    names = list(res)
    count = {m: sum(res[n]["valid"][m] for n in names) for m in K.STRATEGIES}
    none = [n for n in names if not any(res[n]["valid"].values())]
    return "info", (f"In {len(names)} Szenarien ist {max(count, key=count.get)} am häufigsten gültig ({count[max(count, key=count.get)]}), 2SLS in {count['2SLS']}, Backdoor {{Z, W}} in {count['Backdoor {Z, W}']}; "
                    f"naiv und „alles anpassen“ in keinem ({count['naiv']} und {count['alles anpassen']}). In {len(none)} Szenario ({none[0] if none else '–'}) ist keine der sechs gültig. Gültigkeit am Diagramm und exakter Grenzwert stimmen in allen Zellen überein.")


def verdict_crossing(rows: list) -> tuple:
    r3 = min(rows, key=lambda r: abs(r["s"] - 0.3))
    if not math.isfinite(r3["n_iv"]):
        return "info", (f"Bei einer verborgenen Ursache der Stärke {de(r3['s'], 1)} hat das Backdoor den Bias {de(r3['bias'], 3)}; das 2SLS hat hier keine Kreuzung mit ihm (kein Anstoß oder kein Bias), "
                        f"gegen Frontdoor liegt sie bei n = {_n(r3['n_fd'])}.")
    return "info", (f"Bei einer verborgenen Ursache der Stärke {de(r3['s'], 1)} (a = b) hat das Backdoor den Bias {de(r3['bias'], 3)} und streut {de(r3['sd_bd'], 2)} je √n, das 2SLS {de(r3['sd_iv'], 2)}: "
                    f"Erst ab n = {_n(r3['n_iv'])} schlägt das gültige 2SLS das verzerrte Backdoor nach RMSE. Bei s = {de(rows[0]['s'], 1)} liegt die Kreuzung bei {_n(rows[0]['n_iv'])}, bei s = {de(rows[-1]['s'], 1)} bei {_n(rows[-1]['n_iv'])}.")


def verdict_mc(res: dict) -> tuple:
    sc = res["scenarios"]
    names = list(sc)
    first = names[1]                                                      # „Alles beobachtet“: vier gültige Strategien
    gb = winners(sc[first], "Bias (Betrag)", True)
    gr = winners(sc[first], "RMSE", True)
    default = sc[names[0]]
    return "info", (f"Bei „{first}“ sind vier Strategien gültig: nach RMSE gewinnt {gr[0]}, nach Bias-Betrag {gb[0]} (zwischen gültigen entscheidet das Rauschen). "
                    f"Bei „{names[0]}“ (n = {_n(res['n'])}) gewinnt nach RMSE {winners(default, 'RMSE')[0]}, nach Deckung (≥ 93 %) {names_of(winners(default, 'Deckung'))}; {res['reps']} feste Läufe je Zelle.")


def verdict_vortest(res: dict) -> tuple:
    c = res["cells"]
    n = max(res["ns"])
    ok, hid, direct, coupled = (c[(name, n)] for name in ("Alles beobachtet", "Verborgene Ursache, Anstoß da (Voreinstellung)", "Anstoß wirkt direkt", "Anstoß hängt an der Ursache"))
    return "info", (f"Bei n = {_n(n)}: ohne verborgene Ursache verwirft der Vortest in {pct(ok['reject'], 1)} (die Größe), mit verborgener Ursache in {pct(hid['reject'])}; wirkt der Anstoß direkt, in {pct(direct['reject'])} und die Strategie „2SLS, wenn verworfen“ "
                    f"wählt das falsche (RMSE {de(direct['rmse'], 2)}, Deckung {pct(direct['cover'])}); hängt der Anstoß an der Ursache, verwirft er nur in {pct(coupled['reject'], 1)}: beide sind gleich verzerrt. Er zeigt Uneinigkeit, nicht Gültigkeit.")


def verdict_regret(res: dict) -> tuple:
    if res["pick"] is None:
        return "warning", "Keines der gewählten Szenarien hat eine gültige Strategie: Reue ist nicht definiert."
    r = res["rows"][res["pick"]]
    text = f"Über die {r['count']} gewählten Szenarien mit gültiger Strategie (n = {_n(res['n'])}) hat {res['pick']} die kleinste größte Reue ({de(r['max'], 2)}); unter {r['below']} von {r['count']} Szenarien liegt sie unter 0,05."
    if res["skipped"]:
        text += f" Ohne gültige Strategie, daher übersprungen: {names_of(res['skipped'])}."
    return "info", text
