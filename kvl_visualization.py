"""Plotly-Figuren und Tabellen. Alle Achsen fest (fixedrange), damit Touch-Scrollen nicht am Diagramm hängen bleibt."""

from __future__ import annotations

import math

import pandas as pd
import plotly.graph_objects as go

import kvl_constants as K
import kvl_evaluation as E
import kvl_world as W
from kvl_evaluation import Analysis, de, names_of, pct, signed

POS = {K.Z: (0.1, 1.0), K.U: (0.5, 1.0), K.W: (0.9, 1.0), K.I: (0.0, 0.5), K.X: (0.3, 0.5), K.M: (0.58, 0.5), K.Y: (0.86, 0.5), K.C: (0.86, -0.05)}
VIOLATIONS = {(K.X, K.Y), (K.U, K.M), (K.I, K.Y), (K.U, K.I)}


def _fixed(fig: go.Figure, height: int = 380, **layout) -> go.Figure:
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=30, b=10), legend=dict(orientation="h", y=-0.22), **layout)
    fig.update_xaxes(fixedrange=True)
    fig.update_yaxes(fixedrange=True)
    return fig


def _segment(p0, p1, r: float = 0.075):
    dx, dy = p1[0] - p0[0], p1[1] - p0[1]
    ln = (dx * dx + dy * dy) ** 0.5 or 1.0
    ux, uy = dx / ln, dy / ln
    return (p0[0] + ux * r, p0[1] + uy * r), (p1[0] - ux * r, p1[1] - uy * r)


def _arrow(fig: go.Figure, a: int, b: int, color: str, width: float = 3):
    s, e = _segment(POS[a], POS[b])
    fig.add_annotation(x=e[0], y=e[1], ax=s[0], ay=s[1], xref="x", yref="y", axref="x", ayref="y", showarrow=True, arrowhead=3, arrowsize=1.3, arrowwidth=width, arrowcolor=color, text="")


def build_world_diagram(k: dict) -> go.Figure:
    """Das Diagramm der Welt: graue Pfeile der Standardwelt, orange Pfeile der verborgenen Ursache, rote Pfeile, die eine Annahme verletzen."""
    fig = go.Figure()
    for (a, b), s in sorted(W.edge_weights(k).items()):
        color = K.COLOR_BAD if (a, b) in VIOLATIONS else K.COLOR_TRUTH if a == K.U else "#6b6b6b"
        _arrow(fig, a, b, color, 2 + 3 * min(abs(s), 1.0))
    for v, (x, y) in POS.items():
        hidden = v == K.U
        fig.add_trace(go.Scatter(x=[x], y=[y], mode="markers+text", text=[K.NAMES[v]], textposition="middle center", textfont=dict(size=12),
                                 marker=dict(size=52, color="rgba(214,138,46,0.08)" if hidden else "rgba(47,107,101,0.10)", line=dict(width=2, color=K.COLOR_TRUTH if hidden else K.COLOR_GOOD if v in (K.X, K.Y) else "#1b1b1b")),
                                 hoverinfo="skip", showlegend=False))
    fig.update_xaxes(visible=False, range=[-0.15, 1.1])
    fig.update_yaxes(visible=False, range=[-0.25, 1.2])
    return _fixed(fig, 330, showlegend=False)


def build_estimate_chart(a: Analysis) -> go.Figure:
    """Schätzung und 95-%-Intervall je Strategie gegen die Wahrheit; grün gültig, rot ungültig."""
    fig = go.Figure()
    rows = list(reversed(a.rows))
    fig.add_trace(go.Scatter(x=[r.estimate for r in rows], y=[r.strategy for r in rows], mode="markers", marker=dict(size=10, color=[K.COLOR_GOOD if r.valid else K.COLOR_BAD for r in rows]),
                             error_x=dict(type="data", symmetric=False, array=[r.hi - r.estimate for r in rows], arrayminus=[r.estimate - r.lo for r in rows], thickness=2.5, color="#555555"), name="Schätzung mit 95-%-Intervall"))
    fig.add_vline(x=a.truth, line=dict(color=K.COLOR_TRUTH, width=2, dash="dash"), annotation_text=f"Wahrheit {de(a.truth)}", annotation_position="top")
    fig.update_xaxes(title="geschätzte Wirkung des Faktors auf den Servicegrad (grün: Annahme gilt, rot: gilt nicht)")
    return _fixed(fig, 330, showlegend=False)


def build_strategy_table(a: Analysis) -> pd.DataFrame:
    return pd.DataFrame([{"Strategie": r.strategy, "Schätzung": de(r.estimate, 3), "95-%-Intervall": f"{de(r.lo, 2)} bis {de(r.hi, 2)}", "Fehler": signed(r.error, 3), "deckt die Wahrheit": "ja" if r.covers else "nein",
                          "Annahme gilt": "ja" if r.valid else "nein", "Grenzwert-Bias (exakt)": signed(r.limit_bias, 3) if not math.isnan(r.limit_bias) else "—", "Streuung je √n (exakt)": de(r.sd1, 2) if math.isfinite(r.sd1) else "—",
                          "Grund, wenn die Annahme nicht gilt": r.reason} for r in a.rows])


def style_strategy_table(df: pd.DataFrame):
    def row_style(row):
        return ["background-color: rgba(192,57,43,0.15)" if row["Annahme gilt"] == "nein" else ""] * len(row)

    return df.style.apply(row_style, axis=1)


def build_summary_table(a: Analysis) -> pd.DataFrame:
    s = a.settings
    return pd.DataFrame([
        {"Größe": "Wahrheit (Gesamtwirkung c · 0,6 + e)", "Wert": de(a.truth, 3)},
        {"Größe": "Gültige Strategien", "Wert": names_of(a.valid)},
        {"Größe": "Kleinste Streuung unter den gültigen", "Wert": a.best or "—"},
        {"Größe": "F der ersten Stufe (Anstoß; unter 10 schwach)", "Wert": de(a.first_stage_f, 1)},
        {"Größe": "Hausman-Statistik Backdoor {Z, W} gegen 2SLS (über 1,96: verworfen)", "Wert": f"{de(a.hausman_z, 2)} ({'verworfen' if a.hausman_rejected else 'nicht verworfen'})"},
        {"Größe": "Beobachtungen, Seed", "Wert": f"{s.n}, {s.seed}"},
    ])


# ----------------------------------------------------------------------------- Experimente

def build_limits_table(res: dict) -> pd.DataFrame:
    rows = []
    for name, e in res.items():
        row = {"Szenario": name, "Wahrheit": de(e["truth"], 2)}
        for m in K.STRATEGIES:
            b = e["bias"][m]
            row[m] = ("✓ " if e["valid"][m] else "✗ ") + (signed(b, 3) if not math.isnan(b) else "—")
        rows.append(row)
    return pd.DataFrame(rows)


def style_limits_table(df: pd.DataFrame):
    def cell(v):
        v = str(v)
        return "background-color: rgba(47,107,101,0.20)" if v.startswith("✓") else "background-color: rgba(192,57,43,0.15)" if v.startswith("✗") else ""

    return df.style.map(cell, subset=list(K.STRATEGIES))


def build_curves_chart(res: dict) -> go.Figure:
    fig = go.Figure()
    for m in K.STRATEGIES:
        ys = res["curves"][m]
        if all(math.isfinite(y) for y in ys):
            fig.add_trace(go.Scatter(x=list(res["ns"]), y=ys, mode="lines+markers", name=m, line=dict(color=K.STRATEGY_COLORS[m], width=3)))
    fig.update_xaxes(title="Beobachtungen n", type="log")
    fig.update_yaxes(title="asymptotischer RMSE (Bias² + Streuung² / n)", range=[0, 0.7])
    return _fixed(fig, 360)


def build_crossing_table(rows: list) -> pd.DataFrame:
    def n_text(x):
        return "—" if not math.isfinite(x) else f"{int(round(x)):,}".replace(",", " ")

    return pd.DataFrame([{"Stärke s (a = b)": de(r["s"], 1), "Bias Backdoor {Z, W}": signed(r["bias"], 3), "Streuung je √n: Backdoor": de(r["sd_bd"], 2), "Frontdoor": de(r["sd_fd"], 2), "2SLS": de(r["sd_iv"], 2) if math.isfinite(r["sd_iv"]) else "—",
                          "Kreuzung n* gegen Frontdoor": n_text(r["n_fd"]), "Kreuzung n* gegen 2SLS": n_text(r["n_iv"])} for r in rows])


def build_mc_chart(res: dict, scenario: str, measure: str) -> go.Figure:
    entry = res["scenarios"][scenario]
    key = E.MEASURE_KEYS[measure]
    vals = [abs(entry["mc"][m]["bias"]) if key == "bias" else entry["mc"][m][key] for m in K.STRATEGIES]
    fig = go.Figure()
    fig.add_trace(go.Bar(x=list(K.STRATEGIES), y=vals, marker_color=[K.COLOR_GOOD if entry["valid"][m] else K.COLOR_BAD for m in K.STRATEGIES], text=[de(v, 3) if key != "cover" else pct(v) for v in vals], textposition="outside"))
    if key == "cover":
        fig.add_hline(y=E.COVER_OK, line=dict(color="#888888", width=1, dash="dot"), annotation_text="93 %", annotation_position="right")
        fig.update_yaxes(range=[0, 1.12])
    elif key in ("rmse", "width"):
        fig.update_yaxes(type="log")
    fig.update_yaxes(title=f"{measure} (grün: Annahme gilt, rot: gilt nicht)")
    return _fixed(fig, 340, showlegend=False)


def build_winner_table(res: dict, measure: str) -> pd.DataFrame:
    rows = []
    for name, entry in res["scenarios"].items():
        rows.append({"Szenario": name, "gültig": names_of([m for m in K.STRATEGIES if entry["valid"][m]]), "Sieger (alle sechs)": names_of(E.winners(entry, measure)), "Sieger unter den gültigen": names_of(E.winners(entry, measure, True))})
    return pd.DataFrame(rows)


def build_mc_table(res: dict, scenario: str) -> pd.DataFrame:
    entry = res["scenarios"][scenario]
    return pd.DataFrame([{"Strategie": m, "gültig": "ja" if entry["valid"][m] else "nein", "Bias": signed(entry["mc"][m]["bias"], 3), "Streuung": de(entry["mc"][m]["sd"], 3), "RMSE": de(entry["mc"][m]["rmse"], 3),
                          "Median-Fehler": de(entry["mc"][m]["median_error"], 3), "Deckung": pct(entry["mc"][m]["cover"]), "Breite (Median)": de(entry["mc"][m]["width"], 3)} for m in K.STRATEGIES])


def build_vortest_chart(res: dict) -> go.Figure:
    n = max(res["ns"])
    names = list(K.VORTEST_SCENARIOS)
    fig = go.Figure()
    fig.add_trace(go.Bar(x=names, y=[100 * res["cells"][(nm, n)]["reject"] for nm in names], marker_color=K.COLOR_CONTROL, name="Vortest verwirft (%)"))
    fig.add_trace(go.Bar(x=names, y=[100 * res["cells"][(nm, n)]["cover"] for nm in names], marker_color=K.COLOR_GOOD, name="Deckung der Vortest-Strategie (%)"))
    fig.update_layout(barmode="group")
    fig.update_yaxes(title="Prozent", range=[0, 104])
    return _fixed(fig, 380)


def build_vortest_table(res: dict) -> pd.DataFrame:
    rows = []
    for n in res["ns"]:
        for nm in K.VORTEST_SCENARIOS:
            c = res["cells"][(nm, n)]
            rows.append({"n": n, "Szenario": nm, "Vortest verwirft": pct(c["reject"], 1), "mittleres F": de(c["f"], 1), "Vortest-Strategie: RMSE": de(c["rmse"], 3), "Deckung": pct(c["cover"]), "Backdoor allein: RMSE": de(c["rmse_bd"], 3),
                         "2SLS allein: Median-Fehler": de(c["median_error_iv"], 3)})
    return pd.DataFrame(rows)


def build_regret_tables(res: dict) -> list:
    rows = pd.DataFrame([{"Strategie": m, "größte Reue": "unendlich" if math.isinf(v["max"]) else de(v["max"], 3), "mittlere Reue": "unendlich" if math.isinf(v["mean"]) else de(v["mean"], 3), "Szenarien mit Reue < 0,05": f"{v['below']} von {v['count']}"}
                         for m, v in res["rows"].items()])
    detail = pd.DataFrame([{"Szenario": nm, **{m: ("∞" if math.isinf(r[m]) else de(r[m], 3)) for m in K.STRATEGIES}} for nm, r in res["table"].items()])
    return [rows, detail]
