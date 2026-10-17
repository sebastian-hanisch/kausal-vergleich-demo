"""PDF-Export der aktuellen Auswertung (fpdf2). Kernschriften sind latin-1: Umlaute gehen, Pfeile und Sonderzeichen werden ersetzt."""

from __future__ import annotations

import math

from fpdf import FPDF

from kvl_evaluation import Analysis, de, names_of, signed, verdict_analysis

_REPLACE = {"→": "->", "←": "<-", "–": "-", "—": "-", "−": "-", "„": '"', "“": '"', "×": "x", "·": "*", "**": "", "α": "alpha", "π": "pi", "σ": "sigma", "²": "2", "≤": "<=", "≥": ">=", "±": "+/-", "✓": "ja", "✗": "nein", "…": "...", "√": "sqrt"}


def _s(text) -> str:
    out = str(text)
    for old, new in _REPLACE.items():
        out = out.replace(old, new)
    return out.encode("latin-1", "replace").decode("latin-1")


def build_pdf(a: Analysis) -> bytes:
    s = a.settings
    pdf = FPDF(orientation="P", unit="mm", format="A4")
    pdf.set_auto_page_break(auto=True, margin=12)
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 15)
    pdf.cell(0, 9, _s("Welches Verfahren, gemessen woran?"), new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 10)
    knobs = ", ".join(f"{k} = {de(v, 2)}" for k, v in s.knobs.items())
    pdf.multi_cell(0, 5, _s(f"Welt der Linie mit acht Größen (Z, W, I, U verborgen, X, M, Y, C). Regler: {knobs}. {s.n} Beobachtungen, Seed {s.seed}. Wahrheit (Gesamtwirkung): {de(a.truth, 3)}. Alles simuliert."), new_x="LMARGIN", new_y="NEXT")
    pdf.ln(2)
    pdf.set_font("Helvetica", "B", 10)
    pdf.cell(0, 6, _s("Die sechs Strategien"), new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 9)
    for r in a.rows:
        bias = signed(r.limit_bias, 3) if not math.isnan(r.limit_bias) else "nicht definiert"
        pdf.multi_cell(0, 4.6, _s(f"{r.strategy}: Schätzung {de(r.estimate, 3)}, Intervall {de(r.lo, 2)} bis {de(r.hi, 2)}, deckt: {'ja' if r.covers else 'nein'}; Annahme gilt: {'ja' if r.valid else 'nein'}; Grenzwert-Bias {bias}."), new_x="LMARGIN", new_y="NEXT")
    pdf.ln(2)
    pdf.set_font("Helvetica", "I", 9)
    pdf.multi_cell(0, 4.5, _s(verdict_analysis(a)[1]), new_x="LMARGIN", new_y="NEXT")
    pdf.ln(2)
    pdf.set_font("Helvetica", "", 9)
    pdf.multi_cell(0, 4.6, _s(f"Gültig: {names_of(a.valid)}. F der ersten Stufe {de(a.first_stage_f, 1)}; Hausman-Statistik {de(a.hausman_z, 2)} ({'verworfen' if a.hausman_rejected else 'nicht verworfen'})."), new_x="LMARGIN", new_y="NEXT")
    pdf.ln(2)
    pdf.set_font("Helvetica", "I", 8)
    pdf.multi_cell(0, 4, _s("Quelle: kausal-vergleich-demo, Sebastian Hanisch."), new_x="LMARGIN", new_y="NEXT")
    return bytes(pdf.output())
