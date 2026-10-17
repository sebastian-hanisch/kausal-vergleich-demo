"""Die sechs Schätzstrategien der Linie aus der Kovarianzmatrix der beobachteten Größen (die verborgene Ursache wird nie benutzt), ihre Grenzwerte und Streuungen, die Kreuzungsgröße, der Hausman-Vortest und der Monte-Carlo-Lauf. Kein Streamlit."""

from __future__ import annotations

import math

import numpy as np

import kvl_constants as K
import kvl_world as W
from kausal_generator import SplitMix64

Z, WW, I, X, M, Y, C = K.Z, K.W, K.I, K.X, K.M, K.Y, K.C
HUGE_N = 10.0 ** 12                                                      # „unendlich“: Standardfehler · √n gibt die Streuung je √n


def ols(S: np.ndarray, y: int, xs, n: float) -> tuple:
    """Koeffizienten und Standardfehler der Regression von y auf xs aus der (zentrierten) Kovarianzmatrix S bei n Beobachtungen."""
    xs = list(xs)
    sxx = S[np.ix_(xs, xs)]
    beta = np.linalg.solve(sxx, S[xs, y])
    sig2 = (S[y, y] - S[y, xs] @ beta) * (n - 1) / (n - len(xs) - 1)
    return beta, np.sqrt(np.diag(sig2 * np.linalg.inv(sxx) / (n - 1)))


def estimates(S: np.ndarray, n: float) -> dict:
    """Je Strategie (Schätzung, Standardfehler); nur Zeilen und Spalten der beobachteten Größen werden gelesen."""
    out = {"naiv": tuple(v[0] for v in ols(S, Y, [X], n))}
    for name, adj in (("Backdoor {Z}", [Z]), ("Backdoor {Z, W}", [Z, WW]), ("alles anpassen", [Z, WW, I, M, C])):
        beta, se = ols(S, Y, [X] + adj, n)
        out[name] = (beta[0], se[0])
    bc, sc = ols(S, M, [X], n)
    bd, sd = ols(S, Y, [M, X], n)
    out["Frontdoor"] = (bc[0] * bd[0], math.sqrt(bd[0] ** 2 * sc[0] ** 2 + bc[0] ** 2 * sd[0] ** 2))
    cov = [Z, WW]                                                         # 2SLS mit Kovariaten: Teilkovarianzen
    scc = S[np.ix_(cov, cov)]

    def part(a: int, b: int) -> float:
        return S[a, b] - S[a, cov] @ np.linalg.solve(scc, S[cov, b])

    with np.errstate(divide="ignore", invalid="ignore"):
        iv = part(I, Y) / part(I, X)
        sig2 = part(Y, Y) - 2 * iv * part(Y, X) + iv ** 2 * part(X, X)
        out["2SLS"] = (iv, math.sqrt(abs(sig2) * part(I, I) / ((n - 1) * part(I, X) ** 2)))
    return out


def first_stage_f(S: np.ndarray, n: float) -> float:
    """F der ersten Stufe (X auf I, Z, W): Quadrat des t-Werts von I."""
    beta, se = ols(S, X, [I, Z, WW], n)
    return float((beta[0] / se[0]) ** 2)


def limits(k: dict) -> dict:
    """Grenzwert-Bias (Grenzwert minus Wahrheit) und Streuung je √n jeder Strategie in der Bevölkerung (exakt, nan wo der Grenzwert nicht existiert)."""
    est = estimates(W.covariance(W.coefficient_matrix(k)), HUGE_N)
    tau = W.truth(k)
    return {m: (float(est[m][0] - tau), float(est[m][1] * math.sqrt(HUGE_N))) for m in K.STRATEGIES}


def crossing_n(bias_biased: float, sd_biased: float, sd_valid: float) -> float:
    """Stichprobengröße, ab der die gültige Strategie nach RMSE gewinnt: Bias² + σ_b² / n = σ_g² / n, also n* = (σ_g² − σ_b²) / Bias² (Streuungen je √n)."""
    if abs(bias_biased) < 1e-12 or not math.isfinite(sd_valid):
        return math.inf
    return max((sd_valid ** 2 - sd_biased ** 2) / bias_biased ** 2, 0.0)


def asymptotic_rmse(bias: float, sd_per_sqrt_n: float, n: float) -> float:
    return math.sqrt(bias ** 2 + sd_per_sqrt_n ** 2 / n)


def hausman_z(est: dict) -> float:
    """|Backdoor {Z, W} − 2SLS| / √(Var(2SLS) − Var(Backdoor)): Hausman-Statistik, wenn die Backdoor-Regression unter der Nullhypothese effizient ist."""
    (e1, s1), (e2, s2) = est["Backdoor {Z, W}"], est["2SLS"]
    return abs(e1 - e2) / math.sqrt(max(s2 ** 2 - s1 ** 2, 1e-12))


def monte_carlo(k: dict, n: int, reps: int, seed_base: int) -> dict:
    """Je Strategie über `reps` feste Datensätze: Bias, Streuung, RMSE, Median-Fehler, Deckung, mediane Breite, Median und Quartile der Schätzung."""
    tau = W.truth(k)
    res = {m: [] for m in K.STRATEGIES}
    for r in range(reps):
        S = np.cov(W.simulate(k, n, SplitMix64(seed_base + r)), rowvar=False)
        for m, v in estimates(S, n).items():
            res[m].append(v)
    out = {}
    for m, v in res.items():
        a = np.array(v)
        est, se = a[:, 0], a[:, 1]
        err = np.abs(est - tau)
        out[m] = {"bias": float(est.mean() - tau), "sd": float(est.std()), "rmse": float(np.sqrt(np.mean((est - tau) ** 2))), "median_error": float(np.median(err)), "cover": float(np.mean(err <= 1.96 * se)),
                  "width": float(np.median(2 * 1.96 * se)), "median": float(np.median(est)), "q25": float(np.quantile(est, 0.25)), "q75": float(np.quantile(est, 0.75))}
    return out
