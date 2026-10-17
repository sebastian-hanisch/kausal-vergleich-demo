"""Die Welt mit acht Größen als lineares Gauß-Modell: Koeffizienten, Bevölkerungskovarianz, Wahrheit (Gesamtwirkung), Diagramm und Daten aus festen Strömen. Kein Streamlit."""

from __future__ import annotations

import numpy as np

import kvl_constants as K
from kausal_generator import SplitMix64

N = 8


def edge_weights(k: dict) -> dict:
    """{(von, nach): Stärke} aller Pfeile mit Stärke ungleich null."""
    w = dict(K.FIXED)
    w.update({(K.U, K.X): k["a"], (K.U, K.Y): k["b"], (K.I, K.X): k["pi"], (K.X, K.M): k["c"], (K.M, K.Y): K.D_EFFECT, (K.X, K.Y): k["e"], (K.U, K.M): k["f"], (K.I, K.Y): k["g"], (K.U, K.I): k["h"]})
    return {e: float(s) for e, s in w.items() if s != 0}


def coefficient_matrix(k: dict) -> np.ndarray:
    """B[j, i]: Koeffizient von i in j."""
    b = np.zeros((N, N))
    for (i, j), s in edge_weights(k).items():
        b[j, i] = s
    return b


def truth(k: dict) -> float:
    """Gesamtwirkung des Faktors auf den Servicegrad bei Eingriff (do): Pfadprodukte c · 0,6 + e."""
    return k["c"] * K.D_EFFECT + k["e"]


def covariance(b: np.ndarray) -> np.ndarray:
    inv = np.linalg.inv(np.eye(N) - b)
    return inv @ inv.T


def parents_of(k: dict) -> tuple:
    """Diagramm (mit der verborgenen Ursache als Knoten) als Tupel von Elternmengen."""
    es = edge_weights(k)
    return tuple(tuple(sorted(i for (i, j) in es if j == c)) for c in range(N))


def simulate(k: dict, n: int, rng: SplitMix64) -> np.ndarray:
    """n × 8 (Z, W, I, U, X, M, Y, C), Einheitsrauschen: x = (I − B)⁻¹ e."""
    e = rng.normal(n * N).reshape(n, N)
    return e @ np.linalg.inv(np.eye(N) - coefficient_matrix(k)).T
