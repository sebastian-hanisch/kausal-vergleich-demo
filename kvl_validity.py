"""Gültigkeit der sechs Strategien am Diagramm der Welt (mit der verborgenen Ursache als Knoten): Backdoor-Kriterium, Frontdoor-Kriterium, Instrument-Kriterium. Kein Streamlit."""

from __future__ import annotations

import bdr_backdoor as B
import frd_criterion as F
import ins_criterion as N
import kvl_constants as K
import kvl_world as W

ADJUST = {"naiv": (), "Backdoor {Z}": (K.Z,), "Backdoor {Z, W}": (K.Z, K.W), "alles anpassen": (K.Z, K.W, K.I, K.M, K.C)}


def strategy_valid(k: dict, strategy: str) -> bool:
    """Gilt die Annahme der Strategie im Diagramm der Welt? Backdoor: die Menge erfüllt das Backdoor-Kriterium; Frontdoor: der Bestand erfüllt das Frontdoor-Kriterium; 2SLS: der Anstoß erfüllt das Instrument-Kriterium."""
    parents = W.parents_of(k)
    if strategy in ADJUST:
        return B.backdoor_valid(parents, K.X, K.Y, ADJUST[strategy])
    if strategy == "Frontdoor":
        return F.frontdoor_check(parents, K.X, K.Y, (K.M,))["valid"]
    if strategy == "2SLS":
        return N.instrument_check(parents, K.X, K.Y, K.I)["valid"]
    raise ValueError(strategy)


def validity(k: dict) -> dict:
    return {s: strategy_valid(k, s) for s in K.STRATEGIES}


def reason(k: dict, strategy: str) -> str:
    """Warum die Annahme in der Welt nicht gilt (leer, wenn sie gilt)."""
    parents = W.parents_of(k)
    if strategy_valid(k, strategy):
        return ""
    if strategy in ADJUST:
        r = B.backdoor_check(parents, K.X, K.Y, ADJUST[strategy])
        if r.descendants_in_z:
            return "enthält Nachkommen des Faktors (" + ", ".join(K.NAMES[v] for v in sorted(r.descendants_in_z)) + ")"
        return "offener Rückwärtspfad " + " ".join(_path(p) for p in r.open_paths[:1])
    if strategy == "Frontdoor":
        c = F.frontdoor_check(parents, K.X, K.Y, (K.M,))
        if not c["intercepts"]:
            return "der Bestand fängt nicht alle Wege ab (direkter Pfeil)"
        if not c["no_backdoor"]:
            return "offener Rückweg vom Faktor zum Bestand (verborgene Ursache wirkt auf ihn)"
        return "Rückweg vom Bestand zum Servicegrad nicht durch den Faktor blockiert"
    c = N.instrument_check(parents, K.X, K.Y, K.I)
    if not c["relevant"]:
        return "kein Anstoß (nicht relevant)"
    return "der Anstoß wirkt am Faktor vorbei oder hängt an einer gemeinsamen Ursache"


def _path(p) -> str:
    return K.NAMES[p.nodes[0]] + "".join(f" {a} {K.NAMES[v]}" for a, v in zip(p.arrows, p.nodes[1:]))
