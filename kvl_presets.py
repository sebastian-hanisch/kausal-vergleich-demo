"""Permalink (Begrenzen), Presets und Zufalls-Seed-Button (Standardmuster aus dem Demo-Portfolio). Alle Regler sind in jeder Einstellung wirksam und werden immer gezeichnet;
es gibt weder KEPT-Regler noch berechnete Grenzen."""

import random
from dataclasses import dataclass

import streamlit as st

import kvl_constants as K


@dataclass(frozen=True)
class SettingSpec:
    url_param: str
    caster: type
    default: object
    lo: float
    hi: float
    step: float


SETTING_SPECS = {f"{k}_slider": SettingSpec(k, float, v[0], v[1], v[2], v[3]) for k, v in K.KNOBS.items()}
SETTING_SPECS["n_slider"] = SettingSpec("n", int, K.DEFAULT_N, K.N_MIN, K.N_MAX, K.N_STEP)
SETTING_SPECS["seed_input"] = SettingSpec("seed", int, K.DEFAULT_SEED, 0, K.SEED_MAX, 1)
PRESET_KEYS = {**{k: f"{k}_slider" for k in K.KNOBS}, "n": "n_slider", "seed": "seed_input"}


def snap(value, spec: SettingSpec):
    """Auf das Raster des Reglers legen und in die Grenzen klemmen."""
    value = max(spec.lo, min(spec.hi, value))
    if spec.step > 0 and (spec.caster is float or spec.step > 1):
        value = round((value - spec.lo) / spec.step) * spec.step + spec.lo
        value = round(value, 6) if spec.caster is float else int(value)
    return value


def init_session_state_defaults():
    for state_key, spec in SETTING_SPECS.items():
        if state_key not in st.session_state:
            st.session_state[state_key] = spec.default


def bounds(state_key):
    spec = SETTING_SPECS[state_key]
    return spec.lo, spec.hi


def load_permalink_settings():
    """Einmal je Sitzung: Werte aus der Adresszeile übernehmen, auf die Reglergrenzen begrenzt."""
    if "permalink_loaded" in st.session_state:
        return
    init_session_state_defaults()
    qp = st.query_params
    for state_key, spec in SETTING_SPECS.items():
        if spec.url_param not in qp:
            continue
        try:
            value = snap(spec.caster(float(qp[spec.url_param])), spec)
        except (ValueError, TypeError):
            continue
        st.session_state[state_key] = value
    st.session_state["permalink_loaded"] = True


def sync_query_params(values: dict):
    """`values`: {state_key: aktueller Wert}; die Adresszeile spiegelt die Konfiguration."""
    try:
        for state_key, value in values.items():
            st.query_params[SETTING_SPECS[state_key].url_param] = str(value)
    except Exception:
        pass


def apply_preset(name):
    p = K.PRESETS[name]
    for k in K.KNOBS:
        st.session_state[f"{k}_slider"] = p["knobs"][k]
    st.session_state["n_slider"] = p["n"]
    st.session_state["seed_input"] = p["seed"]


def randomize_seed():
    st.session_state["seed_input"] = random.randint(0, K.SEED_MAX)
