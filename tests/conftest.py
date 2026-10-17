"""Gemeinsame Fixtures: die Experimente einmal klein (für Struktur- und Verdikt-Tests) und einmal in voller Größe (nur für test_claims). Die Experimente 1, 2 und 5 sind exakt und billig."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


@pytest.fixture(scope="session")
def limits():
    import kvl_evaluation as ev

    return ev.limits_experiment()


@pytest.fixture(scope="session")
def small_mc():
    import kvl_evaluation as ev

    return ev.mc_experiment(n=300, reps=6)


@pytest.fixture(scope="session")
def small_vortest():
    import kvl_evaluation as ev

    return ev.vortest_experiment(ns=(200,), reps=20)


@pytest.fixture(scope="session")
def full_mc():
    import kvl_evaluation as ev

    return ev.mc_experiment()


@pytest.fixture(scope="session")
def full_vortest():
    import kvl_evaluation as ev

    return ev.vortest_experiment()
