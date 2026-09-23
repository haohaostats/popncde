from __future__ import annotations

from importlib.resources import files

import pandas as pd


def _read(name):
    resource = files("popncde").joinpath("data", name)
    with resource.open("r", encoding="utf-8") as stream:
        return pd.read_csv(stream)


def load_example_data():
    return _read("patients.csv"), _read("doses.csv"), _read("concentrations.csv")


def load_example_patient():
    return (
        _read("new_patient.csv"),
        _read("known_doses.csv"),
        _read("previous_concentrations.csv"),
    )
