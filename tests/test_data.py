import pandas as pd
import pytest

from popncde import StudyData


def frames():
    patients = pd.DataFrame(
        {
            "patient_id": ["P1", "P2"],
            "age_y": [50.0, 60.0],
            "sex_male": [0, 1],
            "weight_kg": [65.0, 80.0],
            "egfr_ml_min": [90.0, 70.0],
        }
    )
    doses = pd.DataFrame(
        {"patient_id": ["P1", "P2"], "time_h": [0.0, 0.0], "dose_mg": [100.0, 100.0]}
    )
    concentrations = pd.DataFrame(
        {
            "patient_id": ["P1", "P2"],
            "time_h": [1.0, 1.0],
            "concentration": [1.0, 1.2],
        }
    )
    return patients, doses, concentrations


def test_data_builds_tensors():
    study = StudyData.from_frames(*frames()).with_validation_split(0.5, 7)
    tensors = study.to_tensors(0.25, "cpu")
    assert tensors.covariates.shape == (2, 4)
    assert tensors.doses.shape[1] == 5
    assert int(tensors.observation_mask.sum()) == 2


def test_unknown_patient_is_rejected():
    patients, doses, concentrations = frames()
    doses.loc[0, "patient_id"] = "missing"
    with pytest.raises(ValueError):
        StudyData.from_frames(patients, doses, concentrations)
