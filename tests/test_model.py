import numpy as np
import pandas as pd

from popncde import PopNCDE, PopNCDEConfig, StudyData


def training_frames():
    patient_ids = [f"P{index}" for index in range(8)]
    patients = pd.DataFrame(
        {
            "patient_id": patient_ids,
            "age_y": np.linspace(40.0, 70.0, 8),
            "sex_male": [0, 1, 0, 1, 0, 1, 0, 1],
            "weight_kg": np.linspace(60.0, 90.0, 8),
            "egfr_ml_min": np.linspace(100.0, 65.0, 8),
        }
    )
    doses = pd.DataFrame(
        {
            "patient_id": patient_ids,
            "time_h": np.zeros(8),
            "dose_mg": np.full(8, 100.0),
        }
    )
    rows = []
    for index, patient_id in enumerate(patient_ids):
        for time, value in zip([0.5, 1.0, 1.5, 2.0], [0.8, 1.2, 1.0, 0.7], strict=True):
            rows.append(
                {
                    "patient_id": patient_id,
                    "time_h": time,
                    "concentration": value * (1.0 + index / 20.0),
                }
            )
    return patients, doses, pd.DataFrame(rows)


def test_fit_predict_save_load(tmp_path):
    patients, doses, concentrations = training_frames()
    config = PopNCDEConfig(
        seed=9,
        validation_fraction=0.25,
        pk_epochs=1,
        pk_patience=1,
        ncde_epochs=1,
        ncde_patience=1,
        batch_size=4,
        map_steps=1,
    )
    model = PopNCDE(config).fit(StudyData.from_frames(patients, doses, concentrations))
    new_patient = patients.iloc[[0]].copy()
    new_patient["patient_id"] = "NEW"
    new_doses = pd.DataFrame({"patient_id": ["NEW"], "time_h": [0.0], "dose_mg": [100.0]})
    new_concentrations = pd.DataFrame(
        {
            "patient_id": ["NEW", "NEW", "NEW"],
            "time_h": [0.5, 1.0, 1.5],
            "concentration": [0.8, 1.2, 1.0],
        }
    )
    times = np.arange(0.0, 2.01, 0.25)
    prediction = model.predict(new_patient, new_doses, new_concentrations, times, 3)
    assert len(prediction.frame) == len(times)
    assert np.isfinite(prediction.frame["prediction"]).all()
    target = tmp_path / "model.pt"
    model.save(target)
    loaded = PopNCDE.load(target)
    repeated = loaded.predict(new_patient, new_doses, new_concentrations, times, 3)
    np.testing.assert_allclose(
        prediction.frame["prediction"], repeated.frame["prediction"], rtol=1e-6, atol=1e-6
    )
