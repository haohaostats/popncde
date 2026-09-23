from popncde.examples import load_example_data, load_example_patient


def test_example_data_are_complete():
    patients, _, concentrations = load_example_data()
    patient, patient_doses, history = load_example_patient()
    assert len(patients) == 8
    assert concentrations.groupby("patient_id").size().min() == 4
    assert len(patient) == 1
    assert len(patient_doses) == 3
    assert len(history) == 3
