import numpy as np

from popncde import (
    PopNCDE,
    PopNCDEConfig,
    StudyData,
    load_example_data,
    load_example_patient,
)

patients, doses, concentrations = load_example_data()
new_patient, known_doses, previous_concentrations = load_example_patient()
prediction_times = np.arange(0.0, 36.01, 0.25)

study = StudyData.from_frames(patients, doses, concentrations)
model = PopNCDE(PopNCDEConfig())
model.fit(study)

prediction = model.predict(
    patients=new_patient,
    doses=known_doses,
    concentrations=previous_concentrations,
    times=prediction_times,
    history_count=3,
)

figure, _ = prediction.plot()
endpoints = prediction.future_endpoints()
model.save("popncde.pt")
figure.savefig("popncde_prediction.svg", bbox_inches="tight")

print(prediction.frame)
print(endpoints)
