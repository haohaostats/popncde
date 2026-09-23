# popncde

`popncde` implements population neural controlled differential equations for individualized pharmacokinetic prediction from baseline covariates, known dosing history, and sparse concentration measurements.

## Installation

```bash
pip install git+https://github.com/haohaostats/popncde.git
```

## Data format

The package accepts three pandas data frames.

```text
patients:       patient_id, age_y, sex_male, weight_kg, egfr_ml_min
doses:          patient_id, time_h, dose_mg
concentrations: patient_id, time_h, concentration
```

Training data may contain a `split` column with `train` and `validation` values. If it is absent, a patient-level validation split is created.

## Usage

```python
from popncde import PopNCDE, PopNCDEConfig, StudyData

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

prediction.plot()
prediction.future_endpoints()
model.save("popncde.pt")
```

## Scope

The package contains the Pop-NCDE method and its population pharmacokinetic backbone. It does not contain benchmark models, manuscript figure reproduction, or repeated simulation workflows.

