# popncde

`popncde` implements population neural controlled differential equations for individualized pharmacokinetic prediction from baseline covariates, known dosing history, and sparse concentration measurements.

## Installation

```bash
pip install git+https://github.com/haohaostats/popncde.git
```

## Run immediately

```bash
popncde-demo
```

This command runs a complete example with data included in the package. It trains a small demonstration model, individualizes a new patient from three previous concentrations, prints future pharmacokinetic endpoints, and creates:

```text
popncde_demo_model.pt
popncde_demo_prediction.svg
```

The demonstration uses a reduced training configuration so that users can verify the installation quickly. It is not intended for substantive analysis.

## Included demo data

The installation includes six inspectable CSV files:

```text
patients.csv
doses.csv
concentrations.csv
new_patient.csv
known_doses.csv
previous_concentrations.csv
```

The complete Python workflow can be run without preparing any data:

```python
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

prediction.plot()
print(prediction.future_endpoints())
model.save("popncde.pt")
```

## Data format

The package accepts three pandas data frames.

```text
patients:       patient_id, age_y, sex_male, weight_kg, egfr_ml_min
doses:          patient_id, time_h, dose_mg
concentrations: patient_id, time_h, concentration
```

Training and validation sets are separated at the patient level. Users may provide this assignment through a `split` column in the patient table. Otherwise, the package randomly assigns 15% of patients to validation using the configured random seed.

## Use with your own data

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
