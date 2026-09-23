from __future__ import annotations

from pathlib import Path

import numpy as np

from .config import PopNCDEConfig
from .data import StudyData
from .examples import load_example_data, load_example_patient
from .model import PopNCDE
from .plotting import save_figure


def main(output_dir="."):
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    patients, doses, concentrations = load_example_data()
    study = StudyData.from_frames(patients, doses, concentrations)
    config = PopNCDEConfig(
        seed=7321,
        validation_fraction=0.25,
        pk_epochs=2,
        pk_patience=2,
        ncde_epochs=2,
        ncde_patience=2,
        batch_size=4,
        map_steps=5,
    )
    model = PopNCDE(config).fit(study)
    patient, known_doses, previous_concentrations = load_example_patient()
    prediction = model.predict(
        patients=patient,
        doses=known_doses,
        concentrations=previous_concentrations,
        times=np.arange(0.0, 36.01, 0.25),
        history_count=3,
    )
    model_path = output / "popncde_demo_model.pt"
    figure_path = output / "popncde_demo_prediction.svg"
    model.save(model_path)
    figure, _ = prediction.plot()
    save_figure(figure, figure_path)
    endpoints = prediction.future_endpoints()
    print("Pop-NCDE demo completed")
    print(f"Model: {model_path.resolve()}")
    print(f"Figure: {figure_path.resolve()}")
    print(f"Future endpoints: {endpoints}")


if __name__ == "__main__":
    main()
