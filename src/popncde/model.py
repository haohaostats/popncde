from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .config import PopNCDEConfig
from .core import PopulationNCDERefinedEffects, PopulationPK
from .data import StudyData
from .prediction import Prediction
from .training import (
    estimate_effects,
    fit_neural_model,
    fit_population_pk,
    prepare_priors,
    set_seed,
)


class PopNCDE:
    def __init__(self, config: PopNCDEConfig | None = None) -> None:
        self.config = config or PopNCDEConfig()
        self.population_model: PopulationPK | None = None
        self.neural_model: PopulationNCDERefinedEffects | None = None
        self.covariate_mean: np.ndarray | None = None
        self.covariate_sd: np.ndarray | None = None
        self.population_history = pd.DataFrame()
        self.training_history = pd.DataFrame()

    @property
    def fitted(self) -> bool:
        return self.population_model is not None and self.neural_model is not None

    def fit(self, data: StudyData) -> PopNCDE:
        set_seed(self.config.seed)
        prepared = data.with_validation_split(self.config.validation_fraction, self.config.seed)
        counts = prepared.concentrations.groupby("patient_id").size()
        training_ids = set(
            prepared.patients.loc[prepared.patients["split"] == "train", "patient_id"]
        )
        if any(counts.get(patient_id, 0) < 4 for patient_id in training_ids):
            raise ValueError("each training patient requires at least four concentrations")
        tensors = prepared.to_tensors(self.config.grid_step_h, self.config.device)
        self.covariate_mean = tensors.covariate_mean.copy()
        self.covariate_sd = tensors.covariate_sd.copy()
        population, population_history = fit_population_pk(tensors, self.config)
        priors, effects = prepare_priors(population, tensors, self.config)
        neural, training_history = fit_neural_model(
            tensors, population, priors, effects, self.config
        )
        self.population_model = population
        self.neural_model = neural
        self.population_history = population_history
        self.training_history = training_history
        return self

    def predict(
        self,
        patients: pd.DataFrame,
        doses: pd.DataFrame,
        concentrations: pd.DataFrame,
        times,
        history_count: int = 3,
    ) -> Prediction:
        if not self.fitted:
            raise RuntimeError("model must be fitted or loaded before prediction")
        if len(patients) != 1:
            raise ValueError("predict currently accepts one patient")
        if history_count not in (0, 1, 2, 3):
            raise ValueError("history_count must be 0, 1, 2, or 3")
        requested = np.asarray(times, dtype=float)
        if requested.ndim != 1 or not len(requested) or (requested < 0).any():
            raise ValueError("times must be a nonempty one-dimensional nonnegative sequence")
        horizon = float(requested.max())
        study = StudyData.from_frames(patients, doses, concentrations, horizon_h=horizon)
        tensors = study.to_tensors(
            self.config.grid_step_h,
            self.config.device,
            self.covariate_mean,
            self.covariate_sd,
        )
        available = int(tensors.observation_mask[0].sum())
        count = min(history_count, available)
        rank = torch.arange(tensors.observation_mask.shape[1], device=tensors.times.device)
        context = tensors.observation_mask & (rank.unsqueeze(0) < count)
        indices = torch.tensor([0], dtype=torch.long, device=tensors.times.device)
        eta = estimate_effects(
            self.population_model,
            tensors,
            indices,
            context,
            self.config.effect_penalties[count],
            self.config,
        )
        with torch.no_grad():
            initial_prior, _ = self.population_model(
                tensors.covariates, tensors.doses, tensors.times, eta
            )
            zeros = torch.zeros_like(eta)
            population, _ = self.population_model(
                tensors.covariates, tensors.doses, tensors.times, zeros
            )
            prediction, _, _, refined_prior = self.neural_model(
                tensors.covariates,
                eta,
                tensors.doses,
                tensors.times,
                initial_prior,
                tensors.observation_indices,
                tensors.observations,
                context,
                return_details=True,
            )
        grid = tensors.times.detach().cpu().numpy()
        frame = pd.DataFrame(
            {
                "time_h": requested,
                "population_pk": np.interp(requested, grid, population[0].cpu().numpy()),
                "individualized_prior": np.interp(requested, grid, refined_prior[0].cpu().numpy()),
                "prediction": np.interp(requested, grid, prediction[0].cpu().numpy()),
            }
        )
        ordered = study.concentrations.sort_values("time_h").reset_index(drop=True)
        landmark = float(ordered.iloc[count - 1]["time_h"]) if count else 0.0
        return Prediction(frame, study.doses.copy(), ordered, count, landmark)

    def save(self, path) -> None:
        if not self.fitted:
            raise RuntimeError("model must be fitted before saving")
        payload = {
            "version": "0.1.1",
            "config": asdict(self.config),
            "covariate_mean": self.covariate_mean,
            "covariate_sd": self.covariate_sd,
            "population_state": self.population_model.state_dict(),
            "neural_state": self.neural_model.state_dict(),
            "population_history": self.population_history.to_dict(orient="list"),
            "training_history": self.training_history.to_dict(orient="list"),
        }
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        torch.save(payload, target)

    @classmethod
    def load(cls, path, device: str | None = None) -> PopNCDE:
        payload = torch.load(path, map_location=device or "cpu", weights_only=False)
        config_values = payload["config"]
        if device is not None:
            config_values["device"] = device
        config_values["effect_penalties"] = tuple(config_values["effect_penalties"])
        instance = cls(PopNCDEConfig(**config_values))
        instance.population_model = PopulationPK().to(instance.config.device)
        instance.population_model.load_state_dict(payload["population_state"])
        instance.population_model.eval()
        instance.neural_model = PopulationNCDERefinedEffects().to(instance.config.device)
        instance.neural_model.load_state_dict(payload["neural_state"])
        instance.neural_model.eval()
        instance.covariate_mean = np.asarray(payload["covariate_mean"], dtype=np.float32)
        instance.covariate_sd = np.asarray(payload["covariate_sd"], dtype=np.float32)
        instance.population_history = pd.DataFrame(payload["population_history"])
        instance.training_history = pd.DataFrame(payload["training_history"])
        return instance
