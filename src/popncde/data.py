from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch

PATIENT_COLUMNS = ("patient_id", "age_y", "sex_male", "weight_kg", "egfr_ml_min")
DOSE_COLUMNS = ("patient_id", "time_h", "dose_mg")
CONCENTRATION_COLUMNS = ("patient_id", "time_h", "concentration")


@dataclass
class TensorData:
    patient_ids: list[str]
    splits: np.ndarray
    covariates: torch.Tensor
    doses: torch.Tensor
    observation_indices: torch.Tensor
    observations: torch.Tensor
    observation_mask: torch.Tensor
    times: torch.Tensor
    covariate_mean: np.ndarray
    covariate_sd: np.ndarray

    def indices(self, split: str) -> np.ndarray:
        return np.flatnonzero(self.splits == split)


@dataclass
class StudyData:
    patients: pd.DataFrame
    doses: pd.DataFrame
    concentrations: pd.DataFrame
    horizon_h: float | None = None

    @classmethod
    def from_frames(
        cls,
        patients: pd.DataFrame,
        doses: pd.DataFrame,
        concentrations: pd.DataFrame,
        horizon_h: float | None = None,
    ) -> StudyData:
        patient_frame = patients.copy()
        dose_frame = doses.copy()
        concentration_frame = concentrations.copy()
        if (
            "concentration_observed" in concentration_frame
            and "concentration" not in concentration_frame
        ):
            concentration_frame = concentration_frame.rename(
                columns={"concentration_observed": "concentration"}
            )
        cls._require(patient_frame, PATIENT_COLUMNS, "patients")
        cls._require(dose_frame, DOSE_COLUMNS, "doses")
        cls._require(concentration_frame, CONCENTRATION_COLUMNS, "concentrations")
        patient_frame["patient_id"] = patient_frame["patient_id"].astype(str)
        dose_frame["patient_id"] = dose_frame["patient_id"].astype(str)
        concentration_frame["patient_id"] = concentration_frame["patient_id"].astype(str)
        if patient_frame["patient_id"].duplicated().any():
            raise ValueError("patients must contain one row per patient_id")
        ids = set(patient_frame["patient_id"])
        if not set(dose_frame["patient_id"]).issubset(ids):
            raise ValueError("doses contain unknown patient_id values")
        if not set(concentration_frame["patient_id"]).issubset(ids):
            raise ValueError("concentrations contain unknown patient_id values")
        numeric = ["age_y", "sex_male", "weight_kg", "egfr_ml_min"]
        if patient_frame[numeric].isna().any().any():
            raise ValueError("patient covariates cannot contain missing values")
        if (dose_frame["time_h"] < 0).any() or (dose_frame["dose_mg"] <= 0).any():
            raise ValueError("dose times must be nonnegative and doses must be positive")
        if (concentration_frame["time_h"] < 0).any() or (
            concentration_frame["concentration"] < 0
        ).any():
            raise ValueError("concentration times and values must be nonnegative")
        dose_frame = dose_frame.sort_values(["patient_id", "time_h"]).reset_index(drop=True)
        concentration_frame = concentration_frame.sort_values(["patient_id", "time_h"]).reset_index(
            drop=True
        )
        return cls(patient_frame.reset_index(drop=True), dose_frame, concentration_frame, horizon_h)

    @staticmethod
    def _require(frame: pd.DataFrame, columns: tuple[str, ...], name: str) -> None:
        missing = [column for column in columns if column not in frame]
        if missing:
            raise ValueError(f"{name} missing required columns: {missing}")

    def with_validation_split(self, fraction: float, seed: int) -> StudyData:
        patients = self.patients.copy()
        if "split" in patients and {"train", "validation"}.issubset(set(patients["split"])):
            return StudyData(
                patients, self.doses.copy(), self.concentrations.copy(), self.horizon_h
            )
        rng = np.random.default_rng(seed)
        order = rng.permutation(len(patients))
        validation_n = max(1, round(len(patients) * fraction))
        split = np.full(len(patients), "train", dtype=object)
        split[order[:validation_n]] = "validation"
        patients["split"] = split
        return StudyData(patients, self.doses.copy(), self.concentrations.copy(), self.horizon_h)

    def to_tensors(
        self,
        grid_step_h: float,
        device: str,
        covariate_mean: np.ndarray | None = None,
        covariate_sd: np.ndarray | None = None,
    ) -> TensorData:
        patients = self.patients.sort_values("patient_id").reset_index(drop=True)
        patient_ids = patients["patient_id"].tolist()
        patient_lookup = {patient_id: index for index, patient_id in enumerate(patient_ids)}
        maximum = 0.0
        if len(self.doses):
            maximum = max(maximum, float(self.doses["time_h"].max()))
        if len(self.concentrations):
            maximum = max(maximum, float(self.concentrations["time_h"].max()))
        if self.horizon_h is not None:
            maximum = max(maximum, float(self.horizon_h))
        steps = int(np.ceil(maximum / grid_step_h))
        times = np.arange(steps + 1, dtype=np.float32) * np.float32(grid_step_h)
        covariate_names = ["age_y", "sex_male", "weight_kg", "egfr_ml_min"]
        values = patients[covariate_names].to_numpy(dtype=np.float32)
        splits = patients.get("split", pd.Series("train", index=patients.index)).to_numpy()
        if covariate_mean is None:
            train = splits == "train"
            covariate_mean = values[train].mean(axis=0)
        if covariate_sd is None:
            train = splits == "train"
            covariate_sd = values[train].std(axis=0, ddof=1 if int(train.sum()) > 1 else 0)
        covariate_mean = np.asarray(covariate_mean, dtype=np.float32)
        covariate_sd = np.asarray(covariate_sd, dtype=np.float32)
        covariate_sd = np.where(np.isfinite(covariate_sd) & (covariate_sd > 0), covariate_sd, 1.0)
        values = (values - covariate_mean) / covariate_sd
        dose_array = np.zeros((len(patients), len(times)), dtype=np.float32)
        for row in self.doses.itertuples(index=False):
            index = round(float(row.time_h) / grid_step_h)
            if index < len(times):
                dose_array[patient_lookup[row.patient_id], index] += float(row.dose_mg) / 100.0
        groups = {
            patient_id: group.sort_values("time_h")
            for patient_id, group in self.concentrations.groupby("patient_id")
        }
        maximum_observations = max([len(group) for group in groups.values()] + [1])
        observation_indices = np.zeros((len(patients), maximum_observations), dtype=np.int64)
        observation_values = np.zeros((len(patients), maximum_observations), dtype=np.float32)
        observation_mask = np.zeros((len(patients), maximum_observations), dtype=bool)
        for patient_id, group in groups.items():
            patient_index = patient_lookup[patient_id]
            count = len(group)
            indices = np.rint(group["time_h"].to_numpy() / grid_step_h).astype(int)
            if (indices >= len(times)).any():
                raise ValueError("observation time exceeds the prediction horizon")
            observation_indices[patient_index, :count] = indices
            observation_values[patient_index, :count] = group["concentration"].to_numpy(
                dtype=np.float32
            )
            observation_mask[patient_index, :count] = True
        return TensorData(
            patient_ids=patient_ids,
            splits=splits,
            covariates=torch.tensor(values, dtype=torch.float32, device=device),
            doses=torch.tensor(dose_array, dtype=torch.float32, device=device),
            observation_indices=torch.tensor(observation_indices, dtype=torch.long, device=device),
            observations=torch.tensor(observation_values, dtype=torch.float32, device=device),
            observation_mask=torch.tensor(observation_mask, dtype=torch.bool, device=device),
            times=torch.tensor(times, dtype=torch.float32, device=device),
            covariate_mean=covariate_mean,
            covariate_sd=covariate_sd,
        )
