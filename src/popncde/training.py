from __future__ import annotations

import copy
import random

import numpy as np
import pandas as pd
import torch

from .config import PopNCDEConfig
from .core import PopulationNCDERefinedEffects, PopulationPK
from .data import TensorData


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def observation_loss(predictions, study, indices, mask):
    predicted = predictions.gather(1, study.observation_indices[indices])
    return (
        (torch.log1p(predicted[mask]) - torch.log1p(study.observations[indices][mask]))
        .square()
        .mean()
    )


def estimate_effects(model, study, indices, context, penalty, config):
    eta = torch.zeros(len(indices), 5, device=study.times.device, requires_grad=True)
    if not context.any():
        return eta.detach()
    optimizer = torch.optim.Adam([eta], lr=config.map_learning_rate)
    for _ in range(config.map_steps):
        optimizer.zero_grad(set_to_none=True)
        predictions, _ = model(study.covariates[indices], study.doses[indices], study.times, eta)
        at_observations = predictions.gather(1, study.observation_indices[indices])
        data_loss = (
            (
                torch.log1p(at_observations[context])
                - torch.log1p(study.observations[indices][context])
            )
            .square()
            .mean()
        )
        objective = data_loss + penalty * model.iiv_penalty(eta)
        objective.backward()
        optimizer.step()
    return eta.detach()


def fit_population_pk(study: TensorData, config: PopNCDEConfig):
    train = study.indices("train")
    validation = study.indices("validation")
    if not len(train) or not len(validation):
        raise ValueError("training requires train and validation patients")
    train_indices = torch.tensor(train, dtype=torch.long, device=study.times.device)
    validation_indices = torch.tensor(validation, dtype=torch.long, device=study.times.device)
    model = PopulationPK().to(study.times.device)
    eta = torch.nn.Embedding(len(study.patient_ids), 5, device=study.times.device)
    torch.nn.init.zeros_(eta.weight)
    optimizer = torch.optim.Adam(
        [*model.parameters(), *eta.parameters()], lr=config.pk_learning_rate
    )
    best = float("inf")
    best_state = None
    stale = 0
    history = []
    for epoch in range(1, config.pk_epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        predictions, _ = model(
            study.covariates[train_indices],
            study.doses[train_indices],
            study.times,
            eta(train_indices),
        )
        data_loss = observation_loss(
            predictions, study, train_indices, study.observation_mask[train_indices]
        )
        objective = data_loss + 0.02 * model.iiv_penalty(eta(train_indices))
        objective.backward()
        torch.nn.utils.clip_grad_norm_([*model.parameters(), *eta.parameters()], 10.0)
        optimizer.step()
        with torch.no_grad():
            zeros = torch.zeros(len(validation_indices), 5, device=study.times.device)
            values, _ = model(
                study.covariates[validation_indices],
                study.doses[validation_indices],
                study.times,
                zeros,
            )
            validation_loss = observation_loss(
                values,
                study,
                validation_indices,
                study.observation_mask[validation_indices],
            )
        score = float(validation_loss)
        history.append(
            {
                "epoch": epoch,
                "train_log_mse": float(data_loss.detach()),
                "validation_log_mse": score,
            }
        )
        if score < best - 1e-6:
            best = score
            best_state = copy.deepcopy(model.state_dict())
            stale = 0
        else:
            stale += 1
        if stale >= config.pk_patience:
            break
    if best_state is None:
        raise RuntimeError("population PK training produced no checkpoint")
    model.load_state_dict(best_state)
    model.eval()
    return model, pd.DataFrame(history)


def prepare_priors(model, study, config):
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    indices = torch.arange(len(study.patient_ids), device=study.times.device)
    valid = study.observation_mask
    rank = torch.arange(valid.shape[1], device=study.times.device).unsqueeze(0)
    predictions = []
    effects = []
    for count in range(4):
        context = valid & (rank < count)
        eta = estimate_effects(
            model, study, indices, context, config.effect_penalties[count], config
        )
        with torch.no_grad():
            curve, _ = model(study.covariates, study.doses, study.times, eta)
        predictions.append(curve)
        effects.append(eta)
    return torch.stack(predictions, dim=1), torch.stack(effects, dim=1)


def masks(valid, counts):
    rank = torch.arange(valid.shape[1], device=valid.device).unsqueeze(0)
    return valid & (rank < counts.unsqueeze(1)), valid & (rank >= 3)


def context_loss(model, study, priors, effects, patient_indices, config):
    base = torch.tensor(patient_indices, dtype=torch.long, device=study.times.device)
    indices = base.repeat_interleave(4)
    counts = torch.arange(4, device=study.times.device).repeat(len(base))
    valid = study.observation_mask[indices]
    context, future = masks(valid, counts)
    predictions, representation = model(
        study.covariates[indices],
        effects[base].reshape(-1, effects.shape[-1]),
        study.doses[indices],
        study.times,
        priors[base].reshape(-1, priors.shape[-1]),
        study.observation_indices[indices],
        study.observations[indices],
        context,
    )
    predicted = predictions.gather(1, study.observation_indices[indices])
    residual = predicted[future] - study.observations[indices][future]
    log_residual = torch.log1p(predicted[future]) - torch.log1p(study.observations[indices][future])
    refinement = sum(value.square().mean() for value in model.effect_refinement.parameters())
    return (
        residual.square().mean()
        + config.log_loss_weight * log_residual.square().mean()
        + config.representation_penalty * representation.square().mean()
        + config.effect_refinement_penalty * refinement
    )


@torch.no_grad()
def validation_rmse(model, study, priors, effects, indices, count):
    base = torch.tensor(indices, dtype=torch.long, device=study.times.device)
    counts = torch.full((len(base),), count, dtype=torch.long, device=study.times.device)
    context, future = masks(study.observation_mask[base], counts)
    predictions, _ = model(
        study.covariates[base],
        effects[base, count],
        study.doses[base],
        study.times,
        priors[base, count],
        study.observation_indices[base],
        study.observations[base],
        context,
    )
    predicted = predictions.gather(1, study.observation_indices[base])
    return float((predicted[future] - study.observations[base][future]).square().mean().sqrt())


def fit_neural_model(study, population_model, priors, effects, config):
    train = study.indices("train")
    validation = study.indices("validation")
    model = PopulationNCDERefinedEffects().to(study.times.device)
    model.population_model.load_state_dict(population_model.state_dict())
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=config.ncde_learning_rate, weight_decay=1e-5)
    rng = np.random.default_rng(config.seed)
    best = float("inf")
    best_state = None
    stale = 0
    history = []
    for epoch in range(1, config.ncde_epochs + 1):
        model.train()
        model.population_model.eval()
        losses = []
        shuffled = rng.permutation(train)
        for start in range(0, len(train), config.batch_size):
            batch = shuffled[start : start + config.batch_size]
            optimizer.zero_grad(set_to_none=True)
            objective = context_loss(model, study, priors, effects, batch, config)
            objective.backward()
            torch.nn.utils.clip_grad_norm_(trainable, 5.0)
            optimizer.step()
            losses.append(float(objective.detach()))
        model.eval()
        rows = [validation_rmse(model, study, priors, effects, validation, k) for k in range(4)]
        score = float(np.mean(rows))
        history.append(
            {
                "epoch": epoch,
                "train_objective": float(np.mean(losses)),
                "validation_mean_rmse": score,
                **{f"validation_rmse_k{k}": rows[k] for k in range(4)},
            }
        )
        if score < best - 1e-4:
            best = score
            best_state = copy.deepcopy(model.state_dict())
            stale = 0
        else:
            stale += 1
        if stale >= config.ncde_patience:
            break
    if best_state is None:
        raise RuntimeError("Pop-NCDE training produced no checkpoint")
    model.load_state_dict(best_state)
    model.eval()
    return model, pd.DataFrame(history)
