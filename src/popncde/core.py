from __future__ import annotations

import torch
from torch import nn


class PopulationPK(nn.Module):
    def __init__(self, covariate_size: int = 4) -> None:
        super().__init__()
        self.log_typical = nn.Parameter(torch.log(torch.tensor([4.0, 30.0, 6.0, 50.0, 1.0])))
        self.covariate_effects = nn.Parameter(torch.zeros(5, covariate_size))
        self.log_omega = nn.Parameter(torch.log(torch.full((5,), 0.25)))

    def individual_parameters(self, covariates: torch.Tensor, eta: torch.Tensor) -> torch.Tensor:
        values = self.log_typical + covariates @ self.covariate_effects.T + eta
        return torch.exp(values.clamp(-5.0, 6.0))

    def forward(self, covariates, doses, times, eta):
        parameters = self.individual_parameters(covariates, eta)
        cl, vc, q, vp, ka = parameters.unbind(dim=1)
        matrix = torch.zeros(len(covariates), 3, 3, device=covariates.device)
        matrix[:, 0, 0] = -ka
        matrix[:, 1, 0] = ka
        matrix[:, 1, 1] = -(cl + q) / vc
        matrix[:, 1, 2] = q / vp
        matrix[:, 2, 1] = q / vc
        matrix[:, 2, 2] = -q / vp
        transition = torch.matrix_exp(matrix * (times[1] - times[0]))
        state = torch.zeros(len(covariates), 3, device=covariates.device)
        predictions = []
        for step in range(len(times)):
            predictions.append(state[:, 1] / vc)
            state = state.clone()
            state[:, 0] = state[:, 0] + 100.0 * doses[:, step]
            if step + 1 < len(times):
                state = torch.bmm(transition, state.unsqueeze(-1)).squeeze(-1)
        return torch.stack(predictions, dim=1), parameters

    def iiv_penalty(self, eta: torch.Tensor) -> torch.Tensor:
        log_omega = self.log_omega.clamp(-4.0, 0.0)
        return (eta.square() * torch.exp(-2.0 * log_omega) + 2.0 * log_omega).mean()


class SequentialHistoryEncoder(nn.Module):
    def __init__(self, covariate_size: int, representation_size: int, width: int) -> None:
        super().__init__()
        self.initial = nn.Sequential(
            nn.Linear(covariate_size, width), nn.Tanh(), nn.Linear(width, width)
        )
        self.update = nn.GRUCell(5, width)
        self.output = nn.Sequential(
            nn.Linear(width, width), nn.Tanh(), nn.Linear(width, representation_size)
        )

    def forward(
        self,
        covariates,
        observation_times,
        observation_values,
        population_values,
        context_mask,
    ):
        hidden = self.initial(covariates)
        previous_time = torch.zeros(len(covariates), device=covariates.device)
        log_observed = torch.log1p(observation_values)
        log_population = torch.log1p(population_values)
        for column in range(observation_values.shape[1]):
            active = context_mask[:, column]
            current_time = observation_times[:, column]
            features = torch.stack(
                (
                    current_time,
                    current_time - previous_time,
                    log_observed[:, column],
                    log_population[:, column],
                    log_observed[:, column] - log_population[:, column],
                ),
                dim=-1,
            )
            updated = self.update(features, hidden)
            hidden = torch.where(active.unsqueeze(-1), updated, hidden)
            previous_time = torch.where(active, current_time, previous_time)
        representation = self.output(hidden)
        return torch.where(
            context_mask.any(dim=1, keepdim=True), representation, torch.zeros_like(representation)
        )


class PopulationNCDERefinedEffects(nn.Module):
    def __init__(
        self,
        covariate_size: int = 4,
        effect_size: int = 5,
        hidden_size: int = 24,
        representation_size: int = 10,
        width: int = 64,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.history_encoder = SequentialHistoryEncoder(covariate_size, representation_size, 56)
        conditioning_size = covariate_size + effect_size + representation_size + 1
        self.initial = nn.Sequential(
            nn.Linear(conditioning_size, width), nn.SiLU(), nn.Linear(width, hidden_size)
        )
        self.vector_field = nn.Sequential(
            nn.Linear(hidden_size + conditioning_size, width),
            nn.SiLU(),
            nn.Linear(width, width),
            nn.SiLU(),
            nn.Linear(width, hidden_size * 4),
            nn.Tanh(),
        )
        self.residual_readout = nn.Sequential(
            nn.Linear(hidden_size + conditioning_size + 6, width),
            nn.SiLU(),
            nn.Linear(width, 32),
            nn.SiLU(),
            nn.Linear(32, 1),
        )
        nn.init.zeros_(self.residual_readout[-1].weight)
        nn.init.zeros_(self.residual_readout[-1].bias)
        self.population_model = PopulationPK(covariate_size)
        for parameter in self.population_model.parameters():
            parameter.requires_grad_(False)
        refinement_size = covariate_size + effect_size + representation_size + 1
        self.effect_refinement = nn.Sequential(
            nn.Linear(refinement_size, 48),
            nn.SiLU(),
            nn.Linear(48, 32),
            nn.SiLU(),
            nn.Linear(32, effect_size),
        )
        nn.init.zeros_(self.effect_refinement[-1].weight)
        nn.init.zeros_(self.effect_refinement[-1].bias)

    def _field(self, state, conditioning):
        return self.vector_field(torch.cat((state, conditioning), dim=-1)).view(
            -1, self.hidden_size, 4
        )

    def forward(
        self,
        covariates,
        individual_effects,
        doses,
        times,
        population_prior,
        observation_indices,
        observations,
        context_mask,
        return_details: bool = False,
    ):
        horizon = times[-1].clamp_min(1.0)
        observation_times = times[observation_indices] / horizon
        prior_at_observations = population_prior.gather(1, observation_indices)
        first_representation = self.history_encoder(
            covariates,
            observation_times,
            observations,
            prior_at_observations.detach(),
            context_mask,
        )
        count = context_mask.sum(dim=1).to(covariates.dtype)
        refinement_features = torch.cat(
            (
                covariates,
                individual_effects,
                first_representation,
                (count / 3.0).unsqueeze(-1),
            ),
            dim=-1,
        )
        delta = 0.5 * torch.tanh(self.effect_refinement(refinement_features))
        delta = torch.where(context_mask.any(1, keepdim=True), delta, torch.zeros_like(delta))
        refined_effects = individual_effects + delta
        refined_prior, _ = self.population_model(covariates, doses, times, refined_effects)
        predictions, representation = self._predict(
            covariates,
            refined_effects,
            doses,
            times,
            refined_prior,
            observation_indices,
            observations,
            context_mask,
        )
        if return_details:
            return predictions, representation, refined_effects, refined_prior
        return predictions, representation

    def _predict(
        self,
        covariates,
        individual_effects,
        doses,
        times,
        population_prior,
        observation_indices,
        observations,
        context_mask,
    ):
        horizon = times[-1].clamp_min(1.0)
        prior_at_observations = population_prior.gather(1, observation_indices)
        observation_times = times[observation_indices] / horizon
        representation = self.history_encoder(
            covariates,
            observation_times,
            observations,
            prior_at_observations.detach(),
            context_mask,
        )
        count = context_mask.sum(dim=1).to(covariates.dtype)
        conditioning = torch.cat(
            (covariates, individual_effects, representation, (count / 3.0).unsqueeze(-1)),
            dim=-1,
        )
        history_residual = torch.log1p(observations) - torch.log1p(prior_at_observations)
        weights = context_mask.to(history_residual.dtype)
        mean_residual = (history_residual * weights).sum(1) / weights.sum(1).clamp_min(1.0)
        last_column = (context_mask.sum(1) - 1).clamp_min(0)
        last_residual = history_residual.gather(1, last_column.unsqueeze(1)).squeeze(1)
        last_time = observation_times.gather(1, last_column.unsqueeze(1)).squeeze(1)
        has_context = context_mask.any(1)
        mean_residual = torch.where(has_context, mean_residual, torch.zeros_like(mean_residual))
        last_residual = torch.where(has_context, last_residual, torch.zeros_like(last_residual))
        last_time = torch.where(has_context, last_time, torch.zeros_like(last_time))
        state = self.initial(conditioning)
        phase = torch.zeros(len(covariates), device=covariates.device)
        log_prior = torch.log1p(population_prior)
        predictions = []
        for step in range(len(times)):
            normalized_time = (times[step] / horizon).expand(len(covariates))
            since_context = (normalized_time - last_time).clamp_min(0.0)
            readout_features = torch.cat(
                (
                    state,
                    conditioning,
                    log_prior[:, step : step + 1],
                    (phase / 12.0).unsqueeze(-1),
                    normalized_time.unsqueeze(-1),
                    since_context.unsqueeze(-1),
                    mean_residual.unsqueeze(-1),
                    last_residual.unsqueeze(-1),
                ),
                dim=-1,
            )
            residual = 1.25 * torch.tanh(self.residual_readout(readout_features).squeeze(-1))
            predictions.append(torch.expm1((log_prior[:, step] + residual).clamp(0.0, 4.0)))
            dose = doses[:, step]
            has_dose = dose > 0
            if has_dose.any():
                jump = torch.stack(
                    (
                        torch.zeros_like(dose),
                        torch.where(has_dose, -phase / 12.0, torch.zeros_like(phase)),
                        dose,
                        torch.zeros_like(dose),
                    ),
                    dim=-1,
                )
                state = state + torch.bmm(
                    self._field(state, conditioning), jump.unsqueeze(-1)
                ).squeeze(-1)
                phase = torch.where(has_dose, torch.zeros_like(phase), phase)
            if step + 1 < len(times):
                dt = times[step + 1] - times[step]
                prior_increment = log_prior[:, step + 1] - log_prior[:, step]
                control = torch.stack(
                    (
                        torch.full_like(prior_increment, float(dt / horizon)),
                        torch.full_like(prior_increment, float(dt / 12.0)),
                        torch.zeros_like(prior_increment),
                        prior_increment,
                    ),
                    dim=-1,
                )
                first = torch.bmm(self._field(state, conditioning), control.unsqueeze(-1)).squeeze(
                    -1
                )
                proposed = state + first
                second = torch.bmm(
                    self._field(proposed, conditioning), control.unsqueeze(-1)
                ).squeeze(-1)
                state = state + 0.5 * (first + second)
                phase = phase + dt
        return torch.stack(predictions, dim=1), representation
