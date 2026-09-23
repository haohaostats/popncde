from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PopNCDEConfig:
    seed: int = 7321
    grid_step_h: float = 0.25
    validation_fraction: float = 0.15
    pk_epochs: int = 180
    pk_learning_rate: float = 2e-3
    pk_patience: int = 30
    ncde_epochs: int = 120
    batch_size: int = 32
    ncde_learning_rate: float = 7e-4
    ncde_patience: int = 25
    log_loss_weight: float = 0.15
    representation_penalty: float = 3e-4
    effect_refinement_penalty: float = 2e-3
    map_steps: int = 100
    map_learning_rate: float = 0.05
    effect_penalties: tuple[float, float, float, float] = (0.0, 0.0003, 0.01, 0.003)
    device: str = "cpu"
