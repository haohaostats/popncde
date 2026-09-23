import torch

from popncde.core import PopulationNCDERefinedEffects, PopulationPK


def test_population_pk_shape():
    model = PopulationPK()
    covariates = torch.zeros(2, 4)
    doses = torch.zeros(2, 9)
    doses[:, 0] = 1.0
    times = torch.arange(9, dtype=torch.float32) * 0.25
    eta = torch.zeros(2, 5)
    predictions, parameters = model(covariates, doses, times, eta)
    assert predictions.shape == (2, 9)
    assert parameters.shape == (2, 5)
    assert torch.all(predictions >= 0)


def test_popncde_shape():
    model = PopulationNCDERefinedEffects()
    covariates = torch.zeros(2, 4)
    effects = torch.zeros(2, 5)
    doses = torch.zeros(2, 9)
    doses[:, 0] = 1.0
    times = torch.arange(9, dtype=torch.float32) * 0.25
    prior, _ = model.population_model(covariates, doses, times, effects)
    indices = torch.tensor([[2, 4], [2, 4]])
    observations = prior.gather(1, indices)
    context = torch.tensor([[True, False], [True, True]])
    predictions, representation = model(
        covariates, effects, doses, times, prior, indices, observations, context
    )
    assert predictions.shape == (2, 9)
    assert representation.shape == (2, 10)
    assert torch.all(predictions >= 0)
