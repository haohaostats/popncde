import numpy as np

from popncde.endpoints import future_endpoints, whole_profile_endpoints


def test_endpoints():
    times = np.array([0.0, 1.0, 2.0, 3.0])
    values = np.array([0.0, 1.0, 2.0, 1.0])
    doses = np.array([0.0, 2.0])
    whole = whole_profile_endpoints(times, values, doses)
    future = future_endpoints(times, values, doses, 1.0)
    assert whole["auc"] == 3.5
    assert whole["cmax"] == 2.0
    assert future["future_auc"] == 3.0
    assert future["future_cmax"] == 2.0
