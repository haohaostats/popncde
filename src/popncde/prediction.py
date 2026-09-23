from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from .endpoints import future_endpoints, whole_profile_endpoints
from .plotting import plot_prediction


@dataclass
class Prediction:
    frame: pd.DataFrame
    doses: pd.DataFrame
    concentrations: pd.DataFrame
    history_count: int
    landmark: float

    def future_endpoints(self):
        return future_endpoints(
            self.frame["time_h"].to_numpy(),
            self.frame["prediction"].to_numpy(),
            self.doses["time_h"].to_numpy(),
            self.landmark,
        )

    def whole_profile_endpoints(self):
        return whole_profile_endpoints(
            self.frame["time_h"].to_numpy(),
            self.frame["prediction"].to_numpy(),
            self.doses["time_h"].to_numpy(),
        )

    def plot(self, ax=None):
        return plot_prediction(self, ax=ax)
