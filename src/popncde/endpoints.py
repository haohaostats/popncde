from __future__ import annotations

import numpy as np


def whole_profile_endpoints(times, concentrations, dose_times):
    result = {
        "auc": float(np.trapezoid(concentrations, times)),
        "cmax": float(np.max(concentrations)),
    }
    if len(dose_times):
        result["final_predose"] = float(np.interp(float(np.max(dose_times)), times, concentrations))
    else:
        result["final_predose"] = float("nan")
    return result


def future_endpoints(times, concentrations, dose_times, landmark):
    mask = times >= landmark
    future_times = times[mask]
    future_values = concentrations[mask]
    if not len(future_times):
        return {
            "future_auc": float("nan"),
            "future_cmax": float("nan"),
            "final_predose": float("nan"),
        }
    later_doses = np.asarray(dose_times)[np.asarray(dose_times) > landmark]
    predose = (
        float(np.interp(float(np.max(later_doses)), times, concentrations))
        if len(later_doses)
        else float("nan")
    )
    return {
        "future_auc": float(np.trapezoid(future_values, future_times)),
        "future_cmax": float(np.max(future_values[future_times > landmark]))
        if np.any(future_times > landmark)
        else float("nan"),
        "final_predose": predose,
    }
