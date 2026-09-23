from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt

COLORS = {
    "population": "#2A9D8F",
    "prior": "#6C5B9B",
    "prediction": "#E76F51",
    "observation": "#263238",
}


def set_plot_theme() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": False,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.bbox": "tight",
        }
    )


def plot_prediction(prediction, ax=None):
    set_plot_theme()
    if ax is None:
        _, ax = plt.subplots(figsize=(7.2, 4.4))
    frame = prediction.frame
    ax.plot(
        frame["time_h"],
        frame["population_pk"],
        color=COLORS["population"],
        lw=2,
        label="Population PK",
    )
    ax.plot(
        frame["time_h"],
        frame["individualized_prior"],
        color=COLORS["prior"],
        lw=2,
        label="Individualized PK prior",
    )
    ax.plot(
        frame["time_h"], frame["prediction"], color=COLORS["prediction"], lw=2.5, label="Pop-NCDE"
    )
    history = prediction.concentrations.sort_values("time_h").head(prediction.history_count)
    if len(history):
        ax.scatter(
            history["time_h"],
            history["concentration"],
            s=36,
            color=COLORS["observation"],
            zorder=4,
            label="Previous concentrations",
        )
        ax.axvline(
            prediction.landmark, color="#87939A", lw=1.3, ls="--", label="Prediction landmark"
        )
    for time in prediction.doses["time_h"].to_numpy():
        ax.axvline(time, ymin=0, ymax=0.035, color="#87939A", lw=1)
    ax.set_xlabel("Time (h)")
    ax.set_ylabel("Concentration")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=3, frameon=False)
    return ax.figure, ax


def plot_individualization(predictions, ax=None):
    set_plot_theme()
    if ax is None:
        _, ax = plt.subplots(figsize=(7.2, 4.4))
    palette = ["#87939A", "#2A9D8F", "#5E81AC", "#E76F51"]
    for color, prediction in zip(palette, predictions, strict=False):
        ax.plot(
            prediction.frame["time_h"],
            prediction.frame["prediction"],
            color=color,
            lw=2,
            label=f"{prediction.history_count} previous concentrations",
        )
    ax.set_xlabel("Time (h)")
    ax.set_ylabel("Predicted concentration")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=2, frameon=False)
    return ax.figure, ax


def save_figure(figure, path, dpi: int = 600) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(target, dpi=dpi)
