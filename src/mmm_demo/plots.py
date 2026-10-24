"""Figures for the saved estimation example."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def save_figures(data, predictions, parameters, output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False,
                         "axes.spines.right": False, "figure.dpi": 140})
    fig, axes = plt.subplots(2, 1, figsize=(9, 5), sharex=True)
    for name, label in [("spend_a", "Channel A"), ("spend_b", "Channel B")]:
        axes[0].plot(data.week, data[name], label=label, linewidth=1)
    axes[0].set_ylabel("Spend (arbitrary units)")
    axes[0].legend(frameon=False, ncol=2)
    axes[1].plot(data.week, data.outcome, color="#52616b", linewidth=1, label="Observed")
    axes[1].plot(data.week, data.true_mean, color="#c66a27", linewidth=1, label="True mean")
    axes[1].set(xlabel="Week", ylabel="Outcome (arbitrary units)")
    axes[1].legend(frameon=False, ncol=2)
    for ax in axes:
        ax.axvline(data.loc[data.split.eq("train"), "week"].max() + 0.5,
                   color="black", linestyle="--", linewidth=0.8)
    fig.tight_layout()
    fig.savefig(output / "01_data.png")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(9, 3.5))
    for ax, names, title in zip(axes, [["alpha_a", "alpha_b"], ["beta_a", "beta_b"]],
                                ["Carryover", "Response coefficient"]):
        selected = parameters.set_index("parameter").loc[names]
        y = np.arange(len(selected))
        ax.hlines(y, selected.credible_low, selected.credible_high, color="#267b98", linewidth=3)
        ax.scatter(selected.posterior_mean, y, color="#267b98", label="Posterior mean")
        ax.scatter(selected.truth, y, color="#c66a27", marker="x", label="Truth", zorder=3)
        ax.set(yticks=y, yticklabels=["Channel A", "Channel B"], title=title)
        ax.margins(y=0.5)
    axes[0].set_xlim(0, 1)
    axes[1].legend(frameon=False, fontsize=8)
    fig.suptitle("Parameter estimates and 90% credible intervals")
    fig.tight_layout()
    fig.savefig(output / "02_parameters.png")
    plt.close(fig)

    test = data.split.eq("test")
    weeks = data.loc[test, "week"].to_numpy()
    predicted = predictions.loc[test]
    residual = data.loc[test, "outcome"].to_numpy() - predicted.prediction.to_numpy()
    fig, axes = plt.subplots(2, 1, figsize=(9, 5), sharex=True)
    axes[0].fill_between(weeks, predicted.predictive_low.to_numpy(),
                         predicted.predictive_high.to_numpy(), alpha=0.2, color="#267b98",
                         label="90% posterior predictive interval")
    axes[0].plot(weeks, predicted.prediction, color="#267b98", label="Predictive mean")
    axes[0].scatter(weeks, data.loc[test, "outcome"], s=15, color="#52616b", label="Observed")
    axes[0].set_ylabel("Outcome")
    axes[0].legend(frameon=False, fontsize=8)
    axes[1].axhline(0, color="black", linewidth=0.8)
    axes[1].plot(weeks, residual, marker="o", markersize=3, color="#52616b", linewidth=0.8)
    axes[1].set(xlabel="Test week", ylabel="Observed - predicted")
    fig.tight_layout()
    fig.savefig(output / "03_prediction.png")
    plt.close(fig)
