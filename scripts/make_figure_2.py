"""
Running pruning experiment on UCI datasets.
"""
import torch
import numpy as np
import matplotlib.pyplot as plt

from scnn.regularizers import NeuronGL1

from solfns.pruning import (
    run_pruning_experiment,
    prune_neuron_norm,
    prune_neuron_activations,
    prune_neuron_random,
    prune_neuron_gradient,
    prune_neuron_residual,
    prune_corrected_norm,
    prune_corrected_activations,
    prune_corrected_random,
    prune_corrected_gradient,
    prune_corrected_residual,
)
from solfns.datasets import load_uci_dataset
from solfns.utils import quantile_metrics


device = "cuda" if torch.cuda.is_available() else "cpu"
np.random.seed(778)


# Data Loading and Optimization


def load_dataset(name, repeat):
    seed = 650 + repeat

    return load_uci_dataset(
        name,
        src="./data/uci/datasets",
        test_prop=0.5,
        use_valid=False,
        split_seed=seed,
        unitize_data_cols=False,
    )


max_neurons = 25
lam = 0.01
repeats = 5
regularizer = NeuronGL1(lam)
datasets = [
    "blood",
    "breast-cancer",
    "fertility",
    "heart-hungarian",
    "hepatitis",
    "hill-valley",
    "mammographic",
    "monks-1",
    "planning",
    "spectf",
]
verbose = False
force = False

pruning_methods = [
    (prune_neuron_norm, "Norm"),
    (prune_neuron_random, "Random"),
    (prune_neuron_gradient, "Gradient"),
    (prune_corrected_residual, "CorrResidual"),
]

train_accuracy = {}
test_accuracy = {}
neurons = {}
optimal_indicator = {}

for dataset_name in datasets:
    print(f"Pruning: {dataset_name}.")

    train, test, n_neurons, optimal = run_pruning_experiment(
        dataset_name,
        load_dataset,
        max_neurons,
        regularizer,
        pruning_methods,
        repeats,
        device,
        force,
        verbose,
    )

    train_accuracy[dataset_name] = {
        key: quantile_metrics(value) for key, value in train.items()
    }
    test_accuracy[dataset_name] = {
        key: quantile_metrics(value) for key, value in test.items()
    }
    neurons[dataset_name] = n_neurons
    optimal_indicator[dataset_name] = optimal


# # Paper Figure

paper_datasets = ["breast-cancer", "fertility", "hill-valley", "monks-1", "spectf"]

y_limits = {
    "breast-cancer": [0.35, 0.8],
    "fertility": [0.4, 0.9],
    "hill-valley": [0.08, 0.65],
    "monks-1": [0.15, 0.95],
    "spectf": [0.25, 0.75],
}

x_limits = {
    "breast-cancer": [0, 20],
    "fertility": [0, 32],
    "hill-valley": [0, 30],
    "monks-1": [0, 20],
    "spectf": [0, 17],
}

line_colors = [
    "#000000",
    "#1f77b4",
    "#ff7f0e",
    "#2ca02c",
    "#d62728",
    "#9467bd",
    "#e377c2",
    "#7f7f7f",
    "#bcbd22",
    "#8c564b",
    "#17becf",
    "#556B2F",
    "#FFFF00",
    "#191970",
]


line_kwargs = {
    "Norm": {
        "c": "#1f77b4",
        "marker": "D",
        "label": "Neuron Magnitude",
    },
    "Activation": {
        "c": "#ff7f0e",
        "marker": "o",
        "label": "Activation Magnitude",
    },
    "Gradient": {
        "c": "#d62728",
        "marker": "o",
        "label": "Gradient Magnitude",
    },
    "Residual": {
        "c": "#17becf",
        "marker": "o",
        "label": "Residual",
    },
    "Random": {
        "c": "#2ca02c",
        "marker": "X",
        "label": "Random",
    },
    "CorrNorm": {
        "c": "#bcbd22",
        "marker": "D",
        "label": "Optimal/Neuron",
    },
    "CorrActivation": {
        "c": "#9467bd",
        "marker": "o",
        "label": "Optimal/Activation",
    },
    "CorrGradient": {
        "c": "#e377c2",
        "marker": "o",
        "label": "Optimal/Gradient",
    },
    "CorrRandom": {
        "c": "#7f7f7f",
        "marker": "o",
        "label": "Optimal/Random",
    },
    "CorrResidual": {
        "c": "#8c564b",
        "marker": "P",
        "label": "Optimal/LS",
    },
}


error_alpha = 0.2
line_alpha = 1

axes_fs = 16
title_fs = 22
ticks_fs = 16

marker_size = 10
markevery = 5
line_width = 3.5

for kwarg in line_kwargs.values():
    kwarg["markevery"] = markevery
    kwarg["markersize"] = marker_size
    kwarg["linewidth"] = line_width

fig, axes = plt.subplots(1, len(paper_datasets), figsize=(12.5, 3.5))

for i, name in enumerate(paper_datasets):

    axes[i].set_title(name, fontsize=title_fs)
    if i == 0:
        axes[i].set_ylabel("Test Accuracy", fontsize=axes_fs)

    for method, metrics in test_accuracy[name].items():
        y = metrics["center"]
        x = np.flip(np.arange(len(y)))

        axes[i].fill_between(
            x,
            metrics["lower"],
            metrics["upper"],
            alpha=error_alpha,
            color=line_kwargs[method]["c"],
        )

        axes[i].plot(x, y, alpha=line_alpha, **line_kwargs[method])
        axes[i].set_xlabel("# Neurons", fontsize=axes_fs)
        axes[i].tick_params(labelsize=ticks_fs)

    axes[i].set_ylim(y_limits[name])
    axes[i].set_xlim(x_limits[name])

handles, labels = axes[0].get_legend_handles_labels()
legend = fig.legend(
    handles,
    labels,
    loc="lower center",
    borderaxespad=0.1,
    fancybox=False,
    shadow=False,
    frameon=False,
    ncol=2,
    fontsize=18,
)

for line in legend.get_lines():
    line.set_linewidth(4.0)

fig.tight_layout()
fig.subplots_adjust(
    wspace=0.3,
    hspace=0.2,
    bottom=0.42,
)

fig.savefig("figures/figure_2.pdf")


# # Additional Figures

axes_fs = 16
title_fs = 22
ticks_fs = 16

marker_size = 16
markevery = 4
line_width = 6

for kwarg in line_kwargs.values():
    kwarg["markevery"] = markevery
    kwarg["markersize"] = marker_size
    kwarg["linewidth"] = line_width


paper_datasets = ["breast-cancer", "fertility", "hill-valley", "monks-1", "spectf"]

y_limits = {
    "breast-cancer": [0.35, 0.8],
    "fertility": [0.4, 0.9],
    "hill-valley": [0.08, 0.65],
    "monks-1": [0.15, 0.95],
    "spectf": [0.25, 0.75],
}

x_limits = {
    "breast-cancer": [0, 20],
    "fertility": [0, 20],
    "hill-valley": [0, 20],
    "monks-1": [0, 20],
    "spectf": [0, 17],
    "blood": [0, 15],
    "heart-hungarian": [0, 15],
    "hepatitis": [0, 15],
    "mammographic": [0, 15],
    "planning": [0, 20],
}

title_fs = 24
axes_fs = 20
ticks_fs = 16


fig, axes = plt.subplots(len(paper_datasets), 2, figsize=(12.5, 12.5))

for i, name in enumerate(paper_datasets):

    axes[i, 0].set_ylabel(name, fontsize=axes_fs)

    if i == 0:
        axes[i, 0].set_title("Train Accuracy", fontsize=title_fs)
        axes[i, 1].set_title("Test Accuracy", fontsize=title_fs)

    for j, results in enumerate([train_accuracy, test_accuracy]):

        for method, metrics in results[name].items():
            y = metrics["center"]
            x = np.flip(np.arange(len(y)))

            axes[i, j].fill_between(
                x,
                metrics["lower"],
                metrics["upper"],
                alpha=error_alpha,
                color=line_kwargs[method]["c"],
            )

            axes[i, j].plot(x, y, alpha=line_alpha, **line_kwargs[method])
            axes[i, j].tick_params(labelsize=ticks_fs)

            axes[i, j].set_xlim(x_limits[name])

            if i == len(paper_datasets) - 1:
                axes[i, j].set_xlabel("# Neurons", fontsize=axes_fs)


handles, labels = axes[0, 0].get_legend_handles_labels()
legend = fig.legend(
    handles,
    labels,
    loc="lower center",
    borderaxespad=0.1,
    fancybox=False,
    shadow=False,
    frameon=False,
    ncol=2,
    fontsize=22,
)

for line in legend.get_lines():
    line.set_linewidth(4.0)

fig.tight_layout()
fig.subplots_adjust(
    wspace=0.2,
    hspace=0.2,
    bottom=0.14,
)

fig.savefig("figures/figure_4.pdf")


extra_datasets = ["blood", "heart-hungarian", "hepatitis", "mammographic", "planning"]


fig, axes = plt.subplots(len(extra_datasets), 2, figsize=(12.5, 12.5))

for i, name in enumerate(extra_datasets):

    axes[i, 0].set_ylabel(name, fontsize=axes_fs)

    if i == 0:
        axes[i, 0].set_title("Train Accuracy", fontsize=title_fs)
        axes[i, 1].set_title("Test Accuracy", fontsize=title_fs)

    for j, results in enumerate([train_accuracy, test_accuracy]):

        for method, metrics in results[name].items():
            y = metrics["center"]
            x = np.flip(np.arange(len(y)))

            axes[i, j].fill_between(
                x,
                metrics["lower"],
                metrics["upper"],
                alpha=error_alpha,
                color=line_kwargs[method]["c"],
            )

            axes[i, j].plot(x, y, alpha=line_alpha, **line_kwargs[method])
            axes[i, j].tick_params(labelsize=ticks_fs)

            axes[i, j].set_xlim(x_limits[name])

            if i == len(paper_datasets) - 1:
                axes[i, j].set_xlabel("# Neurons", fontsize=axes_fs)


handles, labels = axes[0, 0].get_legend_handles_labels()
legend = fig.legend(
    handles,
    labels,
    loc="lower center",
    borderaxespad=0.1,
    fancybox=False,
    shadow=False,
    frameon=False,
    ncol=2,
    fontsize=22,
)

for line in legend.get_lines():
    line.set_linewidth(4.0)

fig.tight_layout()
fig.subplots_adjust(
    wspace=0.2,
    hspace=0.2,
    bottom=0.14,
)

fig.savefig("figures/figure_5.pdf")
