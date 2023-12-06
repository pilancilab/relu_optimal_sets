import os

import torch
import numpy as np
import matplotlib.pyplot as plt

from scnn.private.interface import set_device
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

from solfns.datasets import load_pytorch_dataset, load_transforms
from solfns.utils import quantile_metrics


device = "cuda" if torch.cuda.is_available() else "cpu"
np.random.seed(778)

if device == "cuda":
    set_device(device)


# Data Loading and Optimization


def select_classes(X, y, cls_a, cls_b):
    if torch.is_tensor(X):
        X = X.cpu().numpy()
        y = y.cpu().numpy()

    a_indices = y[:, cls_a] == 1
    b_indices = y[:, cls_b] == 1

    X_sub = np.concatenate([X[a_indices], X[b_indices]], axis=0)
    y_sub = np.ones(X_sub.shape[0])
    y_sub[0 : np.sum(a_indices)] = -1
    y_sub = y_sub.reshape(-1, 1)

    # shuffle datasets
    indices = np.arange(len(y_sub))
    np.random.shuffle(indices)
    X_sub = X_sub[indices]
    y_sub = y_sub[indices]

    return X_sub, y_sub


def load_dataset(name, r):
    seed = 650
    cls_a, cls_b = name.split("vs")
    _, cls_a = cls_a.split("_")
    cls_a, cls_b = int(cls_a), int(cls_b)

    pytorch_src = os.path.join(".", "data", "pytorch")
    transform = load_transforms(["to_tensor", "normalize", "flatten"], "mnist")
    train_data = load_pytorch_dataset(
        "mnist",
        pytorch_src,
        train=True,
        transform=transform,
        use_valid=False,
        split_seed=seed,
    )

    test_data = load_pytorch_dataset(
        "mnist",
        pytorch_src,
        train=False,
        transform=transform,
        use_valid=False,
        split_seed=seed,
    )

    train_data = select_classes(train_data[0], train_data[1], cls_a, cls_b)
    test_data = select_classes(test_data[0], test_data[1], cls_a, cls_b)

    return train_data, test_data


max_neurons = 50
lam = 0.01
repeats = 5

regularizer = NeuronGL1(lam)
verbose = False
force = False

class_pairs = [(2, 6), (1, 7), (5, 9)]

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

for cls_a, cls_b in class_pairs:
    dataset_name = f"mnist_{cls_a}vs{cls_b}"

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


# In[17]:


name_to_label = {
    "mnist_2vs6": "2 vs 6",
    "mnist_1vs7": "1 vs 7",
    "mnist_5vs9": "5 vs 9",
}


# appendix plot

error_alpha = 0.2
line_alpha = 0.8
title_fs = 20
axes_fs = 14
ticks_fs = 12

marker_size = 10
markevery = 5
line_width = 3.5


for kwarg in line_kwargs.values():
    kwarg["markevery"] = markevery
    kwarg["markersize"] = marker_size
    kwarg["linewidth"] = line_width

names = list(train_accuracy.keys())
nrows = len(names)

fig, axes = plt.subplots(nrows, 2, figsize=(8, 6))

for i, name in enumerate(names):

    if i == 0:
        axes[i, 0].set_title("Train Accuracy", fontsize=title_fs)

    axes[i, 0].set_ylabel(name_to_label[name], fontsize=axes_fs)

    for method, metrics in train_accuracy[name].items():
        y = metrics["center"]

        x = np.flip(np.arange(len(y)))

        axes[i, 0].fill_between(
            x,
            metrics["lower"],
            metrics["upper"],
            alpha=error_alpha,
            color=line_kwargs[method]["c"],
        )

        axes[i, 0].plot(x, y, alpha=line_alpha, **line_kwargs[method])

    if i == 0:
        axes[i, 1].set_title("Test Accuracy", fontsize=title_fs)

    axes[i, 0].tick_params(labelsize=ticks_fs)

    for method, metrics in test_accuracy[name].items():
        y = metrics["center"]
        x = np.flip(np.arange(len(y)))

        axes[i, 1].fill_between(
            x,
            metrics["lower"],
            metrics["upper"],
            alpha=error_alpha,
            color=line_kwargs[method]["c"],
        )

        axes[i, 1].plot(x, y, alpha=line_alpha, **line_kwargs[method])

        axes[i, 0].set_xlim([0, 20])
        axes[i, 1].set_xlim([0, 20])

        if i == nrows - 1:
            axes[i, 0].set_xlabel("# Neurons", fontsize=axes_fs)
            axes[i, 1].set_xlabel("# Neurons", fontsize=axes_fs)

    axes[i, 1].tick_params(labelsize=ticks_fs)

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
    fontsize=14,
)

fig.tight_layout()
fig.subplots_adjust(
    wspace=0.2,
    hspace=0.3,
    bottom=0.21,
)

plt.savefig(f"figures/figure_7.pdf")
