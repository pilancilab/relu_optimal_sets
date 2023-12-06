"""
Running tuning experiment on UCI datasets.
"""
import os
import torch
import numpy as np
from functools import reduce

from scnn.regularizers import NeuronGL1

from solfns.datasets import load_uci_dataset
from solfns.utils import quantile_metrics
from solfns.tuning import run_tuning_experiment


# device = "cuda" if torch.cuda.is_available() else "cpu"
device = "cpu"
np.random.seed(778)


def load_dataset(name, repeat):
    seed = 650 + repeat
    return load_uci_dataset(
        name,
        src="./data/uci/datasets",
        test_prop=0.8,
        use_valid=False,
        split_seed=seed,
        unitize_data_cols=False,
    )


max_neurons = 500
lam = 1e-3
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
    "horse-colic",
    "ilpd-indian-liver",
    "parkinsons",
    "pima",
    "tic-tac-toe",
    "statlog-heart",
    "ionosphere",
]

methods = [
    "min_l2_norm",
    "max_l1_norm",
    "valid_mse",
    "test_mse",
]

verbose = False
force = False
use_ipm = True

train_accuracy = {}
test_accuracy = {}

for name in datasets:
    train, test = run_tuning_experiment(
        name,
        load_dataset,
        max_neurons,
        regularizer,
        methods,
        device,
        repeats,
        use_ipm,
        force,
        verbose,
    )

    train_accuracy[name] = {
        key: quantile_metrics(value) for key, value in train.items()
    }
    test_accuracy[name] = {key: quantile_metrics(value) for key, value in test.items()}

# turn results into table

paper_methods = [
    "min_l2_norm",
    "max_l1_norm",
    "valid_mse",
    "test_mse",
]

paper_datasets = [
    "fertility",
    "heart-hungarian",
    "mammographic",
    "monks-1",
    "planning",
    "spectf",
    "horse-colic",
    "ilpd-indian-liver",
    "parkinsons",
    "pima",
]

paper_table = (
    reduce(lambda acc, z: f"{acc}, {z}", paper_methods, "Data") + ", Max Diff. \n"
)

for name in paper_datasets:
    min_l2 = test_accuracy[name]["min_l2_norm"]["center"][0]
    accs = [test_accuracy[name][method]["center"][0] for method in methods]
    diff = round(max(accs) - min(accs), 2)
    paper_table += (
        reduce(
            lambda acc, method: acc
            + f", {round(test_accuracy[name][method]['center'][0], 2)}",
            paper_methods,
            name,
        )
        + f", {diff}"
        + "\n"
    )

os.makedirs(os.path.join(".", "tables"), exist_ok=True)
file_path = os.path.join(".", "tables", "table_1.csv")

with open(file_path, "w") as f:
    f.write(paper_table)

full_table = reduce(lambda acc, z: f"{acc}, {z}", paper_methods, "Data") + ", Range \n"

for name in datasets:
    min_l2 = test_accuracy[name]["min_l2_norm"]["center"][0]
    center = [test_accuracy[name][method]["center"][0] for method in methods]
    upper = [test_accuracy[name][method]["upper"][0] for method in methods]
    lower = [test_accuracy[name][method]["lower"][0] for method in methods]
    center_diff, lower_diff, upper_diff = (
        round(max(center) - min(center), 2),
        round(max(lower) - min(lower), 2),
        round(max(upper) - min(upper), 2),
    )
    full_table += (
        reduce(
            lambda acc, method: acc
            + f", {round(test_accuracy[name][method]['center'][0], 2)} ({round(test_accuracy[name][method]['lower'][0], 2)}/{round(test_accuracy[name][method]['upper'][0], 2)})",
            paper_methods,
            name,
        )
        + f", {center_diff} ({lower_diff}/{upper_diff})"
        + "\n"
    )

file_path = os.path.join(".", "tables", "table_2.csv")

with open(file_path, "w") as f:
    f.write(full_table)
