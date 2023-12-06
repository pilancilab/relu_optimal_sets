"""
Utilities for running tuning experiments.
"""
import os
from copy import deepcopy
from collections import defaultdict
import pickle as pkl

import torch
import numpy as np
import cvxpy as cp


from scnn.models import ConvexReLU
from scnn.solvers import (
    AL,
    CVXPYSolver,
)
from scnn.metrics import Metrics
from scnn.activations import sample_gate_vectors
from scnn.optimize import optimize_model
from scnn.private.interface import build_internal_model, get_nc_formulation
from scnn.private.utils.data import train_test_split
from scnn.private.interface.utils import lab
from scnn.regularizers import NeuronGL1


tuning_objectives = [
    "min_l2_norm",
    "min_l1_norm",
    "min_linf_norm",
    "max_l1_norm",
    "max_smallest",
    "valid_mse",
    "test_mse",
]


def fit_model(
    max_neurons,
    train_set,
    test_set,
    regularizer,
    device,
    repeat,
    use_ipm=False,
    verbose=False,
):
    X_train, y_train = train_set
    X_test, y_test = test_set
    n, d = train_set[0].shape
    G = sample_gate_vectors(123 + repeat, d, max_neurons)
    base_model = ConvexReLU(G, bias=False)

    # note that commercial solvers like MOSEK/Gurobi can be used if they are installed.
    if use_ipm:
        solver = CVXPYSolver(base_model, "mosek")
    else:
        solver = AL(base_model, tol=1e-8, constraint_tol=1e-8)

    metrics = Metrics(
        metric_freq=25,
        model_loss=True,
        train_accuracy=True,
        test_accuracy=True,
        neuron_sparsity=True,
        constraint_gaps=True,
        lagrangian_grad=True,
    )

    base_model, final_metrics, _ = optimize_model(
        base_model,
        solver,
        metrics,
        X_train,
        y_train,
        X_test,
        y_test,
        regularizer=regularizer,
        verbose=verbose,
        return_convex=True,
        unitize_data=False,
        dtype="float64",
        device=device,
    )

    return base_model


def tune_by_optimization(
    Z, Z_valid, y_valid, Z_test, y_test, model_fit, objective_name
):
    _, m = Z.shape
    beta = cp.Variable(m, nonneg=True)

    constraints = [Z @ beta == model_fit]

    # first just try finding minimum norm solution
    if objective_name == "min_l2_norm":
        # compute the min-norm solution.
        objective = cp.Minimize(cp.norm2(beta))
    elif objective_name == "min_l1_norm":
        objective = cp.Minimize(cp.norm1(beta))
    elif objective_name == "min_linf_norm":
        objective = cp.Minimize(cp.norm(beta, "inf"))
    elif objective_name == "max_l1_norm":
        # maximize L1 norm
        objective = cp.Minimize(-cp.sum(beta))
    elif objective_name == "max_smallest":
        # maximize smallest element
        objective = cp.Minimize(cp.max(-beta))
    elif objective_name == "valid_mse":
        # minimize the validation error over the optimal set.
        objective = cp.Minimize(cp.sum((Z_valid @ beta - np.squeeze(y_valid)) ** 2))
    elif objective_name == "test_mse":
        # minimize the validation error over the optimal set.
        objective = cp.Minimize(cp.sum((Z_test @ beta - np.squeeze(y_test)) ** 2))

    problem = cp.Problem(objective, constraints)

    try:
        problem.solve(solver="MOSEK", verbose=False)

        if beta.value is None:
            raise ValueError()
    except:
        print("Falling back to default solver.")
        try:
            problem.solve(solver="ECOS", verbose=False)
        except:
            return np.ones((m, 1))

    print("Problem Status:", problem.status)

    if beta.value is None or problem.status in ["infeasible", "unbounded"]:
        return np.ones((m, 1))

    return beta.value.reshape(-1, 1)


def tune_model(
    base_model, X_train, X_valid, y_valid, X_test, y_test, method="min_norm"
):
    model_fit = np.squeeze(base_model(X_train))

    params = np.stack(base_model.parameters)
    active_set = np.sum(params**2, axis=-1) != 0

    D = base_model.compute_activations(X_train)

    w_pos = np.squeeze(params[0][active_set[0]])
    w_neg = np.squeeze(params[1][active_set[1]])

    D_pos = D[:, np.squeeze(active_set[0])]
    D_neg = D[:, np.squeeze(active_set[1])]

    fits = [
        np.multiply(D_pos[:, i], X_train @ w_pos[i]) for i in range(w_pos.shape[0])
    ] + [-np.multiply(D_neg[:, i], X_train @ w_neg[i]) for i in range(w_neg.shape[0])]

    Z = np.stack(fits).T
    n, m = Z.shape

    Z_valid = np.concatenate(
        [np.maximum(0, X_valid @ w_pos.T), -np.maximum(0, X_valid @ w_neg.T)],
        axis=-1,
    )

    Z_test = np.concatenate(
        [np.maximum(0, X_test @ w_pos.T), -np.maximum(0, X_test @ w_neg.T)],
        axis=-1,
    )

    if method in tuning_objectives:
        # solve optimization problem
        beta = tune_by_optimization(
            Z,
            Z_valid,
            y_valid,
            Z_test,
            y_test,
            model_fit,
            method,
        )
    else:
        raise ValueError(f"Tuning method {method} not recognized!")

    # update model weights
    new_params = deepcopy(params)
    new_params[active_set] = params[active_set] * beta

    new_model = deepcopy(base_model)
    new_model.set_parameters(new_params)

    return new_model


def sample_optimal_polytope(
    base_model,
    X_train,
    y_train,
    X_valid,
    y_valid,
    X_test,
    y_test,
    n_samples,
):
    model_fit = np.squeeze(base_model(X_train))

    params = np.stack(base_model.parameters)
    active_set = np.sum(params**2, axis=-1) != 0

    D = base_model.compute_activations(X_train)

    w_pos = np.squeeze(params[0][active_set[0]])
    w_neg = np.squeeze(params[1][active_set[1]])

    D_pos = D[:, np.squeeze(active_set[0])]
    D_neg = D[:, np.squeeze(active_set[1])]

    fits = [
        np.multiply(D_pos[:, i], X_train @ w_pos[i]) for i in range(w_pos.shape[0])
    ] + [-np.multiply(D_neg[:, i], X_train @ w_neg[i]) for i in range(w_neg.shape[0])]

    Z = np.stack(fits).T
    n, m = Z.shape

    Z_test = np.concatenate(
        [np.maximum(0, X_test @ w_pos.T), -np.maximum(0, X_test @ w_neg.T)],
        axis=-1,
    )
    samples = sample_polytope(Z, model_fit, n_samples=n_samples)
    train_accuracies = np.sum(
        np.sign(Z @ samples.T) == y_train.reshape(-1, 1), axis=0
    ) / len(y_train)
    test_accuracies = np.sum(
        np.sign(Z_test @ samples.T) == y_test.reshape(-1, 1), axis=0
    ) / len(y_test)

    return train_accuracies, test_accuracies


def mse(model, X, y):
    return np.sum((model(X) - y) ** 2) / len(y)


def accuracy(model, X, y):
    return np.sum(np.sign(model(X)) == y) / len(y)


def c_to_nc_cuda(model, regularizer, X_train):
    # manually revert device to CPU.
    internal_model = build_internal_model(
        model, regularizer, torch.tensor(X_train, dtype=lab.get_dtype())
    )
    internal_model.set_weights(
        torch.stack(
            [torch.tensor(m, dtype=lab.get_dtype()) for m in model.parameters]
        ).cuda()
    )

    nc_model = get_nc_formulation(internal_model)

    return nc_model, internal_model


def c_to_nc_cpu(model, regularizer, X_train):
    internal_model = build_internal_model(model, regularizer, X_train)
    internal_model.set_weights(np.stack(model.parameters))

    nc_model = get_nc_formulation(internal_model)

    return nc_model, internal_model


def c_to_nc(model, regularizer, X_train, device):
    if device == "cuda":
        return c_to_nc_cuda(model, regularizer, X_train)
    else:
        return c_to_nc_cpu(model, regularizer, X_train)


def run_tuning_experiment(
    dataset_name,
    dataset_loader,
    max_neurons,
    regularizer,
    tuning_methods,
    device,
    repeats,
    use_ipm=False,
    force=False,
    verbose=False,
):
    train_accuracy = defaultdict(list)
    test_accuracy = defaultdict(list)
    if device == "cuda":
        lab.set_backend("torch")
    else:
        lab.set_backend("numpy")

    lab.set_device(device)
    lab.set_dtype("float64")

    for r in range(repeats):
        print(f"Repeat: {r+1}/{repeats}")

        full_train_set, test_set = dataset_loader(dataset_name, r)
        full_train_set, test_set = lab.all_to_np(full_train_set), lab.all_to_np(
            test_set
        )
        train_set, valid_set = train_test_split(
            full_train_set[0], full_train_set[1], 0.25, split_seed=r
        )

        X_train, y_train = train_set
        X_test, y_test = test_set

        base_model_path = f"./results/tuning/{dataset_name}_base_model_{r}_{repeats}_{regularizer.lam}_{max_neurons}"

        if use_ipm:
            base_model_path += "_ipm"

        base_model_path += ".pkl"

        os.makedirs("./results/tuning/", exist_ok=True)

        if os.path.exists(base_model_path) and not force:
            print("Loading base model.")
            with open(base_model_path, "rb") as f:
                base_model = pkl.load(f)
        else:
            print("Training base model.")
            base_model = fit_model(
                max_neurons,
                train_set,
                test_set,
                regularizer,
                device,
                r,
                use_ipm,
                verbose,
            )
            base_model.dual_parameters = None
            with open(base_model_path, "wb") as f:
                pkl.dump(base_model, f)

        nc_base_model, _ = c_to_nc(base_model, regularizer, X_train, device)

        print("Base MSE", mse(nc_base_model, X_test, y_test))

        train_accuracy["base_model"].append([accuracy(nc_base_model, X_train, y_train)])
        test_accuracy["base_model"].append([accuracy(nc_base_model, X_test, y_test)])

        for objective_name in tuning_methods:
            np.random.seed(778 + repeats)
            # set the seed for repeatability of random methods
            print("Objective name:", objective_name)

            tuned_model_path = f"./results/tuning/{dataset_name}_tuned_{r}_{repeats}_{regularizer.lam}_{objective_name}_{max_neurons}"

            if use_ipm:
                tuned_model_path += "_ipm"

            tuned_model_path += ".pkl"

            if os.path.exists(tuned_model_path) and not force:
                print("Loading tuned model.")
                with open(tuned_model_path, "rb") as f:
                    tuned_model = pkl.load(f)
            else:
                print("Tuning base model.")
                tuned_model = tune_model(
                    base_model,
                    X_train,
                    valid_set[0],
                    valid_set[1],
                    test_set[0],
                    test_set[1],
                    objective_name,
                )
                tuned_model.dual_parameters = None
                with open(tuned_model_path, "wb") as f:
                    pkl.dump(tuned_model, f)

            nc_tuned_model, _ = c_to_nc(
                tuned_model,
                regularizer,
                X_train,
                device,
            )

            # print("diffs", tuned_model(X_train) - nc_tuned_model(X_train))

            train_acc = accuracy(tuned_model, X_train, y_train)
            test_acc = accuracy(tuned_model, X_test, y_test)

            train_accuracy[objective_name].append([train_acc])
            test_accuracy[objective_name].append([test_acc])

    return train_accuracy, test_accuracy


def run_distribution_experiment(
    dataset_name,
    dataset_loader,
    max_neurons,
    lambda_grid,
    n_samples,
    device,
    use_ipm=False,
    force=False,
    verbose=False,
):
    train_accuracy = defaultdict(list)
    test_accuracy = defaultdict(list)

    if device == "cuda":
        lab.set_backend("torch")
    else:
        lab.set_backend("numpy")
    lab.set_device(device)
    lab.set_dtype("float64")

    # only one train/test split
    full_train_set, test_set = dataset_loader(dataset_name, 0)
    full_train_set, test_set = lab.all_to_np(full_train_set), lab.all_to_np(test_set)
    train_set, valid_set = train_test_split(
        full_train_set[0], full_train_set[1], 0.25, split_seed=0
    )

    for lam in lambda_grid:
        regularizer = NeuronGL1(lam)

        print(f"Regularizer: {lam}")

        X_train, y_train = train_set
        X_test, y_test = test_set

        base_model_path = (
            f"./results/tuning/{dataset_name}_base_model_{lam}_{max_neurons}"
        )

        if use_ipm:
            base_model_path += "_ipm"

        base_model_path += ".pkl"

        os.makedirs("./results/tuning/", exist_ok=True)

        if os.path.exists(base_model_path) and not force:
            print("Loading base model.")
            with open(base_model_path, "rb") as f:
                base_model = pkl.load(f)
        else:
            print("Training base model.")
            base_model = fit_model(
                max_neurons,
                train_set,
                test_set,
                regularizer,
                device,
                0,
                use_ipm,
                verbose,
            )
            base_model.dual_parameters = None
            with open(base_model_path, "wb") as f:
                pkl.dump(base_model, f)

        nc_base_model, _ = c_to_nc(base_model, regularizer, X_train, device)

        print("Base MSE", mse(nc_base_model, X_test, y_test))

        np.random.seed(778)

        distribution_path = f"./results/tuning/{dataset_name}_distribution_{regularizer.lam}_{max_neurons}"

        if use_ipm:
            distribution_path += "_ipm"

        distribution_path += ".pkl"

        if os.path.exists(distribution_path) and not force:
            print("Loading tuned model.")
            with open(distribution_path, "rb") as f:
                train, test = pkl.load(f)
        else:
            print("Taking samples from optimal polytope.")
            train, test = sample_optimal_polytope(
                base_model,
                X_train,
                y_train,
                valid_set[0],
                valid_set[1],
                test_set[0],
                test_set[1],
                n_samples,
            )
            with open(distribution_path, "wb") as f:
                pkl.dump((train, test), f)

        train_accuracy[lam] = train
        test_accuracy[lam] = test

    return train_accuracy, test_accuracy
