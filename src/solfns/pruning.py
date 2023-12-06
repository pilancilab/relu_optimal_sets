"""
Utilities for running pruning experiments.
"""

import os
from copy import deepcopy
from collections import defaultdict
import pickle as pkl

import torch
import numpy as np
from scipy.sparse.linalg import lsqr

import cvxpy as cp


from scnn.models import ConvexReLU
from scnn.solvers import AL
from scnn.metrics import Metrics
from scnn.activations import sample_gate_vectors
from scnn.optimize import optimize_model
from scnn.private.interface import build_internal_model, get_nc_formulation
from scnn.private.interface.utils import lab


def fit_model(
    max_neurons,
    train_set,
    test_set,
    regularizer,
    device,
    repeat,
    verbose=False,
):
    X_train, y_train = train_set
    X_test, y_test = test_set
    n, d = train_set[0].shape
    G = sample_gate_vectors(123 + repeat, d, max_neurons)
    base_model = ConvexReLU(G, bias=False)

    # note that commercial solvers like MOSEK/Gurobi can be used if they are installed.
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


# Pruning Functions


def prune_neuron_optimal(fits, w, active_set, idx):
    beta = cp.Variable(len(fits), nonneg=False)

    Z = fits.copy()
    Z[idx] = 0

    constraints = [Z.T @ beta == fits[idx]]

    objective = cp.Minimize(cp.sum(beta**2))

    problem = cp.Problem(objective, constraints)

    # solve the optimization problem
    try:
        problem.solve(solver="MOSEK", verbose=False)
    except:
        return None, None, None

    # check for infeasibility
    if problem.status == "infeasible":
        return None, None, None

    coeffs = beta.value

    return update_fits_opt(fits, w, active_set, idx, coeffs)


def prune_neuron_lstsq(fits, w, active_set, idx):
    Z = fits.copy()
    Z[idx] = 0

    res = lsqr(Z.T, fits[idx])
    coeffs = res[0]

    return update_fits_opt(fits, w, active_set, idx, coeffs)


def update_fits_opt(fits, w, active_set, idx, coeffs):
    coeffs[idx] = -1

    idx_max = np.argmax(np.abs(coeffs))

    beta_max = coeffs[idx_max]

    scales = 1 - (coeffs / beta_max)
    scales[idx_max] = 0
    scales = scales.reshape(-1, 1)

    fits = fits * scales
    active_set[idx_max] = False

    # update weights
    w = w * scales.reshape((w.shape[0], w.shape[1], -1, 1))

    active_set = (np.sum(w**2, axis=-1) != 0).ravel()

    return fits, w, active_set


def prune_neuron_random(
    fits,
    params,
    active_set,
    try_optimal,
    internal_model,
    X_train,
    y_train,
):
    reshaped_active = active_set.reshape(params.shape[0], params.shape[1], -1)
    active_neurons = params[reshaped_active]

    idx = np.random.randint(active_neurons.shape[0])

    # prune neuron
    active_neurons[idx] = 0
    params[reshaped_active] = active_neurons

    active_set = (np.sum(params**2, axis=-1) != 0).ravel()

    return fits, params, active_set, False


def prune_neuron_norm(
    fits,
    params,
    active_set,
    try_optimal,
    internal_model,
    X_train,
    y_train,
):
    reshaped_active = active_set.reshape(params.shape[0], params.shape[1], -1)
    active_neurons = params[reshaped_active]
    norms = np.sum(active_neurons**2, axis=-1)

    idx = np.argmin(norms)

    # prune neuron
    active_neurons[idx] = 0
    params[reshaped_active] = active_neurons

    active_set = (np.sum(params**2, axis=-1) != 0).ravel()

    return fits, params, active_set, False


def prune_neuron_activations(
    fits,
    params,
    active_set,
    try_optimal,
    internal_model,
    X_train,
    y_train,
):
    reshaped_active = active_set.reshape(params.shape[0], params.shape[1], -1)
    active_neurons = params[reshaped_active]
    active_fits = fits[active_set]
    norms = np.sum(active_fits**2, axis=-1)

    idx = np.argmin(norms)

    # prune neuron
    active_neurons[idx] = 0
    params[reshaped_active] = active_neurons

    active_set = (np.sum(params**2, axis=-1) != 0).ravel()

    return fits, params, active_set, False


def prune_neuron_gradient(
    fits,
    params,
    active_set,
    try_optimal,
    internal_model,
    X_train,
    y_train,
):

    gradient = lab.to_np(
        internal_model.grad(
            lab.tensor(X_train, dtype=lab.get_dtype()),
            lab.tensor(y_train, dtype=lab.get_dtype()),
        )
    )

    scores = params * gradient
    reshaped_active = active_set.reshape(params.shape[0], params.shape[1], -1)
    active_scores = scores[reshaped_active]
    active_neurons = params[reshaped_active]
    norms = np.sum(active_scores**2, axis=-1)

    idx = np.argmin(norms)

    # prune neuron
    active_neurons[idx] = 0
    params[reshaped_active] = active_neurons

    active_set = (np.sum(params**2, axis=-1) != 0).ravel()

    return fits, params, active_set, False


def prune_neuron_residual(
    fits,
    params,
    active_set,
    try_optimal,
    internal_model,
    X_train,
    y_train,
):

    residual_norms = []
    for i in range(len(fits)):

        if not active_set[i]:
            residual_norms.append(np.inf)
            continue

        Z = fits.copy()
        Z[i] = 0

        res = lsqr(Z.T, fits[i])

        residual_norms.append(res[3])

    idx = np.argmin(residual_norms)

    # prune neuron
    old_shape = params.shape
    params = params.reshape(-1, old_shape[-1])
    params[idx] = 0
    params = params.reshape(old_shape)

    active_set[idx] = False

    return fits, params, active_set, False


def try_to_prune_optimal(
    fits,
    params,
    active_set,
    try_optimal,
    internal_model,
    X_train,
    y_train,
):
    for i in range(len(fits)):
        if active_set[i]:
            afp, wp, asp = prune_neuron_optimal(fits, params, active_set, i)

            if afp is not None:
                return afp, wp, asp, True

    return fits, params, active_set, False


def prune_corrected_norm(
    fits,
    params,
    active_set,
    try_optimal,
    internal_model,
    X_train,
    y_train,
):

    if try_optimal:
        afp, wp, asp, was_optimal = try_to_prune_optimal(
            fits,
            params,
            active_set,
            try_optimal,
            internal_model,
            X_train,
            y_train,
        )

        # return the optimal prune if it was successful.
        if was_optimal:
            return afp, wp, asp, was_optimal

    # prune using heuristic with correction.
    norms = np.sum(params**2, axis=-1)
    norms[norms == 0] = np.infty
    idx = np.argmin(norms)

    afp, wp, asp = prune_neuron_lstsq(fits, params, active_set, idx)

    return afp, wp, asp, False


def prune_corrected_activations(
    fits,
    params,
    active_set,
    try_optimal,
    internal_model,
    X_train,
    y_train,
):
    if try_optimal:
        afp, wp, asp, was_optimal = try_to_prune_optimal(
            fits,
            params,
            active_set,
            try_optimal,
            internal_model,
            X_train,
            y_train,
        )

        # return the optimal prune if it was successful.
        if was_optimal:
            return afp, wp, asp, was_optimal

    # prune using heuristic with correction.
    norms = np.sum(fits**2, axis=-1)
    norms[norms == 0] = np.infty
    idx = np.argmin(norms)

    afp, wp, asp = prune_neuron_lstsq(fits, params, active_set, idx)

    return afp, wp, asp, False


def prune_corrected_gradient(
    fits,
    params,
    active_set,
    try_optimal,
    internal_model,
    X_train,
    y_train,
):
    if try_optimal:
        afp, wp, asp, was_optimal = try_to_prune_optimal(
            fits,
            params,
            active_set,
            try_optimal,
            internal_model,
            X_train,
            y_train,
        )

        # return the optimal prune if it was successful.
        if was_optimal:
            return afp, wp, asp, was_optimal

    # prune using heuristic with correction.
    gradient = lab.to_np(
        internal_model.grad(
            lab.tensor(X_train, dtype=lab.get_dtype()),
            lab.tensor(y_train, dtype=lab.get_dtype()),
        )
    )
    scores = params * gradient
    norms = np.sum(scores**2, axis=-1)
    norms[norms == 0] = np.infty
    idx = np.argmin(norms)

    afp, wp, asp = prune_neuron_lstsq(fits, params, active_set, idx)

    return afp, wp, asp, False


def prune_corrected_residual(
    fits,
    params,
    active_set,
    try_optimal,
    internal_model,
    X_train,
    y_train,
):
    if try_optimal:
        afp, wp, asp, was_optimal = try_to_prune_optimal(
            fits,
            params,
            active_set,
            try_optimal,
            internal_model,
            X_train,
            y_train,
        )

        # return the optimal prune if it was successful.
        if was_optimal:
            return afp, wp, asp, was_optimal

    residual_norms = []
    for i in range(len(fits)):

        if not active_set[i]:
            residual_norms.append(np.inf)
            continue

        Z = fits.copy()
        Z[i] = 0

        res = lsqr(Z.T, fits[i])

        residual_norms.append(res[3])

    idx = np.argmin(residual_norms)

    afp, wp, asp = prune_neuron_lstsq(fits, params, active_set, idx)

    return afp, wp, asp, False


def prune_corrected_random(
    fits,
    params,
    active_set,
    try_optimal,
    internal_model,
    X_train,
    y_train,
):
    if try_optimal:
        afp, wp, asp, was_optimal = try_to_prune_optimal(
            fits,
            params,
            active_set,
            try_optimal,
            internal_model,
            X_train,
            y_train,
        )

        # return the optimal prune if it was successful.
        if was_optimal:
            return afp, wp, asp, was_optimal

    norms = np.sum(params**2, axis=-1).ravel()
    indices = np.argwhere(norms).ravel()
    idx = np.random.choice(indices)

    afp, wp, asp = prune_neuron_lstsq(fits, params, active_set, idx)

    return afp, wp, asp, False


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


def create_pruning_path(
    pruning_fn,
    base_model,
    active_set,
    fits,
    params,
    regularizer,
    X_train,
    y_train,
    X_test,
    y_test,
    device,
):
    convex_model = deepcopy(base_model)
    starting_width = np.sum(active_set)

    afp = fits.copy()
    wp = params.copy()
    asp = active_set.copy()

    train = []
    test = []
    sparsity_path = []
    optimal_prune = []

    nc_model, internal_model = c_to_nc(convex_model, regularizer, X_train, device)
    train.append(accuracy(nc_model, X_train, y_train))
    test.append(accuracy(nc_model, X_test, y_test))
    sparsity_path.append(starting_width)
    try_optimal = True
    optimal_prune.append(try_optimal)

    for i in range(starting_width - 1):
        afp, wp, asp, try_optimal = pruning_fn(
            afp,
            wp,
            asp,
            try_optimal,
            internal_model,
            X_train,
            y_train,
        )

        convex_model.set_parameters(wp)
        # convert into non-convex model
        nc_model, internal_model = c_to_nc(
            convex_model,
            regularizer,
            X_train,
            device,
        )

        train.append(accuracy(nc_model, X_train, y_train))
        test.append(accuracy(nc_model, X_test, y_test))
        optimal_prune.append(try_optimal)
        sparsity_path.append(starting_width - i)

    return train, test, optimal_prune, sparsity_path


def run_pruning_experiment(
    dataset_name,
    dataset_loader,
    max_neurons,
    regularizer,
    pruning_methods,
    repeats,
    device,
    force=False,
    verbose=False,
):
    train_accuracy = defaultdict(list)
    test_accuracy = defaultdict(list)
    n_neurons = defaultdict(list)
    optimal = defaultdict(list)

    for r in range(repeats):
        print(f"Repeat: {r+1}/{repeats}")

        # load data
        train_set, test_set = dataset_loader(dataset_name, r)
        X_train, y_train = train_set
        X_test, y_test = test_set

        # try loading base model
        base_model_path = f"./results/pruning/{dataset_name}_base_model_{r}_{repeats}_{regularizer.lam}.pkl"
        os.makedirs("./results/pruning/", exist_ok=True)

        if os.path.exists(base_model_path) and not force:
            print("Loading base model.")
            with open(base_model_path, "rb") as f:
                base_model = pkl.load(f)
        else:
            print("Training base model.")
            base_model = fit_model(
                max_neurons, train_set, test_set, regularizer, device, r, verbose
            )
            # remove dual parameters before saving.
            base_model.dual_parameters = None
            with open(base_model_path, "wb") as f:
                pkl.dump(base_model, f)

        D = base_model.compute_activations(X_train)

        params = np.stack(base_model.parameters)
        active_set = (np.sum(params**2, axis=-1) != 0).ravel()

        ws = np.concatenate([params[0], -params[1]], axis=0)
        ws = ws.reshape(-1, params.shape[-1])

        Ds = np.concatenate([D, D], axis=-1)

        fits = np.stack(
            [np.multiply(Ds[:, i], X_train @ ws[i]) for i in range(ws.shape[0])]
        )

        for pruning_fn, name in pruning_methods:

            # try to load results
            file_path = f"./results/pruning/{dataset_name}_pruning_{name}_{r}_{repeats}_{regularizer.lam}.pkl"
            if os.path.exists(file_path) and not force:
                with open(file_path, "rb") as f:
                    train, test, sparsity_path, optimal_prune = pkl.load(f)
            else:
                print(f"Trying Pruning Method: {name}")
                train, test, sparsity_path, optimal_prune = create_pruning_path(
                    pruning_fn,
                    base_model,
                    active_set,
                    fits,
                    params,
                    regularizer,
                    X_train,
                    y_train,
                    X_test,
                    y_test,
                    device,
                )

                with open(file_path, "wb") as f:
                    pkl.dump((train, test, sparsity_path, optimal_prune), f)

            train_accuracy[name].append(train)
            test_accuracy[name].append(test)
            n_neurons[name].append(sparsity_path)
            optimal[name].append(optimal_prune)

    return train_accuracy, test_accuracy, n_neurons, optimal
