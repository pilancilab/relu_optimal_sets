import sys
import numpy as np
import scipy
import matplotlib.pyplot as plt
from copy import deepcopy

from scnn.private.utils.data import gen_regression_data
import cvxpy as cp


from scnn.models import ConvexGatedReLU, ConvexReLU
from scnn.solvers import RFISTA, AL, LeastSquaresSolver, CVXPYSolver, ApproximateConeDecomposition
from scnn.regularizers import NeuronGL1, L2, L1
from scnn.metrics import Metrics
from scnn.activations import sample_gate_vectors
from scnn.optimize import optimize_model, optimize
from scnn.private.interface import build_internal_model, get_nc_formulation

def cartesian_product(*arrays):
    la = len(arrays)
    dtype = np.result_type(*arrays)
    arr = np.empty([len(a) for a in arrays] + [la], dtype=dtype)
    for i, a in enumerate(np.ix_(*arrays)):
        arr[...,i] = a
    return arr.reshape(-1, la)

# Generate realizable synthetic classification problem (ie. Figure 1)
n_train = 4
n_test = 1000
d = 2
kappa = 1  # condition number

(X_train, y_train), (X_test, y_test), _ = gen_regression_data(123, n_train, n_test, d, kappa=kappa)

lam = 0.0001
max_neurons = 20
G = sample_gate_vectors(123, d, max_neurons)
metrics = Metrics(metric_freq=25, 
                  model_loss=True, 
                  train_accuracy=True, 
                  test_accuracy=True, 
                  neuron_sparsity=True, 
                  constraint_gaps=True, 
                  lagrangian_grad=True)



model = ConvexReLU(G)
solver = AL(model, tol=1e-8, constraint_tol=1e-8)
lam = 0.01
regularizer = NeuronGL1(lam)
convex_model, metrics, _ = optimize_model(
    model,
    solver,
    metrics,
    X_train, 
    y_train, 
    X_test, 
    y_test,
    regularizer=regularizer,
    verbose=False,
    return_convex=True,
    unitize_data=False,
    dtype="float64"
)


# compute correlations and dual fit
model_fit = model(X_train)

residual = y_train - model_fit
D = model.compute_activations(X_train)

c = np.einsum("ij, il, ik->ljk", D, residual, X_train) / X_train.shape[0]
c = np.stack([c, -c])

rho = np.stack(model.dual_parameters)

orthant = -(2 * D - np.ones_like(D))
dual_fit = np.einsum(
        "imjk, kj, kl -> imjl",
        np.stack(rho),
        orthant,
        X_train,
    )

v = c - dual_fit

correlations = np.sqrt(np.sum(v**2, axis=-1))

# determine active set

params = np.stack(convex_model.parameters)
active_set = np.sum(params**2, axis=-1) != 0

active_corrs = correlations[active_set]

inactive_corrs = correlations[active_set == False]


# compute prediction matrix for coefficients
model_fit = model(X_train)
residual = y_train - model_fit
w_pos = np.squeeze(params[0][active_set[0]])
w_neg = np.squeeze(params[1][active_set[1]])

D_pos = D[:, np.squeeze(active_set[0])]
D_neg = D[:, np.squeeze(active_set[1])]

fits = ([np.multiply(D_pos[:, i], X_train @ w_pos[i]) for i in range(w_pos.shape[0])] 
+ [-np.multiply(D_neg[:, i], X_train @ w_neg[i]) for i in range(w_neg.shape[0])])

Z = np.stack(fits).T

w_stacked = np.stack(params[active_set])



n, m = Z.shape

# Compute the Null space of Z
U, s, Vh = scipy.linalg.svd(Z)

# coeffcients which span the null space
null_vectors = Vh[n:]

# coeffecients with correct fit:
fit_coeffs = null_vectors

# play with these parameters to control size and density of grid.
grid = [np.arange(-20, 20, 0.7) for i in range(null_vectors.shape[0])]
grid = cartesian_product(*grid)

coeffs_space = grid @ fit_coeffs
coeffs_space = coeffs_space + np.ones((1, 8))

# # keep only positive coefficients
indices = np.sum(coeffs_space >= 0, axis=-1) == 8
valid_coeffs = coeffs_space[indices]


# Apply coefficients to dual vectors

nn_solutions = []
convex_solutions = [] 
for i in range(valid_coeffs.shape[0]):
    nn_solutions.append(((valid_coeffs[i:i+1].T ) ** (1/2)) * w_stacked)
    convex_solutions.append(((valid_coeffs[i:i+1].T )) * w_stacked)

nn_final_sol = np.stack(nn_solutions)
convex_final_sol = np.stack(convex_solutions)


# randomly check some points to make sure the model fit is correct.

test_params = params.copy()
test_model = deepcopy(convex_model)
test_params[active_set] = convex_final_sol[0]
test_model.set_parameters([test_params[0], test_params[1]])


# # Surface Plots


# set of
i = 5
j = 1

fig, axes = plt.subplots(1, 2, figsize=(8, 6), subplot_kw=dict(projection="3d"))

    
azim = 270
elev = 20

axes[0].view_init(elev=elev, azim=azim, roll=0)
axes[1].view_init(elev=elev, azim=azim, roll=0)

axes[0].set_xlabel("$[v_{i}]_1$", fontsize=20)

axes[1].set_xlabel("$[W_{1i}]_1$", fontsize=20)

for k in range(2):
    axes[k].tick_params(      
        bottom=False,      
        top=False,       
        left=False,
        right=False,
        labelbottom=False,
        labeltop=False,
        labelleft=False,
        labelright=False,
    ) 

surf = axes[0].plot_trisurf(convex_final_sol[:, 0+i, j], convex_final_sol[:, 1+i, j], convex_final_sol[:, 2+i, j], cmap='plasma')
axes[0].set_title("Convex", fontsize=26, y=0.92)

surf = axes[1].plot_trisurf(nn_final_sol[:, 0+i, j], nn_final_sol[:, 1+i, j], nn_final_sol[:, 2+i, j], cmap='plasma')
axes[1].set_title("Non-Convex", fontsize=26, y=0.92)

fig.subplots_adjust(
    wspace=-0.82,
    left=0.11,
    hspace=-1,
    bottom=0.1,
)

fig.tight_layout()
plt.savefig("figures/figure_1.png", bbox_inches='tight', pad_inches=0, dpi=500)
