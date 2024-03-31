from pdb import set_trace

import jax
import jax.numpy as jnp
import jax.scipy as jsp
import optax
import tqdm
from jax import jit, value_and_grad, vmap
from tinygp import GaussianProcess, kernels

jax.config.update("jax_enable_x64", True)


# ------------------------------
# Models
# ------------------------------
def build_gp(params):
    Z = params["Z"]
    tau_z = jnp.exp(params["log_tau_z"])
    sgm_z = jnp.exp(params["log_sgm_z"])
    eps_z = jnp.exp(params["log_eps_z"])

    kernel = tau_z * kernels.ExpSquared(scale=sgm_z)
    gp = GaussianProcess(kernel, Z, diag=eps_z)
    return gp


@jit
@value_and_grad
def loss_and_grads(params, X):
    Z = params["Z"]
    N, Q = Z.shape

    gp = build_gp(params)

    # prior
    log_prob = 0
    log_prob += vmap(jsp.stats.multivariate_normal.logpdf, in_axes=(0, None, None))(Z, jnp.zeros(Q), jnp.eye(Q)).sum()

    # likelihood
    log_prob += vmap(gp.log_probability, in_axes=(1,))(X).sum()

    return -log_prob


# ------------------------------
# Inference
# ------------------------------
def inference(req: dict) -> dict:
    n_iter = req["n_iter"]
    X = req["data"]["X"]
    params = req["params"]
    optimizers = req["optimizers"]

    optimizer_states = {}
    for param_key, optimizer in optimizers.items():
        if optimizer is None:
            continue
        optimizer_states[param_key] = optimizer.init(params[param_key])

    loss_vals = []
    with tqdm.trange(n_iter) as progress_bar:
        for i in progress_bar:
            # ------------------------------
            # Compute loss and grands
            # ------------------------------
            loss_val, grads = loss_and_grads(params, X)
            loss_vals.append(loss_val.item())

            # ------------------------------
            # Update params
            # ------------------------------
            for param_key in optimizer_states.keys():
                param_val = params[param_key]
                optimizer = optimizers[param_key]
                state = optimizer_states[param_key]

                updates, new_state = optimizer.update(grads[param_key], state, param_val)
                params[param_key] = optax.apply_updates(param_val, updates)
                optimizer_states[param_key] = new_state

            progress_bar.set_postfix(dict(loss=f"{loss_val:.3f}"))

    return dict(loss_vals=loss_vals, params=params)
