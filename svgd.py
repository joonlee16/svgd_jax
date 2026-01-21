import jax
import jax.numpy as jnp
from jax import jit, vmap, grad
from jax.lax import fori_loop
from functools import partial
from kernels import *
from helper_functions import *
from config import * 

"""
Core SVGD drivers used in the project. This module implements the main SVGD update algorithms:

- run_svgd: a general SVGD update algorithm.
- run_svgd_with_history: a SVGD runner that records intermediate
    particle states (useful for visualization / debugging).
"""

# -------------------------
# SVGD Update Implementation
# -------------------------
@partial(jit, static_argnums=(0,3,4,5,6))
def run_svgd(log_p, state, key, n_particles, T, dim_u, n_steps=50, lr=1e-3):
    """Run SVGD to optimize batched control trajectories.

    This function treats each particle as a full control sequence for `N`
    agents over a horizon `T` with control dimension `dim_u`. The particles
    are initialized from a Normal distribution and evolved for `n_steps`.

    Args:
        log_p: callable (u_flat, state) -> scalar log-probability (or negative cost).
        state: current environment/state passed into `log_p`.
        key: JAX PRNGKey for initialization.
        n_particles: number of particles/samples.
        T: trajectory horizon length.
        dim_u: dimensionality of control per time-step.
        n_steps: number of SVGD iterations to perform.
        lr: step size (epsilon) for SVGD updates.

    Returns:
        Array of shape (n_particles, N, T, dim_u): final particle set after updates.
    """

    # Compute the log p and the initial samples
    grad_log_p = grad(lambda u: log_p(u, state))
    theta_init = jax.random.normal(key, (n_particles, N, T, dim_u)) * 5.0

    @jax.jit
    def svgd_step(theta, lr):
        n, N, T, dim_u = theta.shape
        
        # Flatten each particle: (n, T*dim_u)
        theta_flat = theta.reshape(n, N*T * dim_u)

        # Compute the grad of log p
        grads = vmap(grad_log_p)(theta_flat)  

        # Get the kernel and its gradient
        # K: (n,n), grad_K: (n,n,d)
        K, grad_K = rbf_kernal(theta_flat)  

        # Compute the phi_hat
        phi_hat = (K @ grads+ jnp.sum(grad_K, axis=1)) / n

        # Update and reshape back to trajectory shape
        new_theta = theta_flat + lr * phi_hat
        return new_theta.reshape(n, N, T, dim_u)

    @jax.jit
    def body(i, state):
        theta = state
        return svgd_step(theta, lr)
    
    new_new = fori_loop(0, n_steps, body, theta_init)
    return new_new

## Run the svgd and also record the history 
def run_svgd_with_history(key, log_prob, n_steps=100, lr=1e-3,record_every=10):
    """Run SVGD on a low-dimensional problem and record history for plotting.

    This helper is intended for quick experiments: each particle is a DIM_U-
    dimensional vector (for example, a 2D point) and the function records the
    particle set every `record_every` iterations for visualization.

    Args:
        key: JAX PRNGKey for initialization.
        log_prob: callable mapping a single particle (array shape (DIM_U,)) to a scalar log-prob.
        n_steps: total SVGD iterations.
        lr: step size for SVGD updates.
        record_every: how often (in iterations) to save the particle set.

    Returns:
        new_theta: final particle set (NUM_PARTICLE, DIM_U)
        history: array with shape (n_records, NUM_PARTICLE, DIM_U) containing recorded states.
    """

    # Compute the log p and the initial samples
    theta_init = jax.random.normal(key, (NUM_PARTICLE, DIM_U)) * 5.0
    n_records = n_steps // record_every + 1
    history = jnp.zeros((n_records, NUM_PARTICLE, DIM_U))

    grad_log_p = grad(log_prob)
    @jax.jit
    def svgd_step(theta, lr):
        n, dim_u = theta.shape
        
        # Flatten each particle: (n, dim_u)
        theta_flat = theta.reshape(n, dim_u)

        # Compute the grad of log p
        grads = vmap(grad_log_p)(theta_flat)  

        # Get the kernel and its gradient
        # K: (n,n), grad_K: (n,n,d)
        K, grad_K = rbf_kernal(theta_flat)  

        # Compute the phi_hat
        phi_hat = (K @ grads+ jnp.sum(grad_K, axis=1)) / n

        # Update and reshape back to trajectory shape
        new_theta = theta_flat + lr * phi_hat
        return new_theta.reshape(n, dim_u)

    def body(i, state):
        theta, history = state
        new_theta = svgd_step(theta, lr)

        # Appending the thetas into the history list every record_every iterations
        idx = i // record_every

        history = jax.lax.cond(i % record_every == 0,
                         lambda k: k.at[idx].set(new_theta.reshape(-1,DIM_U)),
                         lambda k: k, history)

        return new_theta, history
    new_theta, history = fori_loop(0, n_steps, body, (theta_init, history))
    return new_theta, history
