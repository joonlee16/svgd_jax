import jax
import jax.numpy as jnp
from jax import jit, vmap, grad
from jax.lax import fori_loop
from functools import partial
from log_prob import *
from kernels import *
from helper_functions import *
from config import * 

# -------------------------
# SVGD Update Implementation
# -------------------------
@partial(jit, static_argnums=(0,3,4,5,6))
def run_svgd(log_p, state, key, n_particles, T, dim_u, n_steps=50, lr=1e-3):
    """
    x_init: (n,d) initial particles (DeviceArray)
    log_prob_fn: single-particle log density function f(x: (d,)) -> scalar
    returns particles after n_steps
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
def run_svgd_with_history(key, log_prob, n_steps=1000, lr=1e-3,record_every=10):

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


if __name__ == "__main__":
    # jax.config.update('jax_log_compiles', True)
    key = jax.random.PRNGKey(0)
    key, sub = jax.random.split(key)

    x0 = jax.random.normal(sub, (NUM_PARTICLE, 1, DIM_U)) * 3.0  # initialization

    # choosing the distribution
    log_prob = log_prob_gaussian_mix(dim=DIM_U, num_peaks = 4)
    
    # Compile and run
    new_theta, history = run_svgd_with_history(key, log_prob, n_steps=SVGD_ITER, lr=0.05)
    animate_svgd(history, log_prob)