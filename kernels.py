import jax.numpy as jnp
import jax
"""
Kernel implementations used by SVGD.

This module provides helper kernels and bandwidth heuristics used to
construct the Stein variational gradient. The public functions are:

- `sq_dists(theta)`: pairwise squared distances between rows of `theta`.
- `median_heur(theta_i)`: median heuristic for RBF bandwidth selection.
- `rbf_kernal(theta)`: RBF kernel matrix and its gradient w.r.t. inputs.

All functions are JIT-compiled with JAX for speed.
"""

# Helper function: compute the squared of matrices
# use (^2).sum - 2 x x^T + (x^2).sum^T
@jax.jit
def sq_dists(theta):
    x_norm = jnp.sum(theta * theta, axis=1, keepdims=True)  # (n,1)
    sq = x_norm + x_norm.T - 2.0 * (theta @ theta.T)
    return jnp.maximum(sq, 0.0)

# Helper function: median heuristics
@jax.jit
def median_heur(theta_i, eps=1e-6):
    n = max(theta_i.shape[0], 1.0)

    # Only look at the upper triangular elements in the matrix, since the matrix is symmetric.
    upper_triangule_ind = jnp.triu_indices(n, k=1)
    upper_triangle_values = sq_dists(theta_i)[upper_triangule_ind]
    return jnp.median(upper_triangle_values)/(jnp.log(n+eps)+eps)

# -------------------------
# RBF Kernel Implementation
# -------------------------
@jax.jit
def rbf_kernal(theta):    
    # Compute the kernel K 
    theta_sq= sq_dists(theta)
    # h -> small, makes each particle converge to each local minimia but slower convergence
    # h -> high, makes each particle converge to a fewer local miniia but faster convergence
    h = jnp.asarray(median_heur(theta))
    K = jnp.exp(-theta_sq/h)

    # compute the gradient grad_K
    theta_i = theta[:, None, :]  # (n,1,d)
    theta_j = theta[None, :, :]  # (1,n,d)
    diffs = theta_i - theta_j    # (n,n,d)
    coef = -2.0 / h
    grad_K = coef * diffs * K[..., None]  
    return K, grad_K