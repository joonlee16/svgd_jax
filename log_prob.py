import jax.numpy as jnp
import jax
import random

# Define single-particle log density for a n-component Gaussian mixture
def log_prob_gaussian_mix(dim, num_peaks):
    # x: (d,) array
    mus = [jnp.array([random.uniform(-1,1)*4, random.uniform(-1,1)*4]) for _ in range(num_peaks)]

    # Form the Gaussian mixture 
    def distribution(x):
        sigma = 0.6
        # per-component log densities (up to constant)
        norms = [-0.5 * jnp.sum((x - mu) ** 2) / (sigma ** 2) - (dim * 0.5) * jnp.log(2 * jnp.pi * sigma ** 2) for mu in mus]

        # log-sum-exp for mixture (weights 0.5,0.5)
        stacked = jnp.stack([norm + jnp.log(0.5) for norm in norms])
        return jax.scipy.special.logsumexp(stacked)
    return distribution