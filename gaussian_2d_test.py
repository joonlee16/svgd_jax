"""2D Gaussian mixture SVGD test and animation

This script runs a short SVGD example using the repository's
`run_svgd_with_history` function and `animate_svgd` helper.

It is intentionally short and usable as a quick smoke-test for the
2D Gaussian mixture functionality.
"""

import jax
import jax.numpy as jnp
from svgd import run_svgd_with_history
from helper_functions import animate_svgd
import random

# Define single-particle log density for a n-component Gaussian mixture model
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


def main():
    """Run the 2D Gaussian mixture SVGD demo.

    Initializes a PRNGKey, constructs a random mixture, runs SVGD while
    recording the particle history, and launches an animation displaying the
    particles and the underlying density.
    """
    key = jax.random.PRNGKey(42)
    key, sub = jax.random.split(key)

    # Create a random 2D Gaussian mixture log-probability 
    log_prob = log_prob_gaussian_mix(dim=2, num_peaks=2)
    # log_prob = log_prob_gaussian_mix(dim=2, num_peaks=4)

    _, history = run_svgd_with_history(sub, log_prob, n_steps=300, lr=0.05, record_every=10)
    # history shape: (n_records, NUM_PARTICLE, DIM_U)

    # animate
    animate_svgd(history, log_prob)


if __name__ == '__main__':
    main()
