#!/usr/bin/env python3

# Objective: implement natural gradient descent in a gaussian distribution with
# non-convex objective function.

# 1. Using closed form Fisher and inverting.
# 2. Using MLE estimate of inverse Fisher matrix.


import jax
import jax.numpy as jnp
from jax.numpy import pi, einsum, sqrt, exp, log, sin, cos, e
from jax.numpy.linalg import det, inv
from jax.scipy.linalg import block_diag

# TODO put in class

# Loss #########################################################################


def loss_function(s):
    """
    A deterministic, non-convex loss fuction over strategies.
    (Not-Vectorized)

    Ackley

    https://en.wikipedia.org/wiki/Test_functions_for_optimization
    for more
    """
    x, y = s * 1

    # return x**2 + y**2

    a = -20 * exp(-0.2 * sqrt(0.5 * (x**2 + y**2)))
    b = -exp(0.5 * cos(2 * pi * x) + cos(2 * pi * y)) + e + 20
    return a + b


vectorized_loss = jax.vmap(loss_function)


def measure_loss(key, x, noise=0):
    """
    A (possibly noisy) wrapper for the loss function
    (Vectorized)

    Args:
        key: A jax pseudo random number generator key
        x: shape (n, 2) : points in R^2

    Returns:
    (
        A mutated key,
        losses: shape (n,)
    )
    """
    key, subkey = jax.random.split(key)

    noises = jax.random.normal(subkey, (x.shape[0],)) * noise

    losses = vectorized_loss(x) + noises

    return key, losses


# Parameterization #############################################################


def init_params(seed, params=None):
    """
    Randomly initialize parameters for a 2d gaussian distribution

    Args:
        seed: an integer

    Returns:
    (
        A mutated key,
        Parameters for a 2d gaussian distribution:
            [mu_x, mu_y, sqrt_sigma_xx, sigma_xy, sqrt_sigma_yy]
    )
    """

    key = jax.random.PRNGKey(seed)
    k = jax.random.split(key, 6)
    key = k[0]

    if params is None:
        mu_x = jax.random.uniform(k[1], minval=-0.5, maxval=0.5)
        mu_y = jax.random.uniform(k[2], minval=-0.5, maxval=0.5)
        sqrt_sigma_xx = jax.random.uniform(k[3], minval=0.0, maxval=1.0)
        sigma_xy = jax.random.uniform(k[4], minval=-0.5, maxval=0.5)
        sqrt_sigma_yy = jax.random.uniform(k[5], minval=0.0, maxval=1.0)

        params = jnp.array([mu_x, mu_y, sqrt_sigma_xx, sigma_xy, sqrt_sigma_yy])

    cov = cov_from_params(params)

    # assert positive semidefinite covariance matrix
    assert det(cov) > 0

    return key, params


def mean_from_params(params):
    """
    Args:
         params: An array
            [mu_x, mu_y, sqrt_sigma_xx, sigma_xy, sqrt_sigma_yy]

    Returns:
        mean: [mu_x, mu_y],
    """

    mu_x, mu_y, sqrt_sigma_xx, sigma_xy, sqrt_sigma_yy = params

    mean = jnp.array([mu_x, mu_y])

    return mean


def cov_from_params(params):
    """
    Args:
         params: An array
            [mu_x, mu_y, sqrt_sigma_xx, sigma_xy, sqrt_sigma_yy]

    Returns:
        cov [[sigma[xx], sigma[xy], [sigma[xy], sigma[yy]]]
    """

    mu_x, mu_y, sqrt_sigma_xx, sigma_xy, sqrt_sigma_yy = params

    cov = jnp.array(
        [
            [sqrt_sigma_xx**2, sigma_xy],
            [sigma_xy, sqrt_sigma_yy**2],
        ]
    )
    return cov


def grad_cov_from_params(params):
    mu_x, mu_y, sqrt_sigma_xx, sigma_xy, sqrt_sigma_yy = params

    return jnp.array(
        [
            [[2 * sqrt_sigma_xx, 0.0], [0.0, 0.0]],
            [[0.0, 1.0], [1.0, 0.0]],
            [[0.0, 0.0], [0.0, 2 * sqrt_sigma_yy]],
        ]
    )


def analytic_inv_fisher(params):
    """
    Analytically compute inverse Fisher matrix of 2d gaussian

    Args:
        params: An array
            [mu_x, mu_y, sqrt_sigma_xx, sigma_xy, sqrt_sigma_yy]
    """
    mu_x, mu_y, sqrt_sigma_xx, sigma_xy, sqrt_sigma_yy = params
    cov = cov_from_params(params)
    sigma_xx = cov[0][0]
    sigma_yy = cov[1][1]

    inv_cov = inv(cov)

    z = grad_cov_from_params(params)

    inv_A = (
        einsum(
            "im,ij,ajk,kl,blm->ab",
            jnp.eye(2),  # delta for trace
            inv_cov,
            z,
            inv_cov,
            z,
        )
        / 2
    )

    return block_diag(sigma_xx, sigma_yy, inv(inv_A))


# PDF / Sampling ###############################################################


def sample_from_gaussian(key, params, num_samples=1):
    """
    Sample `num` points from the 2d gaussian parameterized by `params`

    Args:
        key: A jax pseudo random number generator key
        params: An array
            [mu_x, mu_y, sqrt_sigma_xx, sigma_xy, sqrt_sigma_yy]
        num_samples: The number of points to sample

    Returns:
    (
        A mutated key,
        samples: an array of shape (num, 2)
    )
    """

    key, subkey = jax.random.split(key)

    mean = mean_from_params(params)
    cov = cov_from_params(params)

    sample = jax.random.multivariate_normal(key, mean, cov, shape=(num_samples,))
    return key, sample


def gaussian_pdf_single(params, sample):
    """
    Get probability density of multivariate gausian at a single sample
    (vectorization accomplished *after* automatic differentiation)

    Args:
        params: An array
            [mu_x, mu_y, sqrt_sigma_xx, sigma_xy, sqrt_sigma_yy]
        samples: an array of shape (2,) - a point in R^2

    Returns:
        density - the probability density at the sample
    """

    mean = mean_from_params(params)
    cov = cov_from_params(params)

    dev = sample - mean
    quad_fm = einsum("i,ij,j->", dev, inv(cov), dev)
    density = 1 / sqrt(det(2 * pi * cov)) * exp(-quad_fm / 2)

    return density


# vectorize second argument (sample)
gaussian_pdf = jax.vmap(gaussian_pdf_single, in_axes=(None, 0))


def log_density(params, sample):
    """
    (vectorization accomplished *after* automatic differentiation)
    """
    return log(gaussian_pdf_single(params, sample))


grad_log_density_per_sample = jax.vmap(jax.grad(log_density), in_axes=(None, 0))


def estimate_gradient(key, params, samples):
    """
    Estimate loss gradient of loss function with respect to parameters
    in terms of empirical samples

    Args:
        params: An array
            [mu_x, mu_y, sqrt_sigma_xx, sigma_xy, sqrt_sigma_yy]

    Returns:
    (
        key,
        estimated params gradient
    )
    """

    key, losses = measure_loss(key, samples)

    gradient_estimate = einsum(
        "i,ij->j", losses, grad_log_density_per_sample(params, samples)
    )

    return key, gradient_estimate


# Update #######################################################################


@jax.jit
def update_params(key, params, samples, learning_rate):
    """
    Perform Euler-discretized natural gradient desent on the loss function.

    Args:
        key: A jax pseudo random number generator key
        params: An array
            [mu_x, mu_y, sqrt_sigma_xx, sigma_xy, sqrt_sigma_yy]
        samples: TODO

    Returns:
    (
        key,
        params
    )
    """

    key, gradient_estimate = estimate_gradient(key, params, samples)

    inv_fisher = analytic_inv_fisher(params)

    covariant_derivative = einsum(
        "ij,j->i",
        inv_fisher,
        gradient_estimate,
    )
    # covariant_derivative = gradient_estimate

    new_params = params - covariant_derivative * learning_rate

    return key, new_params


if __name__ == "__main__":
    pass
