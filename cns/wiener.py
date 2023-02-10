#!/usr/bin/env python3

import jax
from jax import numpy as np
from jax.numpy import pi, sqrt, exp, log, cos, einsum

from tqdm import tqdm

from viz import render_heterogeneous_time_series
from cns import gaussian_pdf_single

import matplotlib as mpl
import matplotlib.pyplot as plt


mpl.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": "Nimbus Roman",
        "mathtext.fontset": "cm",
        "mathtext.rm": "serif",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)


def walk_brownian(var, seed=0, p=0.3):
    """
    Generate Brownian motion sampled at heterogeneously spaced times.
    Args:
        seed: random seed (int)
        var: time-varying variance of random walk (n values)
        p: uniform probability that W(t) is observed for each t

    Returns:
        W: the observed values, with prepended 0 value at t=0,
           so one more than lenght of var
        t: the corresponding times
    """
    n = len(var)

    key = jax.random.PRNGKey(seed)
    key1, key2 = jax.random.split(key)

    dW = jax.random.normal(key1, shape=(n,)) * sqrt(var)
    W = np.cumsum(dW)
    t = np.arange(n) + 1
    is_obs = np.concatenate(
        [
            jax.random.bernoulli(key2, p=p, shape=(n - 1,)),
            np.array([True]),  # guarantee we see the last sample
        ]
    )

    return (
        np.concatenate([np.array([0.0]), W[is_obs]]),
        np.concatenate([np.array([0.0]), t[is_obs]]),
    )


def gaussian_mu_var(params, sample):
    """
    Get probability density for single sample according to
    normal distribution wth params (mu, var)

    We use this function for *both*
    + the log probabilty (sample = h) AND
    + the log likelihood loss (sample = x)

    (vectorization accomplished *after* automatic differentiation)
    """

    mu, var = params

    density = 1 / sqrt(2 * pi * var) * exp(-((sample - mu) ** 2) / (var * 2))

    return density


def log_gaussian_density(params_h, sample_x):
    return log(gaussian_mu_var(params_h, sample_x))


def log_gaussian_density_using_log_sigma(params_theta, sample_h):
    mu, log_sigma = params_theta

    mu_var_params = np.array([mu, exp(log_sigma)])

    return log(gaussian_mu_var(mu_var_params, sample_h))


# vectorize first argument (vector h) for calculating log likelihood of x
get_log_gauss_x_param_h = jax.vmap(log_gaussian_density, in_axes=(0, None))

# get grad wrt parameters (theta) over vector of samples (h)
# first grad, then vmap
get_grad_log_gauss_h_param_theta = jax.vmap(
    jax.grad(log_gaussian_density_using_log_sigma), in_axes=(None, 0)
)


@jax.jit
def update_params(key, theta, x, learning_rate):
    """

    Lean a gaussian distribution parameterized by theta = [mu, log_sigma]

    for h, where h is sigma for another guassian with zero mean
    from which x is ostensibly sampled
    """

    num_h_samples = 40

    key, subkey = jax.random.split(key)

    # LEART GAUSSIAN OVER H

    # mean and std of distribution of target paramter
    # We use theta_log_sigma to avoid learning negative values for sigma
    theta_mu, theta_log_sigma = theta

    # sample hypotheses h from gaussian parameterized by theta
    h_as_sample = (
        jax.random.normal(subkey, shape=(num_h_samples,)) * exp(theta_log_sigma)
        + theta_mu
    )

    # gradient in theta of log probability associated with sampling this h
    grad_log_gauss_h_param_theta = get_grad_log_gauss_h_param_theta(theta, h_as_sample)

    # H SPACE IS LOG_SIGMA_OF_GAUSS_OF_X SPACE

    # unpack h as [0, sigma^2] to parameterize Gaussian
    h_as_gauss_params = np.hstack(
        [
            np.zeros((num_h_samples, 1)),  # mu_model
            exp(h_as_sample * 2).reshape(  # H = LOG SIGMA, GAUSS_PARAMETERIZED_WITH_VAR
                num_h_samples, 1
            ),
        ]
    )

    # get loss = -log p(x | h) for h, clipped
    h_log_loss_for_x = np.clip(
        -get_log_gauss_x_param_h(h_as_gauss_params, x),
        -100.0,
        100.0,
    )

    # gradient in theta
    gradient_estimate_theta = einsum(
        "i,ij->j", h_log_loss_for_x, grad_log_gauss_h_param_theta
    )

    # of gauss distribution parameterized by [mu, log_sigma]
    inv_fisher = np.array(
        [[exp(theta_log_sigma * 2), 0.0], [0.0, exp(theta_log_sigma) / 2]]
    )

    covariant_derivative = einsum("ij,j->i", inv_fisher, gradient_estimate_theta)

    new_params = theta - covariant_derivative * learning_rate

    return key, new_params


def estimate(
    key,
    W,
    t,
    init_mu_over_log_sigma=0.0,
    init_log_sigma_over_log_sigma=0.0,
    learning_rate=0.1,
):
    """
    Args:
        key: jax random key
        W: sampled values of Wiener process with unknown, time-varying sigma
        t: times of samples

        init_mu_galt for galton distribution
        init_log_sigma_galt for galton distribution

    Returns:
        key
        empirical sample of log(sigma^2) for each time step

        learned estimates for mean of log(sigma^2)
        learned estimates for var of log(sigma^2)
    """
    n = len(W)
    assert n == len(t)
    dW = W[1:] - W[:-1]
    dt = t[1:] - t[:-1]

    # empirical sample, distributed like N(0, sigma^2)
    x = dW / sqrt(dt)

    # params represent theta:
    # mean of gaussian over log_sigma
    # log_std of gaussian over log_sigma
    params = np.array([init_mu_over_log_sigma, init_log_sigma_over_log_sigma])

    mean_over_log_sigma, sigma_over_log_sigma = params[0], exp(params[1])

    # store history
    hist_mean_over_log_sigma = []
    hist_sigma_over_log_sigma = []

    hist_mean_over_log_sigma.append(mean_over_log_sigma)
    hist_sigma_over_log_sigma.append(sigma_over_log_sigma)

    # iterate over observations x in the data
    for i in tqdm(range(len(x))):
        key, params = update_params(key, params, x[i], learning_rate)

        mean_over_log_sigma, sigma_over_log_sigma = params[0], exp(params[1])

        hist_mean_over_log_sigma.append(mean_over_log_sigma)
        hist_sigma_over_log_sigma.append(sigma_over_log_sigma)

    return (
        key,
        x,  # n - 1, where n is length of W, t
        np.array(hist_mean_over_log_sigma),  # n
        np.array(hist_sigma_over_log_sigma),  # n
    )


seed = 0


def mean_var_galton(mu, sigma):
    mean = exp(mu + sigma**2 / 2.0)
    var = exp((sigma**2 - 1.0) * (2 * mu + sigma**2))
    return mean, var


def main():
    time_window = 1024
    obs_prob = 0.5

    figl, left = plt.subplots(1, 1, figsize=(6, 6))
    figc, center = plt.subplots(1, 1, figsize=(6, 6))
    figr, right = plt.subplots(1, 1, figsize=(6, 6))

    # LEFT PLOT IS FOR W SPACE
    # CENTER PLOT IS FOR DELTA W / sqrt(DELTA T) SPACE
    # RIGHT PLOT IS LOG-SIGMA SPACE

    # the real value for the target parameter (std) in time
    # generate variance as function of time
    log_sigma = cos(np.arange(time_window - 1) / (time_window - 1) * pi) ** 2
    sigma = exp(log_sigma)
    var = sigma**2

    learning_rate = 0.001

    # target is std of brownian motion
    # use variance
    W, t = walk_brownian(var, seed=seed, p=obs_prob)
    render_heterogeneous_time_series(left, W, t)

    key = jax.random.PRNGKey(seed)

    key, observations, hist_mean_over_log_sigma, hist_sigma_over_log_sigma = estimate(
        key,
        W,
        t,
        init_mu_over_log_sigma=log_sigma[0],
        init_log_sigma_over_log_sigma=0.0,
        learning_rate=learning_rate,
    )

    # hist_mean_over_log_sigma = hist_mean_over_log_sigma - 0.5

    data = np.concatenate([observations, sigma])

    ma, mi = data.max(), data.min()

    # no deltas yet at t = 0

    center.plot(
        sigma,
        label="$\sigma$",
        color="k",
        linewidth=2,
        linestyle="--",
    )
    center.scatter(
        t[1:], observations, s=40, color="white", edgecolors="black", alpha=0.9
    )
    for tt in t[1:]:
        center.plot([tt, tt], [ma, mi], color="k", alpha=0.1)

    # start with a prior at t = 0.

    hist_mean_over_sigma, hist_sigma_over_sigma = mean_var_galton(
        hist_mean_over_log_sigma, hist_sigma_over_log_sigma
    )

    center.legend(loc="upper right", fontsize=20)

    right.plot(
        log(sigma),
        label=r"$\log \sigma$",
        color="k",
        linewidth=2,
        linestyle="--",
    )

    right.fill_between(
        t,
        (hist_mean_over_log_sigma - hist_sigma_over_log_sigma),
        y2=(hist_mean_over_log_sigma + hist_sigma_over_log_sigma),
        alpha=0.5,
        color="k",
        linewidth=0,
    )

    right.plot(
        t,
        hist_mean_over_log_sigma,
        color="k",
        linewidth=2,
        label=r"${\rm E}[\log(\Sigma)]}$",
    )
    right.legend(loc="upper right", fontsize=20)

    figl.savefig("left_wiener.pdf")
    figc.savefig("center_wiener.pdf")
    figr.savefig("right_wiener.pdf")


if __name__ == "__main__":
    main()
