#!/usr/bin/env python3

import jax
from jax import numpy as np
from jax.numpy import pi, sqrt, exp, log, sin, einsum

from tqdm import tqdm


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


def walk_brownian(key, var, p=0.3):
    """
    Generate Brownian motion sampled at heterogeneously spaced times.
    Args:
        key: jax pseudo-random number generator key
        var: time-varying variance of random walk (n values) (for unit time interval).
        p: uniform probability that W(t) is observed for each t

    Returns:
        W: the observed values, with prepended 0 value at t=0,
           so one more than lenght of var
        t: the corresponding times
    """

    # n gaps between data implies we generate n+1 values,
    # then downsample at random
    n = len(var)

    key, key1, key2 = jax.random.split(key, 3)

    dW = jax.random.normal(key1, shape=(n,)) * sqrt(var)
    W = np.cumsum(dW)
    t = np.arange(n) + 1  # t=0 starts at W=0

    # whether each point (after t=0) is observed
    is_obs = np.concatenate(
        [
            jax.random.bernoulli(key2, p=p, shape=(n - 1,)),
            np.array([True]),  # guarantee we see the last sample
        ]
    )

    # return mutated key and observed W, corresponding t
    return (
        key,
        np.concatenate([np.array([0.0]), W[is_obs]]),
        np.concatenate([np.array([0]), t[is_obs]]),
    )


def gaussian_mu_var(params, sample):
    """
    Get probability density for single sample according to
    normal distribution wth params (mu, var)

    We use this function for *both*
    + the log probabilty (sample = h | theta) AND
    + the log likelihood loss (sample = x | h)

    (vectorization accomplished *after* automatic differentiation)
    """

    mu, var = params

    density = 1 / sqrt(2 * pi * var) * exp(-((sample - mu) ** 2) / (var * 2))

    return density


def log_gaussian_density_mu_log_sigma(params, sample):
    """
    log probability of gaussian, where params encodes
    (mu, log_sigma)

    Args:
        params_theta: (mu, log_simga)
        sample: a float (h)
    """
    mu, log_sigma = params

    # convert to types of parameters expected by gaissiam_mu_var
    mu_var_params = np.array([mu, exp(log_sigma * 2)])

    return log(gaussian_mu_var(mu_var_params, sample))


def log_gaussian_density_zero_log_sigma(log_sigma, sample):
    """
    log probability of gaussian

    Args:
        params: log_sigma
        sample: a float (x)
    """

    mu = 0.0

    params = np.array([mu, log_sigma])

    return log_gaussian_density_mu_log_sigma(params, sample)


# vectorize first argument (log_sigma)
# for calculating log likelihood of fixed x in terms of vector h
get_log_likelihood_of_vector_h_given_x = jax.vmap(
    log_gaussian_density_zero_log_sigma, in_axes=(0, None)
)

# get grad wrt parameters (theta) over vector of samples (h)
# first grad, then vmap
get_theta_grads_log_p_vector_h = jax.vmap(
    jax.grad(log_gaussian_density_mu_log_sigma), in_axes=(None, 0)
)


@jax.jit
def update_theta(key, theta, x, learning_rate):
    """
    Learn a gaussian distribution parameterized by theta = [mu, log_sigma]

    for h, where h is log_sigma for another guassian with zero mean
    from which x is ostensibly sampled
    """

    num_h_samples = 40

    key, subkey = jax.random.split(key)

    # LEART GAUSSIAN OVER H

    # mean and std of distribution of target paramter
    # We use theta_log_sigma to avoid learning negative values for sigma
    theta_mu, theta_log_sigma = theta

    # sample hypotheses h from gaussian parameterized by theta
    vector_h = (
        jax.random.normal(subkey, shape=(num_h_samples,)) * exp(theta_log_sigma)
        + theta_mu
    )

    # gradient in theta of log probability associated with sampling this h
    # is vector corresponding to h
    theta_grads = get_theta_grads_log_p_vector_h(theta, vector_h)

    # get loss = -log p(x | h) for h, clipped
    # is vector corresponding to h
    log_loss_for_h_given_x = np.clip(
        -get_log_likelihood_of_vector_h_given_x(vector_h, x),
        -100.0,
        100.0,
    )

    # gradient estimate in theta for Monte Carlo h samples
    gradient_estimate_theta = einsum("ij,i->j", theta_grads, log_loss_for_h_given_x)

    # of gauss distribution parameterized by [mu, log_sigma]
    inv_fisher = np.array([[exp(theta_log_sigma * 2), 0.0], [0.0, 0.5]])

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

    theta = np.array([init_mu_over_log_sigma, init_log_sigma_over_log_sigma])

    mean_over_log_sigma, sigma_over_log_sigma = theta[0], exp(theta[1])

    # store history
    hist_mean_over_log_sigma = []
    hist_sigma_over_log_sigma = []

    hist_mean_over_log_sigma.append(mean_over_log_sigma)
    hist_sigma_over_log_sigma.append(sigma_over_log_sigma)

    # iterate over observations x in the data
    for xx in tqdm(x):
        key, theta = update_theta(key, theta, xx, learning_rate)

        mean_over_log_sigma, sigma_over_log_sigma = theta[0], exp(theta[1])

        hist_mean_over_log_sigma.append(mean_over_log_sigma)
        hist_sigma_over_log_sigma.append(sigma_over_log_sigma)

    return (
        key,
        x,  # n - 1, where n is length of W, t (which include (0, 0))
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

    # figl, left = plt.subplots(1, 1, figsize=(6, 6))
    # figc, center = plt.subplots(1, 1, figsize=(6, 6))
    # figr, right = plt.subplots(1, 1, figsize=(6, 6))
    fig, axs = plt.subplots(1, 3, figsize=(18, 6))
    left, center, right = axs

    # LEFT PLOT IS FOR W SPACE
    # CENTER PLOT IS FOR DELTA W / sqrt(DELTA T) SPACE
    # RIGHT PLOT IS LOG-SIGMA SPACE

    # the real value for the target parameter (std) in time
    # generate variance as function of time
    log_sigma = sin(np.arange(time_window - 1) / (time_window - 1) * 4 * pi) * 0.5 + 0.5
    sigma = exp(log_sigma)
    var = sigma**2

    learning_rate = 0.01

    key = jax.random.PRNGKey(seed)

    # target is std of brownian motion
    # use variance
    key, W, t = walk_brownian(key, var, p=obs_prob)

    left.plot(t, W, color="k", alpha=0.5)
    left.scatter(t, W, s=40, color="white", edgecolors="black", alpha=0.9)

    ma, mi = W.max(), W.min()

    for tt in t:
        left.plot([tt, tt], [ma, mi], color="k", alpha=0.1)

    key, observations, hist_mean_over_log_sigma, hist_sigma_over_log_sigma = estimate(
        key,
        W,
        t,
        init_mu_over_log_sigma=0.5,
        init_log_sigma_over_log_sigma=-1.0,
        learning_rate=learning_rate,
    )

    data = np.concatenate([observations, sigma])

    ma, mi = data.max(), data.min()

    # no deltas yet at t = 0

    center.scatter(
        t[1:], observations, s=40, color="white", edgecolors="black", alpha=0.9
    )
    for tt in t[1:]:
        center.plot([tt, tt], [ma, mi], color="k", alpha=0.1)

    center.plot(
        sigma,
        label=r"$\pm\sigma$",
        color="k",
        linewidth=2,
        linestyle="--",
    )
    center.plot(
        -sigma,
        color="k",
        linewidth=2,
        linestyle="--",
    )

    # start with a prior at t = 0.

    # TODO check
    hist_mean_over_sigma, hist_sigma_over_sigma = mean_var_galton(
        hist_mean_over_log_sigma, hist_sigma_over_log_sigma
    )

    center.legend(loc="upper right", fontsize=20)

    right.plot(
        log_sigma,
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

    # figl.savefig("left_wiener.pdf")
    # figc.savefig("center_wiener.pdf")
    # figr.savefig("right_wiener.pdf")
    plt.show()


if __name__ == "__main__":
    main()
