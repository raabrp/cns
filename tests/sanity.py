#!/usr/bin/env python3


import numpy as np
from jax.numpy import pi, log, e
from numpy.linalg import det, norm

import matplotlib as mpl
from matplotlib import patches
import matplotlib.pyplot as plt

from cns import init_params, mean_from_params, cov_from_params
from cns import sample_from_gaussian, gaussian_pdf

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


def sanity_check_sampling(seed=0, num_samples=1000, res=30, params=None):
    # monte-carlo integration of pdf p (gaussian_pdf)
    # using samples from q (sample_from_gaussian)
    # and calculation of H_q(p) - H(p)

    if params is None:
        key, params = init_params(seed)
    else:
        key, _params = init_params(seed)

    mean = mean_from_params(params)
    cov = cov_from_params(params)
    key, samples = sample_from_gaussian(key, params, num_samples)
    densities = gaussian_pdf(params, samples)

    pdf_entropy = log(det(2 * pi * e * cov)) / 2

    cross_entropy = -np.mean(log(densities))

    not_kl_divergence = cross_entropy - pdf_entropy

    print("Should be close to zero:", not_kl_divergence)

    # prepare plot axes
    fig, ax = plt.subplots(figsize=(6, 6))

    # analytic sigma ellipses
    eig_val, eig_vec = np.linalg.eig(cov)
    ellipse_axes = np.einsum(
        "i,ij,i,j->ij",
        np.sqrt(eig_val),
        eig_vec,
        1 / norm(eig_vec, axis=-1),
        np.array(
            [1.0, -1.0]
        ),  # TODO why is y backwards? Also for ellipse, deg is negative...
    )

    std_val = np.sqrt(eig_val)

    deg = np.arctan2(eig_vec[0][1], eig_vec[0][0]) * 180 / np.pi
    e3 = patches.Ellipse(
        mean, *(std_val * 6.0), angle=-deg, fill=True, color="pink", alpha=0.2
    )
    e2 = patches.Ellipse(
        mean, *(std_val * 4.0), angle=-deg, fill=True, color="blue", alpha=0.2
    )
    e1 = patches.Ellipse(
        mean, *(std_val * 2.0), angle=-deg, fill=True, color="red", alpha=0.2
    )
    ax.add_patch(e1)
    ax.add_patch(e2)
    ax.add_patch(e3)

    # sample pdf on grid and display as contour
    plt.arrow(*mean, *ellipse_axes[0], color="red")
    plt.arrow(*mean, *ellipse_axes[1], color="red")
    s = ellipse_axes[0]

    # plt.arrow(*mean, *s)
    ss = np.array([3.0 * s + mean, 2.0 * s + mean, 1.0 * s + mean])
    bnd = np.max(std_val) * 3

    X = np.linspace(-bnd, bnd, res) + mean[0]
    Y = np.linspace(-bnd, bnd, res) + mean[1]
    xx, yy = np.meshgrid(X, Y)
    z = np.array(list(zip(xx.flatten(), yy.flatten())))
    pdf = gaussian_pdf(params, z)
    zz = pdf.reshape((res, res))

    levels = gaussian_pdf(params, ss)
    ax.contour(xx, yy, zz, levels=levels)

    # use for numerical integration
    print("Should be just less than 1:", np.sum(zz) * 4 * bnd**2 / (res**2))

    # scatter plot of samples (on top of pdf)
    x = samples[:, 0]
    y = samples[:, 1]
    ax.scatter(x, y, s=1, color="k", alpha=0.1)

    plt.margins(x=0, y=0)
    plt.grid()
    plt.show()


if __name__ == "__main__":
    sanity_check_sampling(
        seed=1,
        num_samples=100000,
        res=1000,
    )
    # sanity_check_sampling(
    #     seed=0,
    #     num_samples=100000,
    #     res=1000,
    #     params=np.array([0.0, 0.0, 1.0, 0.6, np.sqrt(2.0)]),
    # )
