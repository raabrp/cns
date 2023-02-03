#!/usr/bin/env python3

import numpy as np

import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import patches
from matplotlib.transforms import Bbox, TransformedBbox

from cns import mean_from_params, cov_from_params


class Video:
    """
    Use a matplotlib figure to make a video.
    For each frame must:
      1. draw to figure
      2. call the video.draw method
      3. clear the figure/axes/Artists

    Example:

    fig, ax = plt.subplots(figsize=(6, 6))

    with Video('video_name', fig) as video:
        for _ in range(num_frames):
            render_to_fig()
            video.draw()
            ax.cla()
    """

    def __init__(self, title, fig, fps=10):
        self.video_file = title + ".mp4"

        # ffmpeg backend
        self.writer = animation.FFMpegWriter(
            fps=fps, metadata={"title": title, "artist": "Matplotlib"}
        )

        # canvas
        self.fig = fig

    def __enter__(self):
        # initialize writer
        self.writer.setup(self.fig, self.video_file, dpi=100)
        return self

    def draw(self):
        # save frame
        self.writer.grab_frame()

    def __exit__(self, exc_type, exc_value, exc_traceback):
        # write file and exit
        self.writer.finish()
        print(self.video_file)


class MockVideo:
    """
    Render each frame with plt.show instead
    """

    def __init__(self, title, fig, fps=10):
        self.fig = fig

    def __enter__(self):
        return self

    def draw(self):
        plt.show()

    def __exit__(self, exc_type, exc_value, exc_traceback):
        # write file and exit
        self.writer.finish()


def render_loss_function(ax, func, res=512):
    # render pdf on ax

    X = np.linspace(-2, 2, res)
    Y = np.linspace(-2, 2, res)
    xx, yy = np.meshgrid(X, Y)
    z = np.array(list(zip(xx.flatten(), yy.flatten())))
    f = func(z)
    zz = f.reshape((res, res))

    img = ax.imshow(zz, extent=(-2, 2, -2, 2))
    contour = ax.contour(xx, yy, zz, alpha=0.2, colors=["w"])

    return img, contour


def render_pdf(ax, params, res=30):
    # render pdf on ax
    mean = mean_from_params(params)
    cov = cov_from_params(params)

    # determine 1, 2 sigma levels
    eig_val, eig_vec = np.linalg.eig(cov)
    std_val = np.sqrt(eig_val)

    deg = np.arctan2(eig_vec[0][1], eig_vec[0][0]) * 180 / np.pi
    e1 = patches.Ellipse(mean, *(std_val * 2.0), angle=-deg, fill=False)
    e2 = patches.Ellipse(mean, *(std_val * 4.0), angle=-deg, fill=False)

    ax.add_patch(e1)
    ax.add_patch(e2)

    return e1, e2

    # unit_eig = eig_vec / norm(eig_vec)
    # s = std_val[0] * unit_eig[0]
    # ss = np.array([2.0 * s + mean, 1.0 * s + mean])
    # bnd = np.max(std_val) * 2

    # X = np.linspace(-bnd, bnd, res) + mean[0]
    # Y = np.linspace(-bnd, bnd, res) + mean[1]
    # xx, yy = np.meshgrid(X, Y)
    # z = np.array(list(zip(xx.flatten(), yy.flatten())))
    # pdf = gaussian_pdf(params, z)
    # zz = pdf.reshape((res, res))

    # levels = gaussian_pdf(params, ss)
    # ax.contour(xx, yy, zz, levels=levels)


def render_samples(ax, samples):
    # render samples on ax

    x = samples[:, 0]
    y = samples[:, 1]
    scatter = ax.scatter(x, y, s=2, color="w", alpha=0.5)

    return scatter
