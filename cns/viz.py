#!/usr/bin/env python3

import numpy as np

# from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import patches

from cns import mean_from_params, cov_from_params, loss_function


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

    def __init__(self, title, fig, fps=15):
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

    def __init__(self, title, fig, fps=None):
        self.fig = fig

    def __enter__(self):
        return self

    def draw(self):
        plt.show()

    def __exit__(self, exc_type, exc_value, exc_traceback):
        # write file and exit
        self.writer.finish()


def render_loss_function(ax, func, center=(0, 0), width=4, height=4, res=512):
    # render pdf on ax

    left = center[0] - width / 2
    right = center[0] + width / 2
    bottom = center[1] - width / 2
    top = center[1] + width / 2
    X = np.linspace(left, right, res)
    Y = np.linspace(bottom, top, res)
    xx, yy = np.meshgrid(X, Y)
    z = np.array(list(zip(xx.flatten(), yy.flatten())))
    f = func(z)
    zz = f.reshape((res, res))

    img = ax.imshow(zz, extent=(left, right, bottom, top), cmap="bone")
    contour = ax.contour(xx, yy, zz, alpha=0.2, colors=["w"])

    return img, contour


def render_loss_function_3d(ax, func, center=(0, 0), width=4, height=4, res=512):
    left = center[0] - width / 2
    right = center[0] + width / 2
    bottom = center[1] - width / 2
    top = center[1] + width / 2
    X = np.linspace(left, right, res)
    Y = np.linspace(bottom, top, res)
    xx, yy = np.meshgrid(X, Y)
    z = np.array(list(zip(xx.flatten(), yy.flatten())))
    f = func(z)
    zz = f.reshape((res, res))

    img3d = ax.plot_surface(xx, yy, zz, cmap="bone")
    img = ax.contourf(xx, yy, zz, cmap="bone", zdir="z", offset=0)
    contour = ax.contour(xx, yy, zz, alpha=0.2, colors=["w"], zdir="z", offset=0)

    ax.view_init(elev=12, azim=30)

    return img3d, img, contour


def render_pdf(ax, params, res=30):
    # render probability density function on ax
    mean = mean_from_params(params)
    cov = cov_from_params(params)

    # determine 1, 2 sigma levels
    eig_val, eig_vec = np.linalg.eig(cov)
    std_val = np.sqrt(eig_val)

    deg = np.arctan2(eig_vec[0][1], eig_vec[0][0]) * 180 / np.pi
    e1 = patches.Ellipse(
        mean,
        *(std_val * 2.0),
        angle=-deg,
        facecolor="w",
        edgecolor="k",
        fill=True,
        alpha=0.3
    )
    e2 = patches.Ellipse(
        mean,
        *(std_val * 4.0),
        angle=-deg,
        facecolor="w",
        edgecolor="k",
        fill=True,
        alpha=0.3
    )

    ax.add_patch(e1)
    ax.add_patch(e2)

    return e1, e2


def render_samples(ax, samples):
    # render samples on ax

    x = samples[:, 0]
    y = samples[:, 1]
    scatter = ax.scatter(x, y, s=40, color="white", edgecolors="black", alpha=0.9)

    return scatter
