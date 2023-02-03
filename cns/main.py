#!/usr/bin/env python3

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.patches import Polygon
from tqdm import tqdm

from cns import vectorized_loss, init_params, update_params
from cns import sample_from_gaussian, measure_loss
from viz import render_loss_function, render_pdf, render_samples
from viz import Video


def main():
    title = "proof_of_concept"
    seed = 0
    N_iters = 80
    save_frames = [1, 20, 50]
    center = (0, 0)
    width = 6
    height = 6
    params = np.array([-1.5, -1.5, 0.5, 0.0, 0.5])  # set to None for random

    key, params = init_params(seed, params=params)

    # video fig
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))

    left, right = axs

    # setup left plot with loss function as background
    img, contour = render_loss_function(
        left, vectorized_loss, center=center, width=width, height=height, res=1024
    )
    min_loss, max_loss = img.get_array().min(), img.get_array().max()
    norm = plt.Normalize(min_loss, max_loss)
    cmap = matplotlib.cm.get_cmap("bone")

    # setup right plot
    prev_loss = None
    prev_std = None

    # setup figure
    fig.tight_layout()

    with Video(title, fig) as video:
        for i in tqdm(range(N_iters)):
            key, samples = sample_from_gaussian(key, params, num_samples=5)

            # plot samples and pdf on left axes
            visible_samples = samples[
                (center[0] - width / 2 < samples[:, 0])
                & (center[1] - height / 2 < samples[:, 1])
                & (samples[:, 0] < center[0] + width / 2)
                & (samples[:, 0] < center[1] + height / 2)
            ]
            scatter = render_samples(left, visible_samples)
            e1, e2 = render_pdf(left, params)

            # plot loss on right axes
            key, losses = measure_loss(key, samples)
            loss = np.mean(losses)
            std = np.std(losses)

            if i > 0:
                right.fill_between(
                    (i, i + 1),
                    (prev_loss - prev_std, loss - std),
                    y2=(prev_loss + prev_std, loss + std),
                    alpha=0.5,
                    color=cmap(norm((loss + prev_loss) / 2)),
                    linewidth=0,
                )

                right.plot(
                    (i, i + 1),
                    (prev_loss, loss),
                    color=cmap(norm((loss + prev_loss) / 2)),
                )

            prev_loss = loss
            prev_std = std

            right.set_xlim(1, N_iters)
            right.set_ylim(min_loss, max_loss)

            # draw to video frame
            video.draw()

            if i in save_frames:
                extent = left.get_window_extent().transformed(
                    fig.dpi_scale_trans.inverted()
                )
                fig.savefig(f"{title}_{i}.png", bbox_inches=extent.expanded(1.0, 1.0))
                pass

            # clear nonpersistent objects on left image
            for obj in [e1, e2, scatter]:
                obj.remove()

            key, params = update_params(key, params, samples, learning_rate=0.02)

    extent = right.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    fig.savefig(f"{title}_loss.pdf", bbox_inches=extent.expanded(1.1, 1.2))


if __name__ == "__main__":
    main()
