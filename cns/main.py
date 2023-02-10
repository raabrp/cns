#!/usr/bin/env python3

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from tqdm import tqdm

from cns import vectorized_loss, init_params, update_params
from cns import sample_from_gaussian, measure_loss
from viz import (
    render_loss_function,
    render_loss_function_3d,
    render_pdf,
    render_samples,
)
from viz import Video


def main():
    title = "proof_of_concept"

    # environment
    N_iters = 100
    num_samples = 40
    learning_rate = 0.001
    noise = 0

    # distribution parameters
    # set to None for random, (goverened by seed)
    params = np.array([-1.5, -1.5, 1.0, 0.0, 1.0])
    seed = 0
    key, params = init_params(seed, params=params)

    # image parameters
    save_frames = [0, 20, 50]
    center = (0, 0)
    width = 12
    height = 12
    res = 1024

    # image of loss function in 3d
    ax = plt.axes(projection="3d")
    render_loss_function_3d(
        ax, vectorized_loss, center=center, width=width, height=height, res=res
    )
    plt.savefig(f"{title}_loss_3d.pdf")

    # video fig
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))

    # left, right subplot axes
    left, right = axs

    # setup left plot with loss function as background
    img, contour = render_loss_function(
        left, vectorized_loss, center=center, width=width, height=height, res=res
    )
    min_loss, max_loss = img.get_array().min(), img.get_array().max()
    norm = plt.Normalize(min_loss, max_loss)
    cmap = matplotlib.cm.get_cmap("bone")

    # setup right plot
    prev_max = None
    prev_min = None
    prev_loss = None
    prev_std = None

    # setup figure
    fig.tight_layout()

    with Video(title, fig) as video:
        for i in tqdm(range(N_iters)):
            key, samples = sample_from_gaussian(key, params, num_samples=num_samples)

            # plot samples and pdf on left axes
            visible_samples = samples[
                (center[0] - width / 2 < samples[:, 0])
                & (center[1] - height / 2 < samples[:, 1])
                & (samples[:, 0] < center[0] + width / 2)
                & (samples[:, 0] < center[1] + height / 2)
            ]
            e1, e2 = render_pdf(left, params)
            scatter = render_samples(left, visible_samples)

            # plot loss on right axes
            key, losses = measure_loss(key, samples, noise=noise)
            loss = np.mean(losses)
            l_min = np.min(losses)
            l_max = np.max(losses)
            l_std = np.std(losses)

            if i > 0:
                right.fill_between(
                    (i, i + 1),
                    (prev_loss - prev_std, loss - l_std),
                    y2=(prev_loss + prev_std, loss + l_std),
                    alpha=0.5,
                    color=cmap(norm((loss + prev_loss) / 2)),
                    linewidth=0,
                )

                right.plot(
                    (i, i + 1),
                    (prev_loss, loss),
                    color=cmap(norm((loss + prev_loss) / 2)),
                )

                right.plot(
                    (i, i + 1),
                    (prev_max, l_max),
                    color=cmap(norm((l_max + prev_max) / 2)),
                )

                right.plot(
                    (i, i + 1),
                    (prev_min, l_min),
                    color=cmap(norm((l_min + prev_min) / 2)),
                )

            prev_min = l_min
            prev_max = l_max
            prev_loss = loss
            prev_std = l_std

            right.set_xlim(1, N_iters)
            right.set_ylim(min_loss, max_loss)

            # draw to video frame
            video.draw()

            if i in save_frames:
                extent = left.get_window_extent().transformed(
                    fig.dpi_scale_trans.inverted()
                )
                fig.savefig(f"{title}_{i}.png", bbox_inches=extent.expanded(1.0, 1.0))

            # clear nonpersistent objects on left image
            for obj in [e1, e2, scatter]:
                obj.remove()

            key, params = update_params(
                key, params, samples, learning_rate=learning_rate
            )

    extent = right.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    fig.savefig(f"{title}_loss.pdf", bbox_inches=extent.expanded(1.1, 1.2))


if __name__ == "__main__":
    main()
