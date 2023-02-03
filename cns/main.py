#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from cns import vectorized_loss, init_params, update_params
from cns import sample_from_gaussian
from viz import render_loss_function, render_pdf, render_samples
from viz import Video


def main():
    seed = 0

    key, params = init_params(seed)
    params = np.array([-1.0, -1.0, 0.5, 0.0, 0.5])

    fig, ax = plt.subplots(figsize=(6, 6))

    img, contour = render_loss_function(ax, vectorized_loss)
    fig.tight_layout()

    with Video("test", fig) as video:
        for i in tqdm(range(200)):
            key, samples = sample_from_gaussian(key, params, num_samples=40)

            visible_samples = np.array(
                [s for s in samples if (-2 < s[0] < 2) and (-2 < s[1] < 2)]
            )

            scatter = render_samples(ax, visible_samples)
            e1, e2 = render_pdf(ax, params)

            video.draw()
            for obj in [e1, e2, scatter]:
                obj.remove()

            key, params = update_params(key, params, samples, 0.001)


if __name__ == "__main__":
    main()
