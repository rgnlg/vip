import matplotlib.pyplot as plt
import numpy as np

from matplotlib.colors import LightSource


def plot(images, labels=None, cols=1, fontsize=6, figsize=(20, 20)):
    if images is None or not len(images):
        raise ValueError('No images list passed or list is empty')

    if labels is None:
        labels = []

    figure, axes = plt.subplots(int(np.ceil(len(images) / cols)), cols, squeeze=False, constrained_layout=True)
    figure.set_size_inches(*figsize)

    max_index = 0

    for i, image in enumerate(images):
        ax = axes[i // cols][i % cols]
        ax.imshow(image, cmap='gray')
        ax.axis('off')

        try:
            ax.set_title(labels[i], fontsize=fontsize)
        except IndexError:
            pass

        max_index = i

    if len(images) % cols != 0:
        for empty_index in range((max_index % cols) + 1, cols):
            figure.delaxes(axes[max_index // cols][empty_index])

    plt.show()


def display_surface_matplotlib_fixed(z):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    x, y = np.mgrid[:z.shape[0], :z.shape[1]]

    shades = np.copy(z)
    shades[np.isnan(shades)] = 0
    grey_vals = LightSource(azdeg=-60, altdeg=25.0).shade(shades, cmap=plt.get_cmap('gray'))

    ax.plot_surface(x, y, z, rstride=1, cstride=1, linewidth=0, antialiased=False, facecolors=grey_vals)
