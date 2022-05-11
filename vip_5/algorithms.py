import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from numpy.random import default_rng


def k_means(data, k_clusters, max_iters=500, retries=3, mode='all_pixels'):
    if mode not in ['all_pixels', 'hist']:
        raise ValueError('Mode can only be of value "all_pixels" (handle each intensity as data) or ' +
                         '"hist" (handle intensity counts as data')

    if mode == 'all_pixels':
        X = np.copy(data).reshape((-1, 1))
    else:
        intensities, counts = np.unique(data, return_counts=True)
        X = counts.reshape((-1, 1)) / counts.max()

    cluster_sets = []
    label_sets = []
    retry = 0

    while retry < retries:
        print('K-Means try {}/{} (max iterations: {})'.format(retry + 1, retries, max_iters), flush=True)
        clusters = np.random.randn(k_clusters, X.shape[1]) * np.std(X, axis=0) + np.mean(X, axis=0)

        cluster_assignments = np.ones(X.shape[0]) * -1
        old_assignments = np.ones(X.shape[0]) * -1

        clusters_collapsed = False

        for _ in range(max_iters):
            cluster_assignments = np.array([np.linalg.norm(X - cluster, axis=1) for cluster in clusters]).argmin(axis=0)

            if (cluster_assignments == old_assignments).all():
                break

            clusters_new = np.copy(clusters)

            for i in range(len(clusters)):
                cluster_related_points = X[cluster_assignments == i]

                if len(cluster_related_points):
                    clusters_new[i] = np.mean(cluster_related_points, axis=0)
                else:
                    print('One or more clusters collapsed into one, retrying again...')
                    clusters_collapsed = True
                    break

            clusters = clusters_new

            if clusters_collapsed:
                break

        if not clusters_collapsed:
            cluster_sets.append(clusters)
            label_sets.append(cluster_assignments)
            retry += 1

    wcss_set = []

    print('Choosing best set of clusters from all retries...')
    for i, (clusters, labels) in enumerate(zip(cluster_sets, label_sets)):
        cluster_dists = [np.sum(np.sum((X[labels == i+1] - clusters[i]) ** 2, axis=1)) for i in range(len(clusters))]
        wcss_set.append(sum(cluster_dists))

    optimal_set_index = np.array(wcss_set).argmin()
    print('Best cluster centroids chosen from try #', optimal_set_index + 1)

    if mode == 'all_pixels':
        clustered_img = label_sets[optimal_set_index].reshape(data.shape)
    else:
        clustered_img = np.copy(data)

        for i, intensity in enumerate(intensities):
            clustered_img[clustered_img == intensity] = label_sets[optimal_set_index][i]

    return clustered_img


def otsu(image):
    data = np.copy(image).reshape(-1)

    counts, bins = np.histogram(data, bins=255, range=[0, 255])
    probs = counts / data.size
    wwcvs = []  # computed weighted within-class variances

    for t in range(256):
        weight_a = sum(probs[:t])

        if weight_a == 0:
            wwcvs.append(np.nan)
            continue

        mean_a = sum([i * probs[i] / weight_a for i in range(t)])
        variance_a = sum([((i - mean_a) ** 2) * probs[i] / weight_a for i in range(t)])

        weight_b = sum(probs[t:])

        if weight_b == 0:
            wwcvs.append(np.nan)
            continue

        mean_b = sum([i * probs[i] / weight_b for i in range(t, 255)])
        variance_b = sum([((i - mean_b) ** 2) * probs[i] / weight_b for i in range(t, 255)])

        wwcvs.append(weight_a * variance_a + weight_b * variance_b)

    return np.nanargmin(wwcvs)


def denoise(labeled_img, neighbours=4, threshold=0.5, passes=1):
    if neighbours not in [4, 8]:
        raise ValueError('neighbours= parameter can only be 4 or 8!')

    if threshold < 0 or threshold > 1:
        raise ValueError('threshold= parameter can only be in range [0,1]')

    lbl_data = np.copy(labeled_img)
    lbl_data = np.pad(lbl_data, (1, 1), constant_values=-1)
    lbl_final = np.ones(labeled_img.shape) * -1

    for p in range(passes):
        print('Denoising pass {}/{}'.format(p + 1, passes))
        for i in range(1, lbl_data.shape[0] - 1):
            for j in range(1, lbl_data.shape[1] - 1):
                if neighbours == 4:
                    neighbours_array = \
                        np.array([lbl_data[i, j-1], lbl_data[i, j+1], lbl_data[i-1, j], lbl_data[i+1, j]]).astype('float')
                else:
                    neighbours_array = lbl_data[i-1 : i+2, j-1 : j+2].astype('float')
                    neighbours_array[1, 1] = np.nan

                neighbours_array = neighbours_array[np.logical_and(neighbours_array != -1, ~np.isnan(neighbours_array))]

                classes, counts = np.unique(neighbours_array, return_counts=True)

                percentages = counts / neighbours_array.size
                max_percent_ind = percentages.argmax()

                lbl_final[i-1, j-1] = classes[max_percent_ind] if percentages[max_percent_ind] >= threshold else lbl_data[i, j]

        lbl_data = np.pad(lbl_final, (1, 1), constant_values=-1)

    return lbl_final


def plot(images, cluster_images, labels=None, cols=1, fontsize=12, figsize=(20, 20), title=None):
    if labels is None:
        labels = []

    all_images = images + cluster_images

    figure, axes = plt.subplots(int(np.ceil(len(all_images) / cols)), cols, squeeze=False, constrained_layout=True)
    figure.set_size_inches(*figsize)
    max_index = 0

    for i, image in enumerate(all_images):
        ax = axes[i // cols][i % cols]

        if i < len(images):
            ax.imshow(image, cmap='gray')
        else:
            clusters = np.unique(image)
            img_plot = ax.imshow(image, cmap='gist_gray')

            colors = [img_plot.cmap(img_plot.norm(value)) for value in clusters]
            patches = [mpatches.Patch(color=colors[i], label="Cluster {}".format(int(clusters[i]))) for i in range(len(clusters))]

            ax.legend(handles=patches, borderaxespad=0, fontsize=8)

        ax.axis('off')

        try:
            ax.set_title(labels[i], fontsize=fontsize)
        except IndexError:
            pass

        max_index = i

    if len(images) % cols != 0:
        for empty_index in range((max_index % cols) + 1, cols):
            figure.delaxes(axes[max_index // cols][empty_index])

    if title is not None:
        figure.suptitle(title)
