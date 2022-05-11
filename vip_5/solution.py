import os
import matplotlib.pyplot as plt
import numpy as np
from skimage import img_as_float
from skimage.segmentation import chan_vese

import algorithms


class Assignment:
    def __init__(self, image_dir):
        self.images = [np.round(plt.imread(image_dir + image_name) * 255.0) for image_name in next(os.walk(image_dir))[2]]
        # self.images = [np.round(plt.imread(image_dir + 'camera.png') * 255)]  # for testing 1 image
        self.labeled_k_2 = None
        self.labeled_k_large = None
        self.labeled_otsu = None

    def k_means(self):
        self.labeled_k_2 = []

        for i, orig_image in enumerate(self.images):
            print('\nImage', i+1)
            labeled_img = algorithms.k_means(orig_image, k_clusters=2, max_iters=300, retries=3, mode='all_pixels')
            self.labeled_k_2.append(labeled_img)

        algorithms.plot(self.images, self.labeled_k_2, cols=len(self.images), title='K-Means with K=2')

    def otsu(self):
        self.labeled_otsu = []

        for orig_image in self.images:
            threshold = algorithms.otsu(orig_image)

            labeled_img = np.copy(orig_image)
            labeled_img[labeled_img < threshold] = 0
            labeled_img[labeled_img >= threshold] = 1

            self.labeled_otsu.append(labeled_img)

        algorithms.plot(self.images, self.labeled_otsu, cols=len(self.images), title='Otsu')

    def denoising(self):
        denoised_images = []
        thresholds = [0.1, 0.6, 1]
        plot_labels = ['Original clustered image']

        for threshold in thresholds:
            print('\nDenoising clustered image with threshold =', threshold, '(8 neighbours, 2 passes)')
            plot_labels.append('Threshold = ' + str(threshold))
            denoised_img = algorithms.denoise(self.labeled_k_2[0], neighbours=8, threshold=threshold, passes=2)
            denoised_images.append(denoised_img)

        algorithms.plot([], [self.labeled_k_2[0]] + denoised_images, labels=plot_labels,
                        cols=2, title='Denoising with 8 neighbours, 2 passes and different thresholds')

    def denoising_passes(self):
        denoised_images = []
        passes = [1, 2, 5]
        plot_labels = ['Original clustered image']

        for passes in passes:
            print('\nDenoising clustered image with passes =', passes, '(8 neighbours, threshold fixed to 0.5)')
            plot_labels.append('Total denoising passes = ' + str(passes))
            denoised_img = algorithms.denoise(self.labeled_k_2[0], neighbours=8, threshold=0.5, passes=passes)
            denoised_images.append(denoised_img)

        algorithms.plot([], [self.labeled_k_2[0]] + denoised_images, labels=plot_labels,
                        cols=2, title='Denoising with 8 neighbours, 0.5 threshold and different passes')

    def k_means_with_larger_k(self):
        self.labeled_k_large = []

        for i, orig_image in enumerate(self.images):
            print('\nImage', i+1)
            labeled_img = algorithms.k_means(orig_image, k_clusters=4, max_iters=300, retries=3, mode='all_pixels')
            self.labeled_k_large.append(labeled_img)

        algorithms.plot(self.images, self.labeled_k_large, cols=len(self.images), title='K-Means with K=4')

    def chan_vese(self):
        print('\nStarting Chan-Vese segmentation w/ mu=0.2')
        plot_imgs = []
        plot_labels = ['Original image'] * len(self.images)

        for i, orig_image in enumerate(self.images):
            print('Image', i+1)
            cv = chan_vese(img_as_float(orig_image), mu=0.2, lambda1=1.5, lambda2=1.5, max_iter=300, extended_output=True)
            plot_imgs.extend(cv[:1])
            plot_labels.append('Chan-Vese segmentation - {} iterations'.format(len(cv[2])))

        algorithms.plot(self.images, plot_imgs, labels=plot_labels, cols=len(self.images))
