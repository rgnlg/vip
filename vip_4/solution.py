from sklearn.cluster import MiniBatchKMeans
from sklearn.model_selection import train_test_split
from collections import Counter

import pandas as pd
import numpy as np
import cv2
import os
import json
import matplotlib.pyplot as plt


class Image:
    """
    Class of an Image containing the file name, category, its data,
    whether it is in the train/test set, and its bag of words
    """

    def __init__(self, filename, category, is_train=None, bag=None, data=None):
        self.filename = filename
        self.category = category
        self.is_train = is_train
        self.bag = bag
        self.data = data

    def __repr__(self):
        return '[filename={0}, category={1}, is_train={2}, bag=..., data=...]'\
            .format(self.filename, self.category, self.is_train)


class Assignment:
    save_filename = 'bags.csv'

    def __init__(self, dataset_dir):
        self.sift = cv2.SIFT_create()
        self.dataset_dir = dataset_dir
        self.loaded = os.path.exists(Assignment.save_filename)
        self.k_model = None

        if self.loaded:
            print('CSV file with BoW data detected. Using that.')
            data = pd.read_csv(Assignment.save_filename, index_col=0)

            self.train_images = []
            self.test_images = []

            for _, row in data.iterrows():
                image = Image(row.filename, row.category, row.is_train, json.loads(row.bag))

                if image.is_train:
                    self.train_images.append(image)
                else:
                    self.test_images.append(image)
        else:
            print('No CSV file with BoW data detected. Loading image data from image dataset...', end='', flush=True)

            images = self._load_images()
            self.train_images, self.test_images = train_test_split(images, test_size=0.5)

            for train_image, test_image in zip(self.train_images, self.test_images):
                train_image.is_train = True
                test_image.is_train = False

            print('DONE')

    def codebook_generation(self, k_clusters):
        print('Starting codebook generation...')
        self._compute_bags_of_words(is_train=True, k_clusters=k_clusters)
        print('Codebook generation finished!')

    def indexing(self):
        if self.k_model is None:
            raise ValueError('K means has not been run yet! Please generate the codebook first!')

        print('Starting indexing...')
        self._compute_bags_of_words(is_train=False)
        self._save_image_table()

    def retrieving(self, strat):
        # experiment 1
        print('\nRetrieval on train images (experiment #1)')
        top_3, rank_mean, accuracy = self._experiment(self.train_images, strat)
        print('Accuracy, i.e. percentage of correct predictions: {0:.{1}f}%'.format(accuracy * 100, 4))
        print('True category in top 3 predicted categories: {0:.{1}f}%'.format(top_3 * 100, 4))
        print('Mean reciprocal rank:', np.mean(rank_mean))

        # experiment 2
        print('\nRetrieval on test images (experiment #2)')
        top_3, rank_mean, accuracy = self._experiment(self.test_images, strat)
        print('Accuracy, i.e. percentage of correct predictions: {0:.{1}f}%'.format(accuracy * 100, 4))
        print('True category in top 3 predicted categories: {0:.{1}f}%'.format(top_3 * 100, 4))
        print('Mean reciprocal rank:', np.mean(rank_mean))

        plt.show()

    def _compute_bags_of_words(self, is_train, k_clusters=450):
        images = self.train_images if is_train else self.test_images

        # Extract descriptors from training images.
        # Store image bounds indexes to differentiate between images in train descriptor array, otherwise we will need a nested
        # for loop later, which will make things slower when generating the Bag Of Words for every train/test image
        image_bounds = [0]
        sift_desc = []

        print('Extracting descriptors from images with SIFT...', end='', flush=True)
        for img in images:
            keypoints, descriptors = self.sift.detectAndCompute(img.data, mask=None)
            sift_desc.extend(descriptors)
            image_bounds.append(image_bounds[len(image_bounds) - 1] + len(descriptors))
        print('DONE')

        if is_train:  # Run K-means
            print('Running MiniBatch K-Means...', end='', flush=True)
            self.k_model = MiniBatchKMeans(
                n_clusters=k_clusters,
                batch_size=100,
                max_iter=1500,
                max_no_improvement=100,
                n_init=10
            ).fit(sift_desc)
            print('DONE')

        print('Building Bag of Words for each image...', end='', flush=True)
        for i, image in enumerate(images):
            start, end = image_bounds[i], image_bounds[i+1]
            visual_words = [int(lbl) for lbl in self.k_model.predict(sift_desc[start:end])]
            image.bag = dict(Counter(visual_words))
        print('DONE')

    def _experiment(self, query_images, strat):
        if strat == 'common':
            print('Using common words retrieval method')
            return self._common_words_strat(query_images)
        elif strat == 'tfidf':
            print('Using TF-IDF with cosine similarity retrieval method')
            return self._tfidf_strat(query_images)

    def _common_words_strat(self, query_images):
        top_3_count = 0
        correct_pred_count = 0
        reciprocal_ranks = []

        # randomly choose image to demonstrate how retrieval works
        demonstration_image = np.random.choice(query_images)

        for img_query in query_images:
            common_words = []

            for img_train in self.train_images:
                # count common words with train image
                total_common_words = len(set.intersection(set(img_query.bag.keys()), set(img_train.bag.keys())))
                common_words.append((total_common_words, img_train))

            # get (common_count, image) in descending common_count order
            common_counts = sorted(common_words, key=lambda tup: tup[0], reverse=True)
            desc_categories = [tup[1].category for tup in common_counts]

            # using dict.fromkeys to remove duplicates and get the top 3 categories (order works only for python 3.5+!!!)
            # Example:
            # ['bass', 'beaver', 'bass', 'barrel', ...] ->
            # ['bass', 'beaver', 'barrel', ...][:3] ->
            # ['bass', 'beaver', 'barrel']
            top_3_count += 1 if img_query.category in list(dict.fromkeys(desc_categories))[:3] else 0

            correct_pred_count += 1 if img_query.category == desc_categories[0] else 0
            reciprocal_ranks.append(1 / (desc_categories.index(img_query.category) + 1))

            print('\tFile: {0} | True category: {1} | Predicted category: {2}'
                  .format(img_query.filename, img_query.category, desc_categories[0]))

            if img_query is demonstration_image:
                self._demonstrate(img_query, common_counts)

        accuracy = correct_pred_count / len(query_images)
        top_3_ratio = top_3_count / len(query_images)

        return top_3_ratio, np.mean(reciprocal_ranks), accuracy

    def _tfidf_strat(self, query_images):
        top_3_count = 0
        correct_pred_count = 0
        reciprocal_ranks = []

        all_words = sum([Counter(img.bag) for img in self.train_images], Counter()).keys()
        training_tfidfs = [(self._get_tfidf_vals(img_train, all_words), img_train) for img_train in self.train_images]

        # randomly choose image to demonstrate how retrieval works
        demonstration_image = np.random.choice(query_images)

        for img_query in query_images:
            tfidf_target = self._get_tfidf_vals(img_query, all_words)
            cosine_similarities = []

            for tfidf_train, img_train in training_tfidfs:
                similarity = (tfidf_target @ tfidf_train) / \
                             (np.sqrt(tfidf_target @ tfidf_target.T) * np.sqrt(tfidf_train @ tfidf_train.T))

                cosine_similarities.append((similarity, img_train))

            # get (similarity, image) in descending common_count order
            similarities = sorted(cosine_similarities, key=lambda tup: tup[0], reverse=True)
            desc_categories = [tup[1].category for tup in similarities]

            top_3_count += 1 if img_query.category in list(dict.fromkeys(desc_categories))[:3] else 0

            correct_pred_count += 1 if img_query.category == desc_categories[0] else 0
            reciprocal_ranks.append(1 / (desc_categories.index(img_query.category) + 1))

            print('\tFile: {0} | True category: {1} | Predicted category: {2}'
                  .format(img_query.filename, img_query.category, desc_categories[0]))

            if img_query is demonstration_image:
                self._demonstrate(img_query, similarities)

        accuracy = correct_pred_count / len(query_images)
        top_3_ratio = top_3_count / len(query_images)

        return top_3_ratio, np.mean(reciprocal_ranks), accuracy

    def _get_tfidf_vals(self, image, all_words):
        total_words = np.sum(list(image.bag.values()))  # total words in document/image
        tf_scores = []
        idf_scores = []

        for word in all_words:
            # Total occurrences of the word in a photo / Total words in that photo
            tf_scores.append(image.bag.get(word, 0) / total_words)
            # log_e(Total number of training photos / Number of photos with the word)
            photos_with_word = len([img for img in self.train_images if word in img.bag.keys()])
            idf_scores.append(np.log(len(self.train_images) / photos_with_word))

        return np.array(tf_scores) * np.array(idf_scores)

    def _demonstrate(self, query_image, ranking, top=7):
        # read image data from filenames
        image_data = [cv2.imread(self.dataset_dir + query_image.category + '/' + query_image.filename)]
        image_data.extend([cv2.imread(self.dataset_dir + img.category + '/' + img.filename) for _, img in ranking[:top]])
        labels = [
            'Query image',
            *['Top {0} similar'.format(i+1) for i in range(top)],
        ]

        title = \
            'Clusters: {0}, batch size: {1}, max iters: {2},\ncluster re-initializations: {3}, max steps w/o improvement: {4}'\
            .format(450, 100, 1500, 100, 10)

        plot(image_data, labels=labels, cols=4, title=title, fontsize=22)

    def _load_images(self):
        images = []
        img_categories = next(os.walk(self.dataset_dir))[1]

        for cat in img_categories:  # meow
            img_names = next(os.walk(self.dataset_dir + cat))[2]

            for filename in img_names:
                data = cv2.imread(self.dataset_dir + cat + '/' + filename)
                images.append(Image(filename, cat, data=data))

        return images

    def _save_image_table(self):
        print('Saving image/BoW data into a csv file...', end='', flush=True)
        images = self.train_images + self.test_images

        table = pd.DataFrame({
            'filename': [img.filename for img in images],
            'category': [img.category for img in images],
            'is_train': [img.is_train for img in images],
            'bag': [json.dumps(img.bag) for img in images]
        })

        table.to_csv(Assignment.save_filename)
        print('DONE')


def plot(images, labels=None, cols=1, fontsize=6, figsize=(20, 20), title=None):
    if images is None or not len(images):
        raise ValueError('No images list passed or list is empty')

    if labels is None:
        labels = []

    figure, axes = plt.subplots(int(np.ceil(len(images) / cols)), cols, squeeze=False, constrained_layout=True)
    figure.set_size_inches(*figsize)
    max_index = 0

    for i, image in enumerate(images):
        ax = axes[i // cols][i % cols]
        ax.imshow(image)
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


