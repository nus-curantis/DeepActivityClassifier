"""
Some parts of this code are copied from nus_curantis/medoid project developed by Zhang Tianyang
Codes are changeed by Pouya Kananian
"""

import random
import os

import numpy as np
from scipy.spatial.distance import euclidean
from tqdm import tqdm
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import pdist
from matplotlib import pyplot as plt
from sklearn.cluster import AgglomerativeClustering

from preprocessing.pamap2_reader import get_map_class, ACTIVITIES_MAP, normalized_pamap2_rnn_input_train_test
from clustering.dtw_lib import _dtw_lib


class ClusteringExecutor:
    def __init__(self):
        self.all_train_data = None
        self.all_test_data = None
        self.all_train_labels = None
        self.all_test_labels = None

        self.selected_train_segments = []  # These segments are going to be clustered
        self.selected_test_segments = []  # These segments are going to be clustered
        self.selected_train_data = []  # This array contains all 3 dimensions of the selected data
        self.selected_test_data = []  # This array contains all 3 dimensions of the selected test data
        self.selected_train_labels = []  # label specifying type of activity not num of cluster data belongs to
        self.selected_test_labels = []
        self.selected_train_data_indices = []
        self.selected_test_data_indices = []
        self.train_cluster_nums = []
        self.test_cluster_nums = []
        self.class_name = None

        self.plots_address = None

        # self.load_all_data()

    def set_all_data(self, all_train_data, all_test_data, all_train_labels, all_test_labels):
        self.all_train_data = all_train_data
        self.all_train_labels = all_train_labels
        self.all_test_data = all_test_data
        self.all_test_labels = all_test_labels

    def load_all_data(self, series_max_len=360):
        self.all_train_data, self.all_test_data, self.all_train_labels, self.all_test_labels = \
            normalized_pamap2_rnn_input_train_test(split_series_max_len=series_max_len)  # pamap2 dataset

    def is_data_loaded(self):
        for data in [self.all_train_data, self.all_test_data,
                     self.all_train_labels, self.all_test_labels]:
            if data is None:
                return False

        return True

    def load_data_of_one_class(self, class_name='ironing', axis='x', series_max_len=360, num_segments=200):
        map_class = get_map_class()
        invert_activities_map = {v: k for k, v in ACTIVITIES_MAP.items()}
        class_label = map_class[invert_activities_map[class_name]]

        cartesian_axises = ['x', 'y', 'z']
        axis_num = cartesian_axises.index(axis)

        self.class_name = class_name
        self.plots_address = 'plots/' + self.class_name

        for data in [self.all_train_labels, self.all_test_labels, self.all_train_data, self.all_test_data]:
            if data is None:
                self.load_all_data(series_max_len)
                break

        class_train_segments = []
        class_test_segments = []
        class_train_data = []
        class_test_data = []
        class_train_labels = []
        class_test_labels = []
        class_train_data_indices = []
        class_test_data_indices = []

        counter = 0
        for segment in self.all_train_data:
            if np.argmax(self.all_train_labels[counter]) == class_label:
                class_train_segments.append(segment[:, axis_num])
                class_train_data.append(segment[:, :])
                class_train_labels.append(self.all_train_labels[counter])
                class_test_data_indices.append(counter)

            if len(class_train_segments) > num_segments:
                break

            counter += 1

        counter = 0
        for segment in self.all_test_data:
            if np.argmax(self.all_test_labels[counter]) == class_label:
                class_test_segments.append(segment[:, axis_num])
                class_test_data.append(segment[:, :])
                class_test_labels.append(self.all_test_labels[counter])
                class_test_data_indices.append(counter)

            if len(class_test_segments) > num_segments:
                break

            counter += 1

        # print(np.array(class_train_segments).shape)
        # print(np.array(class_test_segments).shape)
        # print(class_train_segments[0])

        self.selected_train_segments = np.array(class_train_segments)
        self.selected_test_segments = np.array(class_test_segments)
        self.selected_train_data = np.array(class_train_data)
        self.selected_test_data = np.array(class_test_data)
        self.selected_train_labels = np.array(class_train_labels)
        self.selected_test_labels = np.array(class_test_labels)
        self.selected_train_data_indices = np.array(class_train_data_indices)
        self.selected_test_data_indices = np.array(class_test_data_indices)

    def calculate_medoids_and_clusters(self, num_clusters=2):
        def distance(seg1, seg2, relax):
            distance, path, D = _dtw_lib.fastdtw(seg1, seg2, relax=relax, dist=euclidean)
            return distance

        def find_medoid_seg(segs):
            print('len(segs): ', len(segs))
            print(segs.shape)

            length = len(segs)
            result = [0 for _ in range(length)]
            table = [[-1 for _ in range(length)] for _ in range(length)]  # initialize the table to all -1

            for i in tqdm(range(length)):
                for j in range(length):
                    if i == j:
                        table[i][j] = 0
                        continue
                    elif table[j][i] != -1:  # using memoization
                        table[i][j] = table[j][i]
                        result[i] += table[i][j]
                    else:
                        table[i][j] = distance(segs[i], segs[j], 1)
                        result[i] += table[i][j]

            # store([table], ['user' + str(user) + '_' + activity + '_matrix_' + str(duration)])

            min_medoid = min(result)
            for i in range(len(result)):
                if min_medoid == result[i]:
                    return segs[i], min_medoid / len(segs), table

        segs = self.selected_train_segments
        representations, min_medoid, table = find_medoid_seg(segs)

        # print(min_medoid)
        # print(representations)
        # print(table)
        # print('-----------------------------------------')

        self.plot_matrix(table, title=self.class_name, save_folder=self.plots_address + '/train/',
                         plot_name='dist_mat.png')
        self.hierarchical_plot(matrix=table, segments=self.selected_train_segments, method='single',
                               save_folder=self.plots_address + '/train/', plot_name='dendrogram.png')

        self.train_cluster_nums = self.get_hierarchical_cluster(num_cluster=num_clusters, matrix=table)

        segs = self.selected_test_segments
        representations, min_medoid, table = find_medoid_seg(segs)
        self.test_cluster_nums = self.get_hierarchical_cluster(num_cluster=num_clusters, matrix=table)

    def get_clustered_data(self, class_name='lying', num_segments=200, series_max_len=360, num_clusters=2):
        if not self.is_data_loaded():
            self.load_all_data(series_max_len=series_max_len)

        self.load_data_of_one_class(class_name=class_name, num_segments=num_segments, series_max_len=series_max_len)
        self.calculate_medoids_and_clusters(num_clusters=num_clusters)

        return self.selected_train_data, self.selected_train_labels, self.train_cluster_nums, \
            self.selected_test_data, self.selected_test_labels, self.test_cluster_nums

    @staticmethod
    def plot_matrix(matrix, title, save_folder, plot_name):
        fig, ax = plt.subplots()
        im = ax.imshow(matrix)

        title = "distance distribution for %s " % title
        ax.set_title(title)
        fig.tight_layout()
        fig.colorbar(im, ax=ax)

        # plt.show()
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)
        plt.savefig(save_folder + plot_name)

    def hierarchical_plot(self, matrix, segments, method, save_folder, plot_name):
        f = lambda x, y: matrix[x[1]][y[1]]
        X = list(map(lambda x: [0, x], range(len(segments))))
        Y = pdist(X, f)
        linked = linkage(Y, method, metric='')

        labels_list = range(0, len(segments))

        plt.figure(figsize=(10, 7))
        dendrogram(linked,
                   orientation='top',
                   labels=labels_list,
                   distance_sort='descending',
                   show_leaf_counts=True)
        plt.title(self.class_name + "======" + method)

        # plt.show()
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)
        plt.savefig(save_folder + plot_name)

    @staticmethod
    def get_hierarchical_cluster(num_cluster, matrix):
        cluster = AgglomerativeClustering(n_clusters=num_cluster, affinity='precomputed', linkage='complete')
        return cluster.fit_predict(matrix)


# c = ClusteringExecutor()
# for class_name in ['lying',
#                    'sitting',
#                    'standing',
#                    'walking',
#                    'running',
#                    'cycling',
#                    'nordic_walking',
#                    'no_activity']:
#     c.load_data_of_one_class(class_name=class_name)
#     c.calculate_medoids_and_clusters()
