import numpy as np

from sklearn.utils import shuffle

from preprocessing.time_series_reader_and_visualizer import *


def data_to_rnn_input(data_path='../dataset/CC2650/', split_series_max_len=360):
    small_observations = split_segments_into_parts_with_same_len(data_path, split_series_max_len)
    rnn_data = []
    labels = []

    series_max_len = 0

    for observation in small_observations:
        if len(observation.acc_x_series) > series_max_len:
            series_max_len = len(observation.acc_x_series)

    for observation in small_observations:
        acc_data = np.array([
            np.append(
                np.array(observation.acc_x_series), np.zeros(series_max_len - len(observation.acc_x_series))
            )
            ,
            np.append(
                np.array(observation.acc_y_series), np.zeros(series_max_len - len(observation.acc_y_series))
            )
            ,
            np.append(
                np.array(observation.acc_z_series), np.zeros(series_max_len - len(observation.acc_z_series))
            )
        ])

        acc_data = np.reshape(acc_data, newshape=[acc_data.shape[1], -1])

        rnn_data.append(np.array(acc_data))

        labels.append(observation.num)

    return shuffle(np.array(rnn_data), get_one_hot_labels(labels),
                   random_state=0)  # todo: this needs to be removed at some point


def get_one_hot_labels(labels):
    labels_num = max(labels) + 1

    one_hots = []
    for label in labels:
        one_hot = np.zeros(labels_num)
        one_hot[label] = 1
        one_hots.append(one_hot)

    return np.array(one_hots)


rnn_data, labels = data_to_rnn_input()
print('data shape: ', rnn_data.shape)
print('labels shape: ', labels.shape)
