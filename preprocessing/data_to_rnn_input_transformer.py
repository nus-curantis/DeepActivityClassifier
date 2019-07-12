import numpy as np

from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler

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


def normalize_data(rnn_data, split_series_max_len=360):  # todo: test different methods of normalizing
    x_data = np.transpose(np.reshape(rnn_data[:, :, 0], newshape=[-1, split_series_max_len]))
    y_data = np.transpose(np.reshape(rnn_data[:, :, 1], newshape=[-1, split_series_max_len]))
    z_data = np.transpose(np.reshape(rnn_data[:, :, 2], newshape=[-1, split_series_max_len]))

    # scaler = MinMaxScaler()
    scaler = StandardScaler()

    scaler.fit(x_data)
    x_data = scaler.transform(x_data)

    scaler.fit(y_data)
    y_data = scaler.transform(y_data)

    scaler.fit(z_data)
    z_data = scaler.transform(z_data)

    scaled_data = np.array([
        x_data, y_data, z_data
    ])

    return np.transpose(scaled_data)


def data_to_rnn_input_train_test(data_path='../dataset/CC2650/', split_series_max_len=360, test_size=0.2):
    rnn_data, labels = data_to_rnn_input(data_path, split_series_max_len)

    return train_test_split(rnn_data, labels, test_size=test_size)


def normalized_rnn_input_train_test(data_path='../dataset/CC2650/', split_series_max_len=360, test_size=0.2):
    rnn_data, labels = data_to_rnn_input(data_path, split_series_max_len)
    normalized_data = normalize_data(rnn_data)

    return train_test_split(normalized_data, labels, test_size=test_size)


# rnn_data, labels = data_to_rnn_input(data_path='../dataset/Chest_Accelerometer/data/')
# print('data shape: ', rnn_data.shape)
# print('labels shape: ', labels.shape)
# normalized_data = normalize_data(rnn_data)
# print(normalized_data.shape)
# print(normalized_data[0, :, 0].tolist())
# print(rnn_data[0, :, 0].tolist())
