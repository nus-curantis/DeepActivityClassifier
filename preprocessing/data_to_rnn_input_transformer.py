import numpy as np

from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from preprocessing.time_series_reader_and_visualizer import *


def data_to_rnn_input(data_path='../dataset/CC2650/', split_series_max_len=360, ignore_classes=[]):
    small_observations = split_segments_into_parts_with_same_len(data_path, split_series_max_len,
                                                                 ignore_classes=ignore_classes)
    return data_to_rnn_input_(small_observations, ignore_classes=ignore_classes)


def data_to_rnn_input_(split_activities, ignore_classes=[]):
    rnn_data = []
    labels = []

    series_max_len = 0

    for observation in split_activities:
        if len(observation.acc_x_series) > series_max_len:
            series_max_len = len(observation.acc_x_series)

    for observation in split_activities:
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

    return shuffle(np.array(rnn_data), get_one_hot_labels(labels, ignore_classes=ignore_classes),
                   random_state=0)  # todo: this needs to be removed at some point


def get_one_hot_labels(labels, ignore_classes=[]):
    # labels_num = max(labels) + 1
    labels_num = len(set(labels)) + len(ignore_classes)

    print(len(set(labels)))
    print(len(ignore_classes))

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


def data_to_rnn_input_train_test(data_path='../dataset/CC2650/', split_series_max_len=360,
                                 ignore_classes=[], test_size=0.2):
    rnn_data, labels = data_to_rnn_input(data_path, split_series_max_len, ignore_classes=ignore_classes)

    train_data, test_data, train_labels, test_labels = train_test_split(rnn_data, labels, test_size=test_size)
    analyze_train_test_data(train_labels, test_labels, ignore_classes=ignore_classes)

    return train_data, test_data, train_labels, test_labels


def normalized_rnn_input_train_test(data_path='../dataset/CC2650/', split_series_max_len=360,
                                    ignore_classes=[], test_size=0.2):
    rnn_data, labels = data_to_rnn_input(data_path, split_series_max_len, ignore_classes=ignore_classes)
    normalized_data = normalize_data(rnn_data, split_series_max_len=split_series_max_len)

    return train_test_split(normalized_data, labels, test_size=test_size)


def normalized_rnn_input_train_test_(split_activities, test_size=0.2, split_series_max_len=360):
    rnn_data, labels = data_to_rnn_input_(split_activities)
    normalized_data = normalize_data(rnn_data, split_series_max_len=split_series_max_len)

    return train_test_split(normalized_data, labels, test_size=test_size)


def analyze_train_test_data(train_labels, test_labels, ignore_classes=[]):
    train_labels = np.argmax(train_labels, 1)
    test_labels = np.argmax(test_labels, 1)
    ignore_classes = np.array(ignore_classes)

    all_labels = np.concatenate([train_labels, test_labels, ignore_classes])
    num_labels = len(set(all_labels))

    print('~~~~~~~~~~~~~~~~~~~ analyzing test, train data:')
    print('train samples:', len(train_labels))
    print('test samples:', len(test_labels))

    labels_num_in_train_data = np.zeros(num_labels)
    labels_num_in_test_data = np.zeros(num_labels)

    for label in train_labels:
        labels_num_in_train_data[label] += 1

    for label in test_labels:
        labels_num_in_test_data[label] += 1

    print(labels_num_in_train_data)
    print(labels_num_in_test_data)

# rnn_data, labels = data_to_rnn_input(data_path='../dataset/Chest_Accelerometer/data/')
# print('data shape: ', rnn_data.shape)
# print('labels shape: ', labels.shape)
# normalized_data = normalize_data(rnn_data)
# print(normalized_data.shape)
# print(normalized_data[0, :, 0].tolist())
# print(rnn_data[0, :, 0].tolist())

# data_to_rnn_input_train_test(data_path='../dataset/MHEALTHDATASET/', ignore_classes=[0, 12])
# data_to_rnn_input_train_test(data_path='../dataset/Chest_Accelerometer/data/', ignore_classes=[0, 2, 5, 6])
data_to_rnn_input_train_test(ignore_classes=[1, 3, 4, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 17])
data_to_rnn_input_train_test()
