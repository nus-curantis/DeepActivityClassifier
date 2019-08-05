# This file reads pamap file from the matfile created by LSTMEnsemble4HAR, the data in this file is normalized

import scipy.io
import pandas as pd
import numpy as np

from preprocessing.time_series_reader_and_visualizer import Activity, split_segments_of_activity
from preprocessing.data_to_rnn_input_transformer import data_to_rnn_input_train_test_


def read_data(data_path='../dataset/PAMAP2.mat', split_series_max_len=360):
    data = scipy.io.loadmat(data_path)

    X_train = data['X_train']
    X_valid = data['X_valid']
    X_test = data['X_test']
    y_train = data['y_train'].reshape(-1)
    y_valid = data['y_valid'].reshape(-1)
    y_test = data['y_test'].reshape(-1)

    # y_train = pd.get_dummies(y_train, prefix='labels')
    # y_valid = pd.get_dummies(y_valid, prefix='labels')
    # y_test = pd.get_dummies(y_test, prefix='labels')

    data = np.concatenate([X_train, X_valid, X_test], axis=0)
    labels = np.concatenate([y_train, y_valid, y_test], axis=0)

    activities = extract_activities(data, labels)

    split_activities = []
    for activity in activities:
        split_activities += split_segments_of_activity(activity, split_series_max_len=split_series_max_len)

    return activities, split_activities


def extract_activities(data, labels):
    activities = []

    counter = 0
    previous_activity_num = -10
    for row in data:
        activity_num = int(labels[counter])
        if activity_num != previous_activity_num:
            previous_activity_num = activity_num

            activities.append(Activity(activity_num))

        activities[-1].append_acc_data(float(row[4]), float(row[5]), float(row[6]))
        print(float(row[4]), float(row[5]), float(row[6]))

    return activities


def pamap_rnn_input_train_test(data_path='../dataset/PAMAP2.mat', split_series_max_len=360):
    _, split_activities = read_data(data_path, split_series_max_len=split_series_max_len)
    return data_to_rnn_input_train_test_(split_activities=split_activities,
                                         split_series_max_len=split_series_max_len)


d, l = read_data()
# print(l[10])
extract_activities(d, l)
