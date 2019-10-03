"""
 PAMAP2 dataset reader and preprocessor

 !! Some parts of this code is copied from
 https://github.com/NLeSC/mcfly-tutorial/blob/master/utils/tutorial_pamap2.py

 This code is more flexible than the other pamap2_reader and can use all the dimensions of data, the other code
 is only able to use accelerometer data and wrist gyroscope data
"""
import numpy as np
import pandas as pd
from os import listdir
import os.path

from preprocessing.data_to_rnn_input_transformer import get_one_hot_labels, shuffle, train_test_split, \
    analyze_train_test_data

from sklearn import preprocessing


class Activity:
    def __init__(self, num):
        self.num = num
        self.data_series = []

    def append_data(self, row):
        self.data_series.append(row)

    def append_data_series(self, row_series):
        self.data_series += row_series


def read_all_files(target_dir='../dataset/',
                   columns_to_use=['activityID', 'heartrate', 'hand_temperature',
                                   'hand_acc_16g_x', 'hand_acc_16g_y', 'hand_acc_16g_z', 'hand_acc_6g_x',
                                   'hand_acc_6g_y', 'hand_acc_6g_z', 'hand_gyroscope_x',
                                   'hand_gyroscope_y', 'hand_gyroscope_z', 'hand_magnometer_x',
                                   'hand_magnometer_y', 'hand_magnometer_z', 'hand_orientation_0',
                                   'hand_orientation_1', 'hand_orientation_2', 'hand_orientation_3',
                                   'chest_temperature', 'chest_acc_16g_x', 'chest_acc_16g_y',
                                   'chest_acc_16g_z', 'chest_acc_6g_x', 'chest_acc_6g_y', 'chest_acc_6g_z',
                                   'chest_gyroscope_x', 'chest_gyroscope_y', 'chest_gyroscope_z',
                                   'chest_magnometer_x', 'chest_magnometer_y', 'chest_magnometer_z',
                                   'chest_orientation_0', 'chest_orientation_1', 'chest_orientation_2',
                                   'chest_orientation_3', 'ankle_temperature', 'ankle_acc_16g_x',
                                   'ankle_acc_16g_y', 'ankle_acc_16g_z', 'ankle_acc_6g_x',
                                   'ankle_acc_6g_y', 'ankle_acc_6g_z', 'ankle_gyroscope_x',
                                   'ankle_gyroscope_y', 'ankle_gyroscope_z', 'ankle_magnometer_x',
                                   'ankle_magnometer_y', 'ankle_magnometer_z', 'ankle_orientation_0',
                                   'ankle_orientation_1', 'ankle_orientation_2', 'ankle_orientation_3'],
                   exclude_activities=[], split_series_max_len=360):
    """

    :param target_dir: The path where PAMAP2_Dataset is in
    :param columns_to_use: each column of data belongs to one sensor and it's being declared here which sensors to use
    :param exclude_activities: activity types which are going to be ignored in this process.
    :param split_series_max_len: shows the maximum length of the output segments. (Original activity time series are
    divided into segments and length of these segments doesn't exceed this param)
    :return:
        activities: extracted activities from dataset with complete recorded time series
        split_activities: activities with time series divided to smaller segments
    """

    data_dir = os.path.join(target_dir, 'PAMAP2_Dataset', 'Protocol')
    file_names = listdir(data_dir)
    file_names.sort()

    print(data_dir)
    print(file_names)

    print('Start pre-processing all ' + str(len(file_names)) + ' files...')

    # load the files and put them in a list of pandas dataframes:
    datasets = [pd.read_csv(os.path.join(data_dir, fn), header=None, sep=' ')
                for fn in file_names]
    datasets = add_header(datasets)  # add headers to the datasets

    # interpolate dataset to get same sample rate between channels
    datasets_filled = [d.interpolate() for d in datasets]
    datasets_filled = [d.fillna(value=0) for d in datasets]

    for d in datasets_filled:
        print('nan_check: ', d.isnull().values.sum())
        print(d.columns[d.isna().any()].tolist())

    # print(datasets_filled[0].columns)

    # Create mapping for class labels
    class_labels, nr_classes, map_classes = map_class(datasets_filled, exclude_activities)

    selected_datas = [np.array(data[columns_to_use]) for data in datasets_filled]

    # print(class_labels)
    # print(nr_classes)
    # print(map_classes)

    activities = []
    for data in selected_datas:
        print(data)
        activities += extract_activities(data, map_classes)

    print('total recorded activities: ', len(activities))

    split_activities = []
    for activity in activities:
        split_activities += split_segments_of_activity(activity,
                                                       split_series_max_len=split_series_max_len)

    return activities, split_activities


def extract_activities(selected_data, map_classes):
    """

    :param selected_data: a selected data file in dataset
    :param map_classes: in PAMAP2 dataset all of the activity types listed in ACTIVITY_MAP are not present. 0, 1, 2, ...
     is used for labelling the present data.This map is used for projecting 0, 1, 2, 3 ... to the index used in
     ACTIVITY_MAP
    :return: extracted activities from dataset with complete recorded time series
    """
    activities = []

    previous_activity_num = -10
    for row in selected_data:
        activity_num = int(float(map_classes[row[0]]))
        if activity_num != previous_activity_num:
            previous_activity_num = activity_num

            activities.append(Activity(activity_num))

        activities[-1].append_data(row[1:])

    return activities


def split_segments_of_activity(activity, split_series_max_len=360, overlap=0):
    """

    :param activity: input activity
    :param split_series_max_len: shows the maximum length of the output segments
    :param overlap: declares overlap percentage of output segments
    :return: multiple activities extracted from the input activity with smaller segments of the original activity
    """
    split_activities = []

    overlap_len = int(split_series_max_len * overlap)

    for i in range(0, len(activity.data_series) + overlap_len, split_series_max_len):
        split_activities.append(Activity(activity.num))

        if i != 0:
            split_activities[-1].append_data_series(
                activity.data_series[i - overlap_len:
                                     i + split_series_max_len - overlap_len]
            )
        else:
            split_activities[-1].append_data_series(
                activity.data_series[i: i + split_series_max_len],
            )

    return split_activities


def add_header(datasets):
    """
    The columns of the pandas data frame are numbers
    this function adds the column labels
    Parameters
    ----------
    datasets : list
        List of pandas dataframes
    """
    header = get_header()
    for i in range(0, len(datasets)):
        datasets[i].columns = header
    return datasets


def get_header():
    axes = ['x', 'y', 'z']
    IMUsensor_columns = ['temperature'] + \
        ['acc_16g_' + i for i in axes] + \
        ['acc_6g_' + i for i in axes] + \
        ['gyroscope_' + i for i in axes] + \
        ['magnometer_' + i for i in axes] + \
        ['orientation_' + str(i) for i in range(4)]
    header = ["timestamp", "activityID", "heartrate"] + ["hand_" + s
                                                         for s in IMUsensor_columns] \
        + ["chest_" + s for s in IMUsensor_columns] + ["ankle_" + s
                                                       for s in IMUsensor_columns]
    return header


def map_class(datasets_filled, exclude_activities):
    """

    :param datasets_filled: datasets with no null data in
    :param exclude_activities: activity types which are going to be ignored in this process.
    :return:
        class_labels: list of labels
        nr_classes: number of different classes
        map_class: in PAMAP2 dataset all of the activity types listed in ACTIVITY_MAP are not present. 0, 1, 2, ...
     is used for labelling the present data.This map is generated for projecting 0, 1, 2, 3 ... to the index used in
     ACTIVITY_MAP
    """
    y_set_all = [set(np.array(data.activityID)) - set(exclude_activities)
                 for data in datasets_filled]
    class_ids = list(set.union(*[set(y) for y in y_set_all]))
    class_labels = [ACTIVITIES_MAP[i] for i in class_ids]
    nr_classes = len(class_ids)
    map_classes = {class_ids[i]: i for i in range(len(class_ids))}
    return class_labels, nr_classes, map_classes


ACTIVITIES_MAP = {
    0: 'no_activity',
    1: 'lying',
    2: 'sitting',
    3: 'standing',
    4: 'walking',
    5: 'running',
    6: 'cycling',
    7: 'nordic_walking',
    9: 'watching_tv',
    10: 'computer_work',
    11: 'car_driving',
    12: 'ascending_stairs',
    13: 'descending_stairs',
    16: 'vaccuum_cleaning',
    17: 'ironing',
    18: 'folding_laundry',
    19: 'house_cleaning',
    20: 'playing_soccer',
    24: 'rope_jumping'
}


def pamap2_rnn_input_train_test(target_dir='../dataset/', split_series_max_len=360):
    """

    :param target_dir: The path where PAMAP2_Dataset is in
    :param split_series_max_len: shows the maximum length of the output segments. (Original activity time series are
    divided into segments and length of these segments doesn't exceed this param)
    :return: Time series formatted properly for RNN input without normalization. RNN input is divided into train and
             test data
    """

    # todo: add 'ignore classes' and etc
    _, split_activities = read_all_files(target_dir, split_series_max_len=split_series_max_len)
    return data_to_rnn_input_train_test_flexible(split_activities=split_activities,
                                                 split_series_max_len=split_series_max_len)


def normalized_pamap2_rnn_input_train_test(target_dir='../dataset/', split_series_max_len=360):
    """

    :param target_dir: The path where PAMAP2_Dataset is in
    :param split_series_max_len: shows the maximum length of the output segments. (Original activity time series are
    divided into segments and length of these segments doesn't exceed this param)
    :return: Normalized time series formatted properly for RNN input. RNN input is divided into train and test data
    """

    # todo: add 'ignore classes' and etc
    _, split_activities = read_all_files(target_dir, split_series_max_len=split_series_max_len)
    return data_to_normalized_rnn_input_train_test_flexible(split_activities=split_activities,
                                                            split_series_max_len=split_series_max_len)


def data_to_rnn_input_flexible(split_activities, ignore_classes=[]):
    """

    :param split_activities: Activities with short segments produced by splitting original segments into parts with same
                             len
    :param ignore_classes: classes that are going to be ignored in this function
    :return: shuffled data with suitable format for RNN alongside one hot labels
    """
    rnn_data = []
    labels = []

    series_max_len = 0

    for observation in split_activities:
        if len(observation.data_series) > series_max_len:
            series_max_len = len(observation.data_series)

    for observation in split_activities:
        data = np.transpose(np.array(observation.data_series))
        print(np.shape(data))
        data = np.concatenate((data,
                               np.zeros((np.shape(data)[0], series_max_len - len(observation.data_series)))), axis=1
                              )

        data = np.transpose(data)

        rnn_data.append(data)

        labels.append(observation.num)

    return shuffle(np.array(rnn_data), get_one_hot_labels(labels, ignore_classes=ignore_classes),
                   random_state=0)  # todo: this needs to be removed at some point


def data_to_rnn_input_train_test_flexible(split_activities, split_series_max_len=360,
                                          ignore_classes=[], test_size=0.2):
    """

    :param split_activities: Activities with short segments produced by splitting original segments into parts with same
                             len
    :param split_series_max_len: not used, should be deleted from code (TODO)
    :param ignore_classes: classes that are going to be ignored in this function
    :param test_size: proportion of test samples
    :return: shuffled train and test data with suitable format for RNN alongside one hot labels
    """
    rnn_data, labels = data_to_rnn_input_flexible(split_activities)

    train_data, test_data, train_labels, test_labels = train_test_split(rnn_data, labels, test_size=test_size,
                                                                        stratify=labels)
    analyze_train_test_data(train_labels, test_labels, ignore_classes=ignore_classes)

    return train_data, test_data, train_labels, test_labels


def data_to_normalized_rnn_input_train_test_flexible(split_activities, split_series_max_len=360,
                                                     ignore_classes=[], test_size=0.2):
    """

    :param split_activities: Activities with short segments produced by splitting original segments into parts with same
                             len
    :param split_series_max_len: not used, should be deleted from code (TODO)
    :param ignore_classes: classes that are going to be ignored in this function
    :param test_size: proportion of test samples
    :return: normalized shuffled train and test data with suitable format for RNN alongside one hot labels
    """
    rnn_data, labels = data_to_rnn_input_flexible(split_activities)

    rnn_data_shape = np.shape(rnn_data)
    normalized_data = preprocessing.scale(np.reshape(rnn_data, newshape=[-1, rnn_data_shape[-1]]))
    rnn_data = np.reshape(normalized_data, newshape=rnn_data_shape)

    train_data, test_data, train_labels, test_labels = train_test_split(rnn_data, labels, test_size=test_size,
                                                                        stratify=labels)
    analyze_train_test_data(train_labels, test_labels, ignore_classes=ignore_classes)

    return train_data, test_data, train_labels, test_labels

