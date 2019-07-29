"""
 !! Some parts of this code is copied from
 https://github.com/NLeSC/mcfly-tutorial/blob/master/utils/tutorial_pamap2.py
"""
import numpy as np
import pandas as pd
from os import listdir
import os.path

from preprocessing.time_series_reader_and_visualizer import Activity, split_segments_of_activity
from preprocessing.data_to_rnn_input_transformer import normalized_rnn_input_train_test_, data_to_rnn_input_train_test_


def read_all_files(target_dir='../dataset/', columns_to_use=
                   ['activityID', 'hand_acc_16g_x', 'hand_acc_16g_y', 'hand_acc_16g_z'],
                   exclude_activities=[], split_series_max_len=360):
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

    # for dataset in datasets:
    #     print(dataset)

    # interpolate dataset to get same sample rate between channels
    datasets_filled = [d.interpolate() for d in datasets]

    # Create mapping for class labels
    class_labels, nr_classes, map_classes = map_class(datasets_filled, exclude_activities)

    selected_datas = [np.array(data[columns_to_use]) for data in datasets_filled]

    # print(class_labels)
    # print(nr_classes)
    # print(map_classes)

    activities = []
    for data in selected_datas:
        print(data)
        activities += extract_activities(data)

    print('total recorded activities: ', len(activities))

    for activity in activities:
        print(activity.num, len(activity.acc_x_series))
        print(activity.acc_x_series[0:10])

    split_activities = []
    for activity in activities:
        split_activities += split_segments_of_activity(activity, split_series_max_len=split_series_max_len)

    return activities, split_activities


def extract_activities(selected_data):
    activities = []

    previous_activity_num = -10
    for row in selected_data:
        activity_num = int(float(row[0]))
        if activity_num != previous_activity_num:
            previous_activity_num = activity_num

            activities.append(Activity(activity_num))

        activities[-1].append_acc_data(float(row[1]), float(row[2]), float(row[3]))

    return activities


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


def normalized_pamap2_rnn_input_train_test(target_dir='../dataset/', split_series_max_len=360):
    _, split_activities = read_all_files(target_dir, split_series_max_len=split_series_max_len)
    return normalized_rnn_input_train_test_(split_activities=split_activities,
                                            split_series_max_len=split_series_max_len)


def pamap2_rnn_input_train_test(target_dir='../dataset/', split_series_max_len=360):
    # todo: add 'ignore classes' and etc
    _, split_activities = read_all_files(target_dir, split_series_max_len=split_series_max_len)
    return data_to_rnn_input_train_test_(split_activities=split_activities,
                                         split_series_max_len=split_series_max_len)


read_all_files()
