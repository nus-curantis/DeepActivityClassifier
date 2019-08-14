"""
 !! Some parts of this code is copied from
 https://github.com/NLeSC/mcfly-tutorial/blob/master/utils/tutorial_pamap2.py
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from os import listdir
import os.path

from preprocessing.data_to_rnn_input_transformer import normalized_rnn_input_train_test_, data_to_rnn_input_train_test_


class Activity:
    def __init__(self, num):
        self.num = num
        self.acc_x_series = []
        self.acc_y_series = []
        self.acc_z_series = []
        self.gyr_x_series = []
        self.gyr_y_series = []
        self.gyr_z_series = []

    def append_acc_data(self, x, y, z):
        self.acc_x_series.append(x)
        self.acc_y_series.append(y)
        self.acc_z_series.append(z)

    def append_gyr_data(self, x, y, z):
        self.gyr_x_series.append(x)
        self.gyr_y_series.append(y)
        self.gyr_z_series.append(z)

    def append_acc_data_series(self, x_series, y_series, z_series):
        self.acc_x_series += x_series
        self.acc_y_series += y_series
        self.acc_z_series += z_series

    def append_gyr_data_series(self, x_series, y_series, z_series):
        self.gyr_x_series += x_series
        self.gyr_y_series += y_series
        self.gyr_z_series += z_series


def read_all_files(target_dir='../dataset/', include_gyr_data=False,
                   exclude_activities=[], split_series_max_len=360):
    columns_to_use = ['activityID', 'hand_acc_16g_x', 'hand_acc_16g_y', 'hand_acc_16g_z']
    if include_gyr_data:
        columns_to_use += ['hand_gyroscope_x', 'hand_gyroscope_y', 'hand_gyroscope_z']

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

    print(datasets_filled[0].columns)

    # Create mapping for class labels
    class_labels, nr_classes, map_classes = map_class(datasets_filled, exclude_activities)

    selected_datas = [np.array(data[columns_to_use]) for data in datasets_filled]

    print('kherrrrrrrrrrrrrrrrrrrrrrrrrrrrrssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssss')
    print(class_labels)
    print(nr_classes)
    print(map_classes)

    activities = []
    for data in selected_datas:
        print(data)
        activities += extract_activities(data, map_classes, include_gyr_data=include_gyr_data)

    print('total recorded activities: ', len(activities))

    # for activity in activities:
    #     print(activity.num, len(activity.acc_x_series))
    #     print(activity.acc_x_series[0:10])

    split_activities = []
    for activity in activities:
        split_activities += split_segments_of_activity(activity,
                                                       split_series_max_len=split_series_max_len,
                                                       include_gyr_data=include_gyr_data)

    return activities, split_activities


def extract_activities(selected_data, map_classes, include_gyr_data=False):
    activities = []

    previous_activity_num = -10
    for row in selected_data:
        activity_num = int(float(map_classes[row[0]]))
        if activity_num != previous_activity_num:
            previous_activity_num = activity_num

            activities.append(Activity(activity_num))

        activities[-1].append_acc_data(float(row[1]), float(row[2]), float(row[3]))

        if include_gyr_data:
            activities[-1].append_gyr_data(float(row[4]), float(row[5]), float(row[6]))

    return activities


def split_segments_of_activity(activity, split_series_max_len=360, overlap=0, include_gyr_data=False):
    split_activities = []

    overlap_len = int(split_series_max_len * overlap)

    for i in range(0, len(activity.acc_x_series) + overlap_len, split_series_max_len):
        split_activities.append(Activity(activity.num))

        if i != 0:
            split_activities[-1].append_acc_data_series(
                activity.acc_x_series[i - overlap_len:
                                      i + split_series_max_len - overlap_len],
                activity.acc_y_series[i - overlap_len:
                                      i + split_series_max_len - overlap_len],
                activity.acc_z_series[i - overlap_len:
                                      i + split_series_max_len - overlap_len]
            )

            if include_gyr_data:
                split_activities[-1].append_gyr_data_series(
                    activity.gyr_x_series[i - overlap_len:
                                          i + split_series_max_len - overlap_len],
                    activity.gyr_y_series[i - overlap_len:
                                          i + split_series_max_len - overlap_len],
                    activity.gyr_z_series[i - overlap_len:
                                          i + split_series_max_len - overlap_len]
                )

        else:
            split_activities[-1].append_acc_data_series(
                activity.acc_x_series[i: i + split_series_max_len],
                activity.acc_y_series[i: i + split_series_max_len],
                activity.acc_z_series[i: i + split_series_max_len]
            )

            if include_gyr_data:
                split_activities[-1].append_gyr_data_series(
                    activity.gyr_x_series[i: i + split_series_max_len],
                    activity.gyr_y_series[i: i + split_series_max_len],
                    activity.gyr_z_series[i: i + split_series_max_len]
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
    y_set_all = [set(np.array(data.activityID)) - set(exclude_activities)
                 for data in datasets_filled]
    class_ids = list(set.union(*[set(y) for y in y_set_all]))
    class_labels = [ACTIVITIES_MAP[i] for i in class_ids]
    nr_classes = len(class_ids)
    map_classes = {class_ids[i]: i for i in range(len(class_ids))}
    return class_labels, nr_classes, map_classes


def get_inverted_map_class(target_dir='../dataset/', include_gyr_data=False,
                           exclude_activities=[], split_series_max_len=360):
    columns_to_use = ['activityID', 'hand_acc_16g_x', 'hand_acc_16g_y', 'hand_acc_16g_z']
    if include_gyr_data:
        columns_to_use += ['hand_gyroscope_x', 'hand_gyroscope_y', 'hand_gyroscope_z']

    data_dir = os.path.join(target_dir, 'PAMAP2_Dataset', 'Protocol')
    file_names = listdir(data_dir)
    file_names.sort()

    # load the files and put them in a list of pandas dataframes:
    datasets = [pd.read_csv(os.path.join(data_dir, fn), header=None, sep=' ')
                for fn in file_names]
    datasets = add_header(datasets)  # add headers to the datasets

    # for dataset in datasets:
    #     print(dataset)

    # interpolate dataset to get same sample rate between channels
    datasets_filled = [d.interpolate() for d in datasets]

    print(datasets_filled[0].columns)

    # Create mapping for class labels
    class_labels, nr_classes, map_classes = map_class(datasets_filled, exclude_activities)

    return dict(map(reversed, map_classes.items()))


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


def normalized_pamap2_rnn_input_train_test(target_dir='../dataset/', split_series_max_len=360, include_gyr_data=False):
    _, split_activities = read_all_files(target_dir, split_series_max_len=split_series_max_len,
                                         include_gyr_data=include_gyr_data)
    return normalized_rnn_input_train_test_(split_activities=split_activities,
                                            split_series_max_len=split_series_max_len,
                                            include_gyr_data=include_gyr_data)


def pamap2_rnn_input_train_test(target_dir='../dataset/', split_series_max_len=360, include_gyr_data=False):
    # todo: add 'ignore classes' and etc
    _, split_activities = read_all_files(target_dir, split_series_max_len=split_series_max_len,
                                         include_gyr_data=include_gyr_data)
    return data_to_rnn_input_train_test_(split_activities=split_activities,
                                         split_series_max_len=split_series_max_len,
                                         include_gyr_data=include_gyr_data)


def get_pamap_dataset_labels_names(ignore_classes=[]):  # todo: clean this code
    return ['no_activity', 'lying', 'sitting', 'standing', 'walking', 'running', 'cycling', 'nordic_walking',
            'ascending_stairs', 'descending_stairs', 'vaccuum_cleaning', 'ironing', 'rope_jumping']

    # {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7, 12: 8, 13: 9, 16: 10, 17: 11, 24: 12}

    # labels_names = []
    # for i in range(0, len(ACTIVITIES_MAP)):
    #     if i in ACTIVITIES_MAP.keys() and i not in ignore_classes:
    #         labels_names.append(ACTIVITIES_MAP[i])
    #
    # return labels_names


inverted_class_map = get_inverted_map_class()


def plot_series(save_folder, record_num, time_series, axis_name, label, pred_label=None):
    corrected_label = inverted_class_map[label]
    corrected_pred_label = inverted_class_map[pred_label]

    save_folder += ACTIVITIES_MAP[corrected_label] + '/'
    if pred_label is None:
        save_folder += 'no_pred_done/'
    elif label == pred_label:
        save_folder += 'correct_pred/' + ACTIVITIES_MAP[corrected_pred_label] + '/'
    else:
        save_folder += 'wrong_pred/'

    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    plt.clf()
    plt.plot(time_series)

    if pred_label is not None:
        plt.xlabel('activity: ' + str(ACTIVITIES_MAP[corrected_label]) + ' - pred as: ' +
                   str(ACTIVITIES_MAP[corrected_pred_label]))
    else:
        plt.xlabel('activity: ' + str(ACTIVITIES_MAP[corrected_label]))

    plt.ylabel('accelerometer ' + axis_name + ' series')
    plt.savefig(save_folder + 'series_' +
                str(record_num) + '_' + str(ACTIVITIES_MAP[corrected_label]) + '_axis_' + axis_name + '.png')


# read_all_files()
