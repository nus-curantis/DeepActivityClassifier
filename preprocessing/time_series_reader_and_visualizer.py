"""
Functions used for preprocessing different datasets. Functions and Activity class of this file are used in other
preprocessing scripts too.
"""

import os
import csv
import matplotlib.pyplot as plt

num_to_activity = {
    -1: 'Not tagged',
    0: 'Walking',
    1: 'Running',
    2: 'Commute in bus',
    3: 'Eating using fork and spoon',
    4: 'Using mobile phone or texting',
    5: 'Working on laptop',
    6: 'Sitting',
    7: 'Washing hands',
    8: 'Eating with hand',
    9: 'Conversing while sitting',
    10: 'Elevator',
    11: 'Opening door',
    12: 'Standing',
    13: 'Climbing upstairs',
    14: 'Jogging',
    15: 'Video chat while sitting',
    16: 'Sleeping',
    17: 'Walking with hand in pocket'#,
    # 18: 'Writing'
}


class Activity:
    def __init__(self, num):
        self.num = num
        self.acc_x_series = []
        self.acc_y_series = []
        self.acc_z_series = []

    def append_acc_data(self, x, y, z):
        self.acc_x_series.append(x)
        self.acc_y_series.append(y)
        self.acc_z_series.append(z)

    def append_acc_data_series(self, x_series, y_series, z_series):
        self.acc_x_series += x_series
        self.acc_y_series += y_series
        self.acc_z_series += z_series


def read_all_files(data_path='../dataset/CC2650/'):
    """

    :param data_path: The folder where all the text files of the dataset are
    :return: extracted activities from dataset with complete recorded time series
    """

    files = []
    # r = root, d = directories, f = files
    for r, d, f in os.walk(data_path):
        for file in f:
            if '.csv' in file:
                files.append(os.path.join(r, file))

    print('reading the following files:')

    # recorded_activities_num = 0
    # for f in files:
    #     print(f)
    #     recorded_activities_num += count_activities_num(f)
    #
    # print('total recorded activities: ', recorded_activities_num)

    recorded_activities = []
    for f in files:
        print(f)
        recorded_activities += extract_activities(f)

    print('total recorded activities: ', len(recorded_activities))

    return recorded_activities


def visualize_all_files(data_path='../dataset/CC2650/'):  # path containing all files of dataset
    """
    Draws plots of activity time series

    :param data_path: The folder where all the text files of the dataset are
    """

    recorded_activities = read_all_files(data_path)

    record_num = 0
    for activity in recorded_activities:
        record_num += 1

        try:
            print(len(activity.acc_x_series) / 36, num_to_activity[activity.num])
        except KeyError:
            pass

        for axis in ['x', 'y', 'z']:
            plot_activity_data(activity, record_num, axis)


def count_activities_num(file_addr):
    """

    :param file_addr: Path of one data file
    :return: number of activities in the file
    """
    with open(file_addr, 'r') as file:
        activities = 0

        reader = csv.reader(file)
        next(reader, None)  # skip the headers

        previous_activity = -10
        for row in reader:
            # print(', '.join(row))
            if int(row[-1]) != previous_activity:
                previous_activity = int(row[-1])
                activities += 1

        return activities


def extract_activities(file_addr):
    """

    :param file_addr: file address
    :return: extracted activities
    """
    with open(file_addr, 'r') as file:
        activities = []

        reader = csv.reader(file)
        next(reader, None)  # skip the headers

        previous_activity_num = -10
        for row in reader:
            # activity_num = int(row[-1])
            activity_num = int(float(row[-1]))  # int(float()) can handle 0.0 but int() can't
            if activity_num != previous_activity_num:
                previous_activity_num = activity_num

                activities.append(Activity(activity_num))

            activities[-1].append_acc_data(float(row[1]), float(row[2]), float(row[3]))

        return activities


def plot_activity_data(activity, record_num, axis='x'):
    """
    Plots one of the axises of time series of an activity

    :param activity: activity which is going to be plotted
    :param record_num: index of this activity in list of all activities
    :param axis: x, y or z
    """
    plt.clf()

    series = activity.acc_x_series
    if axis == 'y':
        series = activity.acc_y_series
    elif axis == 'z':
        series = activity.acc_z_series
    elif axis != 'x':
        raise Exception('Axis should be either x, y or z')

    try:
        plt.plot(series[:500])
        plt.xlabel('activity: ' + str(num_to_activity[activity.num]))
        plt.ylabel('accelerometer ' + axis + ' series')
        plt.savefig('plots/series_initial_visualization_record_' +
                    str(record_num) + '_' + str(num_to_activity[activity.num]) + '_axis_' + axis + '.png')
    except KeyError:
        pass


def split_segments_of_activity(activity, split_series_max_len=360, overlap=0):
    """

    :param activity: input activity
    :param split_series_max_len: time series of the activity will be divided to small segments. This param indicates
           the maximum length of segments
    :param overlap: declares overlap percentage of output segments
    :return: activities with short segments
    """
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
        else:
            split_activities[-1].append_acc_data_series(
                activity.acc_x_series[i: i + split_series_max_len],
                activity.acc_y_series[i: i + split_series_max_len],
                activity.acc_z_series[i: i + split_series_max_len]
            )

    return split_activities


def split_segments_into_parts_with_same_len(data_path='../dataset/CC2650/', split_series_max_len=360,
                                            ignore_classes=[]):
    """
    calls split_segments_of_activity function for all the activities of dataset

    :param data_path: The folder where all the text files of the dataset are
    :param split_series_max_len: shows the maximum length of the output segments. (Original activity time series are
    divided into segments and length of these segments doesn't exceed this param)
    :param ignore_classes: includes activity types which are going to be ignored in this process. Data of these classes
           won't be fed to deep model.
    :return: activities with short segments
    """
    recorded_activities = read_all_files(data_path)

    split_activities = []

    for activity in recorded_activities:
        if activity.num not in ignore_classes:
            split_activities += split_segments_of_activity(activity, split_series_max_len)

    return split_activities


def get_our_dataset_labels_names(ignore_classes=[]):
    """

    :param ignore_classes: includes activity types which are going to be ignored in this process.
    :return: names of activities present in our dataset ( This func only works on our own dataset )
    """
    labels_names = []

    if -1 not in ignore_classes and 19 - 1 not in ignore_classes:  # todo: clean this code
        labels_names.append(num_to_activity[-1])

    for i in range(0, len(num_to_activity) - 1):
        if i not in ignore_classes:
            labels_names.append(num_to_activity[i])

    return labels_names


# if __name__ == '__main__':
#     # visualize_all_files()
#     small_observations = split_segments_into_parts_with_same_len()
#     print(len(small_observations))
#     print(len(small_observations[-1].acc_x_series))
