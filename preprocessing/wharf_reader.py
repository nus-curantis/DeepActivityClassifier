import os
from preprocessing.time_series_reader_and_visualizer import Activity, split_segments_of_activity
from preprocessing.data_to_rnn_input_transformer import normalized_rnn_input_train_test_

activity_to_num = {
    'use_telephone': 0,
    'drink_glass': 1,
    'eat_soup': 2,
    'descend_stairs': 3,
    'getup_bed': 4,
    'brush_teeth': 5,
    'standup_chair': 6,
    'walk': 7,
    'comb_hair': 8,
    'liedown_bed': 9,
    'sitdown_chair': 10,
    'climb_stairs': 11,
    'pour_water': 12,
    'eat_meat': 13
}


def read_all_files(data_path='../dataset/WHARF/Data/', split_series_max_len=360):
    files = []
    activity_names = []
    task_conductors = []
    # r = root, d = directories, f = files
    for r, d, f in os.walk(data_path):
        for file in f:
            if '.txt' in file:
                files.append(os.path.join(r, file))
                activity_names.append(os.path.join(r, file).split('-')[-2])
                task_conductors.append(os.path.join(r, file).split('-')[-1].split('.')[0])

    activities = []

    counter = 0
    for file in files:
        activities.append(extract_activity(file, activity_names[counter]))
        counter += 1

    for activity in activities:
        print(activity.num, len(activity.acc_x_series))
        print(activity.acc_x_series[0:10])

    split_activities = []
    for activity in activities:
        split_activities += split_segments_of_activity(activity, split_series_max_len=split_series_max_len)

    return activities, split_activities


def extract_activity(file_addr, activity_name):
    activity_num = activity_to_num[activity_name]
    activity = Activity(activity_num)

    with open(file_addr, 'r') as file:

        next(file, None)  # skip the headers
        for line in file:
            row = line.split(' ')
            if activity_name == 'brush_teeth':
                row = line.split(',')
            row[-1] = row[-1][:len(row[-1]) - 1]  # deleting \n from end of line

            activity.append_acc_data(float(row[0]), float(row[1]), float(row[2]))

        return activity


def normalized_wharf_rnn_input_train_test(data_path='../dataset/WHARF/Data/', split_series_max_len=360):
    _, split_activities = read_all_files(data_path, split_series_max_len=split_series_max_len)
    return normalized_rnn_input_train_test_(split_activities=split_activities,
                                            split_series_max_len=split_series_max_len)


# read_all_files()
# train_data, test_data, train_labels, test_labels = normalized_wharf_rnn_input_train_test()
# print(train_data.shape, test_data.shape, train_labels.shape, test_labels.shape)
