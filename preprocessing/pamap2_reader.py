

def read_all_files(data_path='../dataset/PAMAP2_Dataset/Protocol'):
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