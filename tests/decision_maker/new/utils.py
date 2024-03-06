import pandas as pd
import math

def read_waypoints(filename):
    pass

def dist(point1, point2):
    return math.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2 + (point1[2] - point2[2]) ** 2)

def mobile_to_drone_match(mobile_ids, drone_ids):
    dict = {}
    for idx, mobile_id in enumerate(mobile_ids):
        dict[mobile_id] = drone_ids[idx]
    return dict



def find_nearest_waypoint_by_time(df, timestamp):
    closest_row_index = (df['time'] - timestamp).abs().idxmin()
    waypoint = df.loc[closest_row_index]["waypoint_location"]
    return waypoint

def find_nearest_ap(waypoint, access_points):
    nearest_ap = None
    minimun_distance = float("inf")
    for ap_id, ap_pos in enumerate(access_points):
        if dist(waypoint, ap_pos) < minimun_distance:
            nearest_ap = ap_id
            minimun_distance = dist(waypoint, ap_pos)
    return nearest_ap, minimun_distance


def parse_waypoint_file(filename="../../status_tracker/waypoints.csv"):
    # filename = "../../status_tracker/waypoints.csv"
    df = pd.read_csv(filename)
    drone_ids = df["drone_id"].unique()

    grouped = df.groupby("drone_id")
    # for group_name, group_df in grouped:
    #     drone_id = group_name

    return grouped
    # group = grouped.get_group(3)
    # print(group)

def split_timesteps(time_range, time_unit):
    start_time = time_range[0]
    end_time = time_range[1]
    time_steps = list(range(start_time, end_time, time_unit))
    return time_steps

def find_waypoint_df_by_id(grouped, dev_id, epoch=0):
    group = grouped.get_group(dev_id)
    start_row = 0
    end_row = -1
    pre_idx = 0
    for idx, row in group.iterrows():
        if idx == 0:
            continue
        if idx - pre_idx == 1:
            end_row = idx
            pre_idx = idx
        else:
            break
    group = group.iloc[start_row: end_row]
    # return the default first epoch
    return group

def calculate_determine_sequence(timesteps, df1, df2, access_points):
    if df1 is None and df2 is None:
        return [1 for _ in range(len(timesteps))]
    determine_sequence = [0 for _ in range(len(timesteps))]
    for idx, timestamp in enumerate(timesteps):
        ap1 = None
        ap2 = None
        if df1 is None and df2 is not None:
            ap1 = 1
            w2 = find_nearest_waypoint_by_time(df2, timestamp)
            ap2 = find_nearest_ap(w2, access_points)
        elif df1 is not None and df2 is None:
            ap2 = 1
            w1 = find_nearest_waypoint_by_time(df1, timestamp)
            ap1 = find_nearest_ap(w1, access_points)
        elif df1 is not None and df2 is not None:
            w1 = find_nearest_waypoint_by_time(df1, timestamp)
            w2 = find_nearest_waypoint_by_time(df2, timestamp)
            ap1 = find_nearest_ap(w1, access_points)
            ap2 = find_nearest_ap(w2, access_points)

        if ap1 is not None and ap2 is not None:
            determine_sequence[idx] = 1
    return determine_sequence

def find_time_range(grouped_data):
    # find the first epoch of each group
    min_time_all = float("inf")
    max_time_all = float("-inf")
    for group_name, group_df in grouped_data:
        group_epoch0 = find_waypoint_df_by_id(grouped_data, group_name)
        min_time = group_epoch0["time"].min()
        max_time = group_epoch0["time"].max()
        if min_time < min_time_all:
            min_time_all = min_time
        if max_time > max_time_all:
            max_time_all = max_time
    return [min_time_all, max_time_all]


def calculate_effective_rate(determine_sequence, continous_threshold, time_unit):
    current_length = 0
    total_length = 0
    for item in determine_sequence:
        if item == 1:
            current_length += 1
        else:
            if current_length >= continous_threshold:
                total_length += (current_length-1)*time_unit
            current_length = 0

    if current_length >= continous_threshold:
        total_length += (current_length-1)*time_unit
    return total_length


def calculate_effective_transmission_time(devices, access_points):
    n = len(devices)
    continous_threshold = 2
    effective_threshold = 0.6
    effective_time_matrix = [[1 for _ in range(n)] for _ in range(n)]
    grouped_data = parse_waypoint_file()

    m = grouped_data.size()
    drone_ids = range(m)

    mobile_dev_ids = []
    for i in range(n):
        type = devices[i]["type"]
        if type == "IoT-mobile":
            mobile_dev_ids.append(i)

    mapping = mobile_to_drone_match(mobile_dev_ids, drone_ids)

    timeunit = 2
    time_range = find_time_range(grouped_data)
    epoch = time_range[1] - time_range[0]
    timesteps = split_timesteps(time_range, timeunit)
    for i in range(n):
        for j in range(i+1, n):
            type1 = devices[i]["type"]
            type2 = devices[j]["type"]

            if type1 == "IoT-mobile":
                df1 = find_waypoint_df_by_id(grouped_data, i)
            else:
                df1 = None
            if type2 == "IoT-mobile":
                df2 = find_waypoint_df_by_id(grouped_data, i)
            else:
                df2 = None

            determine_sequence = calculate_determine_sequence(timesteps, df1, df2, access_points)
            effective_time = calculate_effective_rate(determine_sequence, continous_threshold, timeunit)
            effective_rate = effective_time / epoch
            if effective_rate >= effective_threshold:
                effective_time_matrix[i][j] = effective_time
            else:
                effective_time_matrix[i][j] = 0
    return effective_time_matrix

# grouped_data = parse_waypoint_file()
# df1 = find_waypoint_df_by_id(grouped_data, 0)
# calculate_effective_transmission_time()
# sequence = [1,1,1,1,1,0,0,0,1,0,1,1,1, 0, 1,0]
# rate = calculate_effective_rate(sequence, 2)
# print(rate)