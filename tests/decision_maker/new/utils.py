import pandas as pd
def read_waypoints(filename):
    pass

def find_nearest_waypoint(df, timestamp):
    closest_row_index = (df['time'] - timestamp).abs().idxmin()
    waypoint = df.loc[closest_row_index]["waypoint_location"]
    return waypoint
def parse_waypoint_file(filename="../../status_tracker/waypoints.csv"):
    # filename = "../../status_tracker/waypoints.csv"
    df = pd.read_csv(filename)
    drone_ids = df["drone_id"].unique()

    grouped = df.groupby("drone_id")
    for group_name, group_df in grouped:
        drone_id = group_name
        waypoint = find_nearest_waypoint(group_df, 70)
        print(waypoint)
    group = grouped.get_group(3)
    # print(group)



def calculate_determine_sequence(waypoint_sequence_1, waypoint_sequence_2, access_points):
    pass

def calculate_effective_transmission_time(devices, access_points):
    n = len(devices)
    continous_threshold = 2
    effective_time_matrix = [[1 for _ in range(n)] for _ in range(n)]
    for i in range(n):
        for j in range(i+1, n):
            if devices[i]["type"] == "IoT-mobile":
                filename = ""
                waypoint_sequences_1 = read_waypoints(filename)
            if devices[j]["type"] == "IoT-mobile":
                filename = ""
                waypoint_sequences_2 = read_waypoints(filename)
            calculate_determine_sequence(waypoint_sequences_1, waypoint_sequences_2, access_points)
            effective_time_matrix[i][j] = 1

    return effective_time_matrix

parse_waypoint_file()