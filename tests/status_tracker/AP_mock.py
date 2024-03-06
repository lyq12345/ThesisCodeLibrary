def determine_ap_positions(x1, x2, y1, y2, z1, z2, r):
    x_positions = list(range(x1, x2, r))
    y_positions = list(range(y1, y2, r))
    z_positions = list(range(z1, z2, r))
    return [(x, y, z) for x in x_positions for y in y_positions for z in z_positions]

def determine_number_of_aps(x1, x2, y1, y2, z1, z2, r):
    x_length = x2-x1
    y_length = y2-y1
    z_length = z2-z1

    num_x = x_length // r
    num_y = y_length // r
    num_z = z_length // r

    return num_x*num_y*num_z

def generate_access_points(x1, x2, y1, y2, z1, z2, r):
    ap_positions = determine_ap_positions(x1, x2, y1, y2, z1, z2, r)
    return ap_positions
