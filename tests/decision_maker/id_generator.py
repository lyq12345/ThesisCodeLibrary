global_op_id = 0

def generate_operator_id():
    global global_op_id
    new_id = global_op_id
    global_op_id += 1
    return new_id

