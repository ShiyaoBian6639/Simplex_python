import numpy as np


def read_mps(file_name):
    instance = open(file_name)
    problem_name = ""
    row_name_idx_dict = dict()  # initialize a dictionary recording row names and their corresponding indices
    col_name_idx_dict = dict()  # initialize a dictionary recording column names and their corresponding indices
    while instance.seekable():
        line = next(instance)
        str_list = line_formatter(line)
        if str_list[0] == "NAME":
            problem_name = str_list[1]
        if str_list[0] == "ROWS":
            row_name_idx_dict, row_name_sense_dict, objective_name, num_row = read_mps_row(instance)
            A, col_name_idx_dict, variable_type, variable_bound = read_mps_column(instance, row_name_idx_dict, num_row)
        print(str_list)
        if str_list and str_list[0] == "ENDATA":
            break
    return problem_name, row_name_idx_dict, row_name_sense_dict, objective_name, num_row,  A, col_name_idx_dict, variable_type, variable_bound


def line_formatter(line):
    """
    remove redundant spaces and \n
    :param line: 'NAME          markshare_4_0\n'
    :return: ['NAME', 'markshare_4_0']
    """
    return line.strip().split()


def read_mps_row(instance):
    """
    :param instance:
    :return:
    """
    row_name_idx_dict = dict()  # initialize a dictionary recording row names and their corresponding indices
    row_name_sense_dict = dict()  # initialize a dictionary recording constraint senses
    objective_name = ""

    row_count = 0
    line = line_formatter(next(instance))
    while line[0] != "COLUMNS":
        constraint_sense = line[0]  # get constraint sense
        row_name = line[1]  # get constraint name
        if constraint_sense == "N":  # "N" indicates objective function
            objective_name = row_name  # assign objective function
        row_name_idx_dict[row_name] = row_count
        row_name_sense_dict[row_name] = constraint_sense
        row_count += 1
        line = line_formatter(next(instance))
    return row_name_idx_dict, row_name_sense_dict, objective_name, row_count


def read_mps_column(instance, row_name_idx_dict, num_row):
    """
    variable type: 0 continuous
                   1 integer
    :param instance:
    :param row_name_idx_dict:
    :param num_row:
    :return:
    """
    # constants
    is_integer = 0
    col_count = 0
    max_num_col = 1000

    # return variables

    col_name_idx_dict = dict()  # initialize a dictionary recording col names and their corresponding indices
    variable_bound = np.zeros((max_num_col, 2), dtype=int)
    variable_type = np.zeros((max_num_col, 1), dtype=int)
    A = np.zeros((num_row, max_num_col), dtype=float)

    line = line_formatter(next(instance))
    while line[0] != "RHS":
        if col_count == max_num_col:
            A = extend_column(A)
            variable_bound = extend_column(variable_bound)
            variable_type = extend_column(variable_type)
            max_num_col *= 2
        if len(line) == 3 and line[2] == "'INTORG'":
            is_integer = 1
            line = line_formatter(next(instance))
        if len(line) == 3 and line[2] == "'INTEND'":
            is_integer = 0
            line = line_formatter(next(instance))

        col_name = line[0]

        if col_name in col_name_idx_dict:
            col_idx = col_name_idx_dict[col_name]
        else:
            col_idx = col_count
            col_name_idx_dict[col_name] = col_idx
            col_count += 1

        if is_integer:
            variable_type[col_idx] = 1
            variable_bound[col_idx][1] = 1
        else:
            variable_type[col_idx] = 0

        row_idx = 0  # initialize row index
        for j in range(1, len(line)):

            if j % 2:
                row_name = line[j]
                row_idx = row_name_idx_dict[row_name]
            else:
                value = line[j]
                A[row_idx, col_idx] = value
        line = line_formatter(next(instance))
    return A[:, : col_count - 1], col_name_idx_dict, variable_type, variable_bound


def extend_column(arr):
    num_row, num_col = arr.shape
    tmp = np.zeros((num_row, num_col * 2))
    tmp[:, :num_col] = arr
    return tmp


