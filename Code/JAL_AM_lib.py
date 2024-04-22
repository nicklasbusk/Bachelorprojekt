import numpy as np
from model_lib import *


# Update model based on actions


def Q_func_JAL_AM(p_curr_idx, s_curr_idx, predicted_action_index, alpha, gamma, Q_table, price_grid, s_next):
