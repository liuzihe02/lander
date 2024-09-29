import numpy as np


def get_t(t_max, dt):
    """
    Generate a array of time intervals, and the total number of intervals
    Based on the maximum time, and width of each step
    """
    t_array = np.arange(0, t_max, dt)
    n = len(t_array)
    return t_array, n
