import math
import random
import string

import numpy as np
import pandas as pd


def random_df(rows=5, cols=5, row_min=0, col_min=0, str_frac=0):
    """

    :rtype: pd.DataFrame()
    """
    index_id = [i for i in range(row_min, row_min + rows)]
    col_id = [i for i in range(col_min, col_min + cols)]

    col_values = []
    for col in range(cols):
        if col < math.floor(str_frac * cols):
            col_values.append([random_string(5) for _ in range(rows)])
        else:
            col_values.append(np.random.randint(low=3, high=10, size=rows))

    return pd.DataFrame(index=index_id, data=dict(zip(col_id, col_values)))


def random_string(size=8):
    """

    :type size: integer
    """

    basket = string.ascii_lowercase + string.ascii_uppercase + string.digits
    return "".join(random.choices(basket, k=size))


def random_series(size=8, val_type='i'):
    if val_type == 'i':
        return pd.Series(np.random.randint(low=1, high=1000, size=size))
    else:
        return pd.Series([random_string(5) for _ in range(size)])


def my_plotter(axx, x, y, param_dict):
