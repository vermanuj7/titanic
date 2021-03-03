#!/bin/python3

import math
import os
import random
import re
import sys
import numpy as np


# Complete the hourglassSum function below.
def hourglassSum(arr):
    arr2 = np.array(arr)
    print(arr2)
    rows = len(arr)
    cols = len(arr[0])

    h_glasses = (rows - 2) * (cols - 2)
    print('h_glasses:', h_glasses)

    sums = []
    for i in range(1, rows, 1):
        for j in range(1, cols, 1):
            summ = arr2[i, j] + arr2[i - 1, j - 1] + arr2[i - 1, j]
            + arr2[i - 1, j + 1] + arr2[i + 1, j - 1] + arr2[i + 1, j] + arr2[
                i + 1, j + 1]

            sums.append(summ)

    return max(sums)


if __name__ == '__main__':
    # fptr = open(os.environ['.'], 'w')

    arr = []

    for _ in range(6):
        arr.append(list(map(int, input().rstrip().split())))

    print(arr)
    result = hourglassSum(arr)

    # fptr.write(str(result) + '\n')

    # fptr.close()

;
