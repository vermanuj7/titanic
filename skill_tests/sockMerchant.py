#! /Users/anujverma/.conda/envs/jb_tutorial/bin/python
import math
import os
import random
import re
import sys
import collections as c

os.environ['OUTPUT_PATH'] = ''.join([os.getcwd(), '/sockMerchant_output.txt'])
print('Output Path: {}'.format(os.environ['OUTPUT_PATH']))


# Complete the sockMerchant function below.
def sockMerchant(n, ar):
    return sum([i // 2 for i in list(c.Counter(ar).values())])


if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')

    n = int(input("Type the number of socks:"))

    ar = list(map(int, input(" Input array of sock sizes separated by "
                             "space:").rstrip(

    ).split()))

    result = sockMerchant(n, ar)
    print('result:', result)
    fptr.write(str(result) + '\n')

    fptr.close()
