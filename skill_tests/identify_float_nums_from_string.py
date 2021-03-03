import re
import sys

line_num = 0
for line in sys.stdin:

    if line_num > 0:

        pattern = r"[+-.]*[0-9]*[.]*[0-9]+"
        if re.search(pattern, line) and len(re.split(r'\.', line)) == 2:
            print('True')
        else:
            print("False")
