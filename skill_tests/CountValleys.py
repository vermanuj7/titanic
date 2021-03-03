import logging
import sys
import time

# logging level set to INFO
logging.basicConfig(format='%(message)s', level=logging.INFO)
Log = logging.getLogger(__name__)
Log.disabled = True
total_steps = 8
actual_path = 'UDDDUDUU'


def countValleys(steps, path):
    valley_count = 0
    altitude = 0
    alt_trace = []
    Log.info("The length of the trail is: {}".format(steps))
    for idx, i in enumerate(list(path)):
        Log.info('-------------------')
        Log.info('The list index is:{}'.format(idx))
        Log.info('The list element is:{}'.format(i))
        if i == 'U':
            altitude = altitude + 1
        else:
            altitude = altitude - 1
        Log.info("Altitude is: {}".format(altitude))

        alt_trace.insert(idx, altitude)

    size = len(alt_trace)
    idx_list = [idx for idx, val in enumerate(alt_trace) if val == 0]
    Log.info("The split around 0 index list is: {}".format(idx_list))
    Log.info("The original alt_trace is: {}".format(alt_trace))
    zippie = zip([0] + idx_list,
                 idx_list +
                ([size] if idx_list[-1] != size else [])
                 )

    zip_list = list(zippie)
    Log.info("The zip list is: {}".format(zip_list))
    sub_traces = [alt_trace[i:j] for i, j in zip([0] + idx_list,
                 idx_list +
                ([size] if idx_list[-1] != size else [])
                 )]

    for sub in sub_traces:
        if sum(sub) < 0:
            valley_count = valley_count+1





    Log.info("The sub-lists are: {}".format(sub_traces))
    return valley_count


result = countValleys(total_steps, actual_path)

print('The final output is:', result)
