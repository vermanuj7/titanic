import logging

# logging level set to INFO
import time

logging.basicConfig(format='%(message)s', level=logging.INFO)
log = logging.getLogger(__name__)


def jumpingOnClouds(c):
    tracer_id = 0
    min_jumps = 0
    start = 0
    end = len(c) - 1

    # logger begin
    log.info('begin - min jumps value: {}'.format(min_jumps))
    log.info("begin - tracer at index: {}".format(tracer_id))

    while tracer_id < end:
        # time.sleep(5)
        print("The value of end index is : {}".format(end))
        print(" tracer_id {} {} jumps {}".format(tracer_id, '\n', min_jumps))

        if end - tracer_id >= 2 and c[tracer_id + 2] == 0:
            min_jumps = min_jumps + 1
            tracer_id = tracer_id + 2
        elif end - tracer_id >= 1 and c[tracer_id + 1] == 0:
            min_jumps = min_jumps + 1
            tracer_id = tracer_id + 1
        else:
            pass

    # logger end
    log.info('min jumps value: {}'.format(min_jumps))
    log.info("tracer at index: {}".format(tracer_id))

    return min_jumps


# TODO: delete this line later
result = jumpingOnClouds([0, 0, 1, 0, 0, 1, 0])

print('The final output is:', result)
