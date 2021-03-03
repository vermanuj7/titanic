import math


def int2rom(numbr):
    out = 0
    exp = 0
    for i in range(10):
        if numbr / math.pow(10, i) > 1:
            exp = i

    # print('The power of {} is {}'.format(numbr, exp))
    r_dict = {}

    for i in range(exp, -1, -1):
        r_dict[i] = math.floor((numbr % math.pow(10, i + 1)) / math.pow(10, i))

    # print('The decimal representation of {} is {}'
    #       .format(numbr, repr(r_dict)))

    roman_dict = {1000: 'M'
        , 900: 'CM'
        , 800: 'CM'
        , 700: 'DCC'
        , 600: 'DC'
        , 500: 'D'
        , 400: 'CD'
        , 300: 'CCC'
        , 200: 'CC'
        , 100: 'C'
        , 90: 'XC'
        , 80: 'LXXX'
        , 70: 'LXX'
        , 60: 'LX'
        , 50: 'L'
        , 40: 'XL'
        , 30: 'XXX'
        , 20: 'XX'
        , 10: 'X'
        , 9: 'IX'
        , 8: 'VIII'
        , 7: 'VII'
        , 6: 'VI'
        , 5: 'V'
        , 4: 'IV'
        , 3: 'III'
        , 2: 'II'
        , 1: 'I'
                  }

    r_name = ' '.join(
        [roman_dict[math.pow(10, k) * v] for k, v in r_dict.items()])

    # print('The roman representation of {} is {}'
    #       .format(numbr, r_name))
    return r_name


# int2rom(5)
# print('\n-----\n')
# int2rom(39)
# print('\n-----\n')
# int2rom(298)
# print('\n-----\n')
# int2rom(1984)
