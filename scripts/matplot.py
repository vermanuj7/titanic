import matplotlib.pyplot as plt
import numpy as np
import matplotlib.style as mplstyle

mplstyle.use(['dark_background', 'ggplot', 'fast'])


def my_plotter(axx, x, y, param_dict):
    out = axx.plot(x, y, **param_dict)
    return out

"""
data1, data2, data3, data4 = np.random.randn(4, 100)
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
my_plotter(ax1, data1, data2, {'marker': 'x'})
my_plotter(ax2, data3, data4, {'marker': 'o', 'color': 'green'})
my_plotter(ax3, data1, data2, {'marker': '^'})
my_plotter(ax4, data3, data4, {'marker': 'o', 'color': 'cyan'})
fig.show()
"""
# TODO: Delete this file as this file contains duplicated code
