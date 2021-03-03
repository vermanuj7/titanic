import numpy as np
import matplotlib.pyplot as plt

x = np.random.randint(low=5, high=51, size=50)
y = x + np.random.randint(1, 2, size=x.size)
data = np.column_stack((x, y))

fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(8, 4))

ax1.scatter(x=x, y=y, marker='^', c='r', edgecolor='r')
ax1.set_title('Scatter: $x$ versus $y$')
ax1.set_xlabel("$x$")
ax1.set_ylabel("$y$")

ax2.hist(data, bins=np.arange(data.min(), data.max()), label=('x', 'y'))
ax2.legend(loc=(0.15, 0.8))
ax2.set_title('Frequencies of $x$ and $y$')
ax2.yaxis.tick_right()

fig.show()
