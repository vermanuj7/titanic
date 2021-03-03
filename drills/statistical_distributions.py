from numpy.random import seed
from numpy.random import randint
from numpy import mean
# seed the random number generator, so that the experiment is #replicable
seed(1)
# generate a sample of men's weights
weights = randint(60, 90, 50)
print(weights)
print('The average weight is {} kg'.format(mean(weights)))

import matplotlib.pyplot as plt
plt.clf()
plt.hist([mean(randint(60, 90, 500000)) for _ in range(1000)])
plt.show()
